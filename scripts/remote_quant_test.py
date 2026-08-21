#!/usr/bin/env python3
"""
remote_quant_test.py — 在远程设备上实测指纹方法对"量化"的识别能力:

  A) Qwen-0.8B (L8): weight-only RTN(round-to-nearest, per-row absmax) 量化,
     —— 位宽扫描 bits ∈ {2,3,4,5,6,8}
     —— 量化层比例扫描 frac ∈ {0.1,0.25,0.5,0.75,1.0} @ int4 (连续单调曲线)
  B) Llama-3.2-1B / Gemma-3-1b: int8 / int4 单点 (跨模型族一致性)

指纹统计量 = 论文主统计量 D_KL = mean_x KL( p_0(·|x) || p_q(·|x) ),
  同族内, 同一模型的 clean 指纹 vs 量化后指纹 (无跨族 tokenizer 混淆)。
  同时报:
    - KL_full  : 全词表 KL (logp clamp at -30, 白盒口径)
    - KL_trunc : top-20 截断 KL (tail-bucket, 黑盒口径)
  量化: weight-only RTN, 保留 input embeddings 与 lm_head 为 fp16(贴近真实 serving),
        逐行(absmax)对称量化, 反量化回 bf16 前向 → 完全可控、确定性。
"""
from __future__ import annotations
import argparse, copy, json, os, re, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from align import Aligner

TARGET = "deepseek-ai/DeepSeek-V4-Flash-0731"
Q = "Qwen/Qwen3.5-0.8B"
L = "unsloth/Llama-3.2-1B"
G = "unsloth/gemma-3-1b-pt"
REFERENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "The committee approved the new budget proposal this morning.",
    "She walked slowly through the old town under the rain.",
]

# ---------------------------------------------------------------- KL utils (同 remote_lora_test)
@torch.no_grad()
def next_token_logp(model, input_ids):
    """[Vfull] 序列末尾 next-token 的 log_softmax。"""
    logits = model(input_ids).logits[0, -1].float()
    return torch.log_softmax(logits, dim=-1)


def full_kl(lp0, lpr, floor=-30.0):
    p0 = lp0.exp()
    return (p0 * (lp0.clamp(min=floor) - lpr.clamp(min=floor))).sum()


def truncated_kl(lp0, lpr, k=20):
    top = torch.cat([torch.topk(lp0, k).indices, torch.topk(lpr, k).indices]).unique()
    q0 = lp0[top].exp(); qr = lpr[top].exp()
    tail0 = (1.0 - q0.sum()).clamp_min(0.0); tailr = (1.0 - qr.sum()).clamp_min(0.0)
    kl = (q0 * (lp0[top] - lpr[top])).sum()
    if tail0 > 0:
        kl = kl + tail0 * (torch.log(tail0) - torch.log(tailr + 1e-30))
    return kl


def mean_kl(lp_cands, lp_ref, mode, floor=-30.0, k=20):
    n = lp_ref.shape[0]
    tot = 0.0
    for i in range(n):
        tot += (full_kl(lp_ref[i], lp_cands[i], floor).item() if mode == "full"
                else truncated_kl(lp_ref[i], lp_cands[i], k).item())
    return tot / n


# ---------------------------------------------------------------- weight-only RTN quantizer
def _layer_idx(name: str) -> int | None:
    """从参数名解析 transformer 层索引 (model.layers.<i>.*)。非层参数 -> None。"""
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def rtn_quantize(model, bits: int, per: str = "row",
                 quant_emb: bool = False, frac: float = 1.0):
    """weight-only RTN: 逐行/逐张量 absmax 对称量化到 bits 位, 反量化回 bf16。
    默认保留 embeddings 与 lm_head 为 fp16(贴近真实 serving); frac<1 只量化
    前 frac*n_layers 层块。返回 (deepcopy 的量化模型, 量化张量数)。"""
    m = copy.deepcopy(model)
    m.eval()
    n_layers = getattr(m.config, "num_hidden_layers", None)
    nq = 0
    for name, p in m.named_parameters():
        if p.ndim < 2:
            continue
        low = name.lower()
        if not quant_emb and ("embed" in low or "lm_head" in low):
            continue
        li = _layer_idx(name)
        if li is not None and n_layers and li >= frac * n_layers:
            continue
        w = p.detach().float()
        amax = w.abs().amax(dim=-1, keepdim=True) if per == "row" else w.abs().amax()
        qmax = float(2 ** (bits - 1) - 1)
        scale = (amax / qmax).clamp_min(1e-8)
        q = (w / scale).round().clamp(-2 ** (bits - 1), qmax) * scale
        p.data = q.to(p.dtype)
        nq += 1
    return m, nq


# ---------------------------------------------------------------- per-model run
def run_model(tag, model_id, layer, dtok_set, args, calib):
    dev = "cuda"
    print(f"\n========== [{tag}] {model_id} (L{layer}) ==========", flush=True)
    aligner = Aligner(model_id, layer=layer, device=dev, dtype=torch.bfloat16)
    aligner.model.eval()
    shared = set(aligner.tok.get_vocab().keys()) & dtok_set
    aligner.restrict_vocab(shared)
    print(f"  shared vocab with DeepSeek: {len(shared)}", flush=True)

    probes, probe_strs = [], []
    for ref in REFERENCES:
        ids = aligner.tok(ref, return_tensors="pt")["input_ids"].to(aligner.vocab.device)
        cands = aligner.sample_near_synonyms(ids, M=args.m, top_k=6, k_neighbors=200)
        probes += cands
        probe_strs += aligner.decode(cands)
    n = len(probes)
    print(f"  probes: {n}", flush=True)

    LP0 = torch.stack([next_token_logp(aligner.model, p) for p in probes])  # [n, V]

    # 零噪声 sanity
    reLP0 = torch.stack([next_token_logp(aligner.model, p) for p in probes])
    zero = {"KL_full": mean_kl(reLP0, LP0, "full"),
            "KL_trunc20": mean_kl(reLP0, LP0, "trunc", k=20)}
    print(f"  zero-noise sanity: D_KL(full)={zero['KL_full']:.6f}  "
          f"D_KL(trunc20)={zero['KL_trunc20']:.6f}", flush=True)

    def eval_quant(mq):
        LPs = torch.stack([next_token_logp(mq, p) for p in probes])
        return {"KL_full": mean_kl(LPs, LP0, "full"),
                "KL_trunc20": mean_kl(LPs, LP0, "trunc", k=20)}

    res = {"tag": tag, "model_id": model_id, "layer": layer, "n_probes": n,
           "zero_noise": zero, "probes": probe_strs}

    # 位宽扫描
    bit_rows = []
    for bits in args.bits:
        t0 = time.time()
        mq, nq = rtn_quantize(aligner.model, bits=bits, per=args.per, quant_emb=args.quant_emb)
        r = eval_quant(mq)
        bit_rows.append({"bits": bits, **r, "n_quant": nq})
        print(f"  bits={bits:2d}  D_KL(full)={r['KL_full']:9.4f}  "
              f"D_KL(trunc20)={r['KL_trunc20']:9.4f}  ({time.time()-t0:.0f}s)", flush=True)
        del mq; torch.cuda.empty_cache()
    res["bit_sweep"] = bit_rows

    # 量化层比例扫描 @ int4 (连续单调曲线)
    frac_rows = []
    for fr in args.fracs:
        t0 = time.time()
        mq, nq = rtn_quantize(aligner.model, bits=args.frac_bits, per=args.per,
                              quant_emb=args.quant_emb, frac=fr)
        r = eval_quant(mq)
        frac_rows.append({"frac": fr, **r, "n_quant": nq})
        print(f"  frac={fr:4.2f}@{args.frac_bits}bit  D_KL(full)={r['KL_full']:9.4f}  "
              f"D_KL(trunc20)={r['KL_trunc20']:9.4f}  ({time.time()-t0:.0f}s)", flush=True)
        del mq; torch.cuda.empty_cache()
    res["frac_sweep"] = frac_rows

    return res


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=20)
    ap.add_argument("--bits", default="2,3,4,5,6,8", help="Qwen 位宽扫描")
    ap.add_argument("--fracs", default="0.1,0.25,0.5,0.75,1.0", help="Qwen int4 层比例扫描")
    ap.add_argument("--frac-bits", type=int, default=4)
    ap.add_argument("--per", default="row", choices=["row", "tensor"])
    ap.add_argument("--quant-emb", action="store_true", help="也量化 embeddings/lm_head")
    ap.add_argument("--cross-bits", default="4,8", help="Llama/Gemma 单点位宽")
    ap.add_argument("--out", default="../data/quant_test_results.json")
    args = ap.parse_args()
    # 逗号分隔参数解析为数值列表 (run_model 里按位宽/比例迭代)
    args.bits = [int(x) for x in args.bits.split(",")]
    args.fracs = [float(x) for x in args.fracs.split(",")]
    args.cross_bits = [int(x) for x in args.cross_bits.split(",")]

    dev = "cuda"
    dt = AutoTokenizer.from_pretrained(TARGET, trust_remote_code=True)
    dtok_set = set(dt.get_vocab().keys())

    res = {
        "method": "weight-only RTN per-row absmax, embeddings/lm_head fp16",
        "args": vars(args),
    }
    # resume: 若已有部分结果, 跳过已完成模型
    if os.path.exists(args.out):
        try:
            old = json.load(open(args.out))
            for k in ("qwen", "cross_models"):
                if k in old:
                    res[k] = old[k]
            print("[resume] loaded existing partial results", flush=True)
        except Exception as e:
            print(f"[resume] ignoring existing ({e})", flush=True)

    def save():
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)

    # Qwen 主实验
    if "qwen" not in res:
        res["qwen"] = run_model("qwen", Q, 8, dtok_set, args, calib=None)
        save()
        print(f"\n[checkpoint] wrote {args.out}", flush=True)

    # 跨模型族一致性
    cross = res.get("cross_models", [])
    done_tags = {c["tag"] for c in cross}
    want = args.cross_bits  # 已在上面解析为 list[int]
    for tag, mid, layer in [("llama", L, 8), ("gemma", G, 8)]:
        if tag in done_tags:
            continue
        r = run_model(tag, mid, layer, dtok_set, args, calib=None)
        r["cross_points"] = [row for row in r["bit_sweep"] if row["bits"] in want]
        r["bit_sweep"] = None  # 不重复存整条
        cross.append(r)
        res["cross_models"] = cross
        save()
        print(f"\n[checkpoint] cross-model {tag} done", flush=True)

    save()
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
