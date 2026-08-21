#!/usr/bin/env python3
"""
remote_lora_test.py — 在远程设备上实测指纹方法对两种"篡改"的识别能力:

  A) Qwen 被 LoRA 微调过(同一族, 结构微扰)
  B) Qwen 被偷换成 Llama(跨族, tokenizer 不匹配)

指纹统计量 = 论文主统计量 D_KL = mean_x KL( p_0(·|x) || p_cand(·|x) ),
  逐探针、逐探针平均; 同时报:
    - KL_full  : 全词表 KL (logp clamp at -30)
    - KL_trunc : top-20 截断 KL (tail-bucket), 即黑盒口径
  灵敏度: 对 LoRA 的 ΔW 按尺度 s 插值 (W_s = W0 + s·ΔW), 得 D_KL(s),
          拟合二次, 并用 Qwen L8 高斯噪声标定映射到"等价相对噪声尺度 r"。
  跨族  : 把探针按字符串重切到 Llama, 在跨族共享 surface token 集上算 KL。
"""
from __future__ import annotations
import argparse, copy, json, os, time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from align import Aligner

TARGET = "deepseek-ai/DeepSeek-V4-Flash-0731"
Q = "Qwen/Qwen3.5-0.8B"
L = "unsloth/Llama-3.2-1B"
REFERENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "The committee approved the new budget proposal this morning.",
    "She walked slowly through the old town under the rain.",
]
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# ---------------------------------------------------------------- KL utils
@torch.no_grad()
def next_token_logp(model, input_ids):
    """[Vfull] 序列末尾 next-token 的 log_softmax。"""
    logits = model(input_ids).logits[0, -1].float()
    return torch.log_softmax(logits, dim=-1)


def full_kl(lp0, lpr, floor=-30.0):
    """全词表 KL(白盒), logp clamp 防止下溢。"""
    p0 = lp0.exp()
    return (p0 * (lp0.clamp(min=floor) - lpr.clamp(min=floor))).sum()


def truncated_kl(lp0, lpr, k=20):
    """top-k 截断 KL(黑盒): 两分布 top-k 并集 + 一个 tail bucket。"""
    top = torch.cat([torch.topk(lp0, k).indices, torch.topk(lpr, k).indices]).unique()
    q0 = lp0[top].exp(); qr = lpr[top].exp()
    tail0 = (1.0 - q0.sum()).clamp_min(0.0); tailr = (1.0 - qr.sum()).clamp_min(0.0)
    kl = (q0 * (lp0[top] - lpr[top])).sum()
    if tail0 > 0:
        kl = kl + tail0 * (torch.log(tail0) - torch.log(tailr + 1e-30))
    return kl


def mean_kl(lp_cands, lp_ref, mode, floor=-30.0, k=20):
    """lp_ref: [n,V], lp_cands: [n,V] -> mean per-probe KL(ref || cand) [float]."""
    n = lp_ref.shape[0]
    tot = 0.0
    for i in range(n):
        tot += (full_kl(lp_ref[i], lp_cands[i], floor).item() if mode == "full"
                else truncated_kl(lp_ref[i], lp_cands[i], k).item())
    return tot / n


# ---------------------------------------------------------------- LoRA train
def make_prompt(ins, inp, out):
    if inp:
        return f"### Instruction:\n{ins}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
    return f"### Instruction:\n{ins}\n\n### Response:\n{out}"


def train_lora(base_model, tok, dataset, steps=250, bs=8, lr=2e-4, max_len=160, seed=0, device="cuda"):
    torch.manual_seed(seed)
    cfg = LoraConfig(r=16, lora_alpha=32, target_modules=LORA_TARGETS,
                     lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    m = get_peft_model(copy.deepcopy(base_model), cfg)
    m = m.to(device=device, dtype=torch.bfloat16)
    m.train()
    n = len(dataset)
    rng = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr)
    n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"trainable params: {n_tr:,}", flush=True)
    step = 0
    t0 = time.time()
    while step < steps:
        idx = torch.randperm(n, generator=rng).tolist()
        for i in idx:
            e = dataset[i]
            p = make_prompt(e["instruction"] or "", e.get("input") or "", e["output"] or "")
            ids = tok(p, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]
            if len(ids) < 8:
                continue
            inp = torch.tensor([ids], dtype=torch.long, device=device)
            out_t = m(input_ids=inp).logits
            ce = F.cross_entropy(out_t[0, :-1].float().reshape(-1, out_t.size(-1)),
                                 inp[0, 1:].reshape(-1))
            opt.zero_grad()
            ce.backward()
            opt.step()
            step += 1
            if step % 50 == 0:
                print(f"  step {step}/{steps}  loss {ce.item():.4f}  ({time.time()-t0:.0f}s)", flush=True)
            if step >= steps:
                break
    return m


# ---------------------------------------------------------------- scaled merge
def scaled_lora_model(clean_base, peft_model, s):
    """返回 clean_base 的 deepcopy, 权重 = W0 + s·ΔW_lora (仅被 LoRA 包裹的 Linear)。"""
    m = copy.deepcopy(clean_base)
    net = peft_model.base_model.model if hasattr(peft_model.base_model, "model") else peft_model.base_model
    for name, target in net.named_modules():
        if not hasattr(target, "lora_A") or not hasattr(target, "lora_B"):
            continue
        base_w = target.base_layer.weight.data
        A = target.lora_A["default"].weight.data      # [r, in]
        B = target.lora_B["default"].weight.data      # [out, r]
        scale = float(target.scaling["default"])
        delta = (B @ A).to(base_w.dtype) * scale      # [out, in]
        parts = name.split(".")
        cur = m
        ok = True
        for pp in parts:
            if not hasattr(cur, pp):
                ok = False
                break
            cur = getattr(cur, pp)
        if not ok or not hasattr(cur, "weight"):
            continue
        cur.weight.data.add_(delta.to(cur.weight.device) * s)
    return m


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--m", type=int, default=20)
    ap.add_argument("--sweep", default="0,0.05,0.1,0.2,0.4,0.7,1.0")
    ap.add_argument("--out", default="../data/lora_test_results.json")
    args = ap.parse_args()

    dev = "cuda"
    dt = AutoTokenizer.from_pretrained(TARGET, trust_remote_code=True)
    dtok_set = set(dt.get_vocab().keys())

    print("=== [A] load Qwen + shared-vocab probes + clean fingerprint ===", flush=True)
    aligner = Aligner(Q, layer=8, device=dev, dtype=torch.bfloat16)
    aligner.model.eval()
    shared = set(aligner.tok.get_vocab().keys()) & dtok_set
    aligner.restrict_vocab(shared)
    print(f"shared vocab size: {len(shared)}", flush=True)

    probes = []
    probe_strs = []
    for ref in REFERENCES:
        ids = aligner.tok(ref, return_tensors="pt")["input_ids"].to(aligner.vocab.device)
        cands = aligner.sample_near_synonyms(ids, M=args.m, top_k=6, k_neighbors=200)
        for c in cands:
            probes.append(c)
            probe_strs.append(aligner.decode(c)[0])
    n = len(probes)
    print(f"probe count: {n}", flush=True)

    LP0 = torch.stack([next_token_logp(aligner.model, p) for p in probes])   # [n, Vqwen]

    # ---- A: LoRA fine-tune + sensitivity sweep ----
    print("=== [A] LoRA fine-tune on alpaca-cleaned ===", flush=True)
    ds = load_from_disk(os.environ.get("ALPACA_DIR", "/root/autodl-tmp/work/alpaca_sub_2500"))
    peft_m = train_lora(aligner.model, aligner.tok, ds, steps=args.steps)
    peft_m.eval()

    print("=== [A] adapter-scale sweep D_KL ===", flush=True)
    sweep = [float(x) for x in args.sweep.split(",")]
    lora_rows = []
    for s in sweep:
        ms = scaled_lora_model(aligner.model, peft_m, s)
        ms.eval()
        LPs = torch.stack([next_token_logp(ms, p) for p in probes])
        kl_full = mean_kl(LPs, LP0, "full")
        kl_trunc = mean_kl(LPs, LP0, "trunc", k=20)
        lora_rows.append({"s": s, "KL_full": kl_full, "KL_trunc20": kl_trunc})
        print(f"  s={s:5.2f}  D_KL(full)={kl_full:9.4f}  D_KL(trunc20)={kl_trunc:9.4f}", flush=True)
        del ms
        torch.cuda.empty_cache()

    # ---- B: cross-family swap (Qwen -> Llama) ----
    print("=== [B] Qwen<->Llama cross-tokenizer swap ===", flush=True)
    qv = aligner.tok.get_vocab()
    lt = AutoTokenizer.from_pretrained(L)
    lv = lt.get_vocab()
    shared_toks = set(qv.keys()) & set(lv.keys())
    print(f"Qwen<->Llama shared surface tokens: {len(shared_toks)}", flush=True)

    q_id_t = torch.tensor([qv[t] for t in sorted(shared_toks)], device=dev)
    l_id_t = torch.tensor([lv[t] for t in sorted(shared_toks)], device=dev)

    lm = AutoModelForCausalLM.from_pretrained(L, dtype=torch.bfloat16, device_map=dev)
    lm.eval()
    cross_full, cross_trunc = [], []
    lp0_shared = LP0[:, q_id_t]                       # [n, shared]
    for i, pstr in enumerate(probe_strs):
        lids = lt(pstr, add_special_tokens=False)["input_ids"]
        if not lids:
            continue
        lin = torch.tensor([lids], dtype=torch.long).to(dev)
        lpl = next_token_logp(lm, lin)[l_id_t]       # [shared] on Llama
        klf = (lp0_shared[i].exp() * (lp0_shared[i] - lpl)).sum().item()
        cross_full.append(klf)
        t = torch.cat([torch.topk(lp0_shared[i], 20).indices, torch.topk(lpl, 20).indices]).unique()
        q0 = lp0_shared[i][t].exp(); qr = lpl[t].exp()
        tail0 = (1 - q0.sum()).clamp_min(0); tailr = (1 - qr.sum()).clamp_min(0)
        kl = (q0 * (lp0_shared[i][t] - lpl[t])).sum()
        if tail0 > 0:
            kl = kl + tail0 * (torch.log(tail0) - torch.log(tailr + 1e-30))
        cross_trunc.append(kl.item())
    cross_res = {
        "KL_full_shared": (sum(cross_full) / len(cross_full)) if cross_full else None,
        "KL_trunc_shared": (sum(cross_trunc) / len(cross_trunc)) if cross_trunc else None,
        "n_probes_evaluated": len(cross_full),
        "shared_tokens": len(shared_toks),
    }
    print(f"  cross-family D_KL(full, shared)={cross_res['KL_full_shared']:.4f}  "
          f"D_KL(trunc, shared)={cross_res['KL_trunc_shared']:.4f}  (n={cross_res['n_probes_evaluated']})", flush=True)

    # ---- zero-noise sanity ----
    reLP0 = torch.stack([next_token_logp(aligner.model, p) for p in probes])
    zero_full = mean_kl(reLP0, LP0, "full")
    zero_trunc = mean_kl(reLP0, LP0, "trunc", k=20)
    print(f"  zero-noise sanity: D_KL(full)={zero_full:.6f}  D_KL(trunc)={zero_trunc:.6f}  (expect ~0)", flush=True)

    res = {
        "model": Q, "layer": 8, "n_probes": n,
        "zero_noise": {"KL_full": zero_full, "KL_trunc20": zero_trunc},
        "lora_sweep": lora_rows,
        "lora_train_steps": args.steps,
        "cross_family_qwen_to_llama": cross_res,
        "probes": probe_strs,
    }
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
