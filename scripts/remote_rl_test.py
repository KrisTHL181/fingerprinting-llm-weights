#!/usr/bin/env python3
"""
remote_rl_test.py — 实测指纹方法对"强化学习(RL)对齐微调"的识别能力 (Qwen-0.8B, L8)。

方法与论文附录 LoRA-SFT 实验对齐, 但用 RL 目标:
  DPO (Direct Preference Optimization) — RLHF-family 的偏好优化对齐。
  - 参考模型 = 冻结的 clean Qwen; 策略模型 = LoRA(r=16) 包裹的 Qwen。
  - 训练数据: Dahoas/rm-static 偏好对 (chosen/rejected), 经 HF 镜像下载子集。
  - DPO loss = -log σ( β·[(logp_θ(y+)-logp_θ(y-)) - (logp_ref(y+)-logp_ref(y-))] )

指纹统计量 = 论文主统计量 D_KL = mean_x KL( p_0(·|x) || p_cand(·|x) ),
  逐探针平均; 同时报 KL_full(全词表, clamp -30) 与 KL_trunc(top-20 截断, 黑盒口径)。
  与 LoRA-SFT 实验相同: 把训练得到的 LoRA 低秩更新 ΔW 视为固定方向,
  按尺度 s 插值 (W_s = W0 + s·ΔW), 得 D_KL(s), 拟合二次,
  并用 Qwen L8 高斯噪声标定映射到"等价相对噪声尺度 r_eq"。
  对比: RL(DPO) 的指纹灵敏度 vs. 论文附录 SFT(LoRA) 的灵敏度。
"""
from __future__ import annotations
import argparse, copy, json, time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from align import Aligner

TARGET = "deepseek-ai/DeepSeek-V4-Flash-0731"
Q = "Qwen/Qwen3.5-0.8B"
REFERENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "The committee approved the new budget proposal this morning.",
    "She walked slowly through the old town under the rain.",
]
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# ---------------------------------------------------------------- KL utils (同 remote_lora_test)
@torch.no_grad()
def next_token_logp(model, input_ids):
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


# ---------------------------------------------------------------- DPO helpers
def response_logp(model, prompt_ids, resp_ids):
    """对 prompt+response teacher-forced 前向, 返回 response 逐 token logp 之和 (标量)。"""
    P, R = len(prompt_ids), len(resp_ids)
    if P < 1 or R < 1:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    inp = torch.tensor([prompt_ids + resp_ids], dtype=torch.long, device=next(model.parameters()).device)
    logits = model(inp).logits[0].float()          # [T, V]
    logp = torch.log_softmax(logits, dim=-1)
    r = torch.tensor(resp_ids, device=inp.device)
    pred = logp[P - 1 : P + R - 1]                 # R 行: 预测 resp 每个 token
    return pred.gather(1, r.unsqueeze(1)).sum()


def dpo_loss(policy, ref_model, tok, batch, beta, max_prompt, max_resp, dev):
    """batch: list[(prompt, chosen, rejected)] -> mean DPO loss."""
    tot = 0.0; n = 0
    for prompt, chosen, rejected in batch:
        pc = tok(prompt, add_special_tokens=False, truncation=True, max_length=max_prompt)["input_ids"]
        ch = tok(chosen, add_special_tokens=False, truncation=True, max_length=max_resp)["input_ids"]
        rej = tok(rejected, add_special_tokens=False, truncation=True, max_length=max_resp)["input_ids"]
        if not pc or not ch or not rej:
            continue
        lpc_ch = response_logp(policy, pc, ch)
        lpc_rej = response_logp(policy, pc, rej)
        with torch.no_grad():
            lr_ch = response_logp(ref_model, pc, ch)
            lr_rej = response_logp(ref_model, pc, rej)
        ratio = beta * ((lpc_ch - lpc_rej) - (lr_ch - lr_rej))
        tot = tot + (-F.logsigmoid(ratio))
        n += 1
    return tot / max(n, 1)


def load_pref_data(n_data, seed=0, max_prompt=96, max_resp=256):
    """从 HF 镜像下载 rm-static, 截断并取 n_data 条偏好对。"""
    ds = load_dataset("Dahoas/rm-static", split="train")
    ds = ds.shuffle(seed=seed)
    out = []
    for e in ds:
        prompt = (e["prompt"] or "").strip()
        chosen = (e["chosen"] or "").strip()
        rejected = (e["rejected"] or "").strip()
        if not prompt or not chosen or not rejected:
            continue
        if len(prompt) > 400 or len(chosen) > 800 or len(rejected) > 800:
            continue
        out.append((prompt, chosen, rejected))
        if len(out) >= n_data:
            break
    return out


def train_dpo(base_model, tok, data, steps=200, bs=4, lr=2e-4, beta=0.1,
              max_prompt=96, max_resp=256, seed=0, device="cuda"):
    """返回训练好的 peft_model (policy), 参考模型=base_model(冻结)。"""
    torch.manual_seed(seed)
    cfg = LoraConfig(r=16, lora_alpha=32, target_modules=LORA_TARGETS,
                     lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    policy = get_peft_model(copy.deepcopy(base_model), cfg)
    policy = policy.to(device=device, dtype=torch.bfloat16)
    policy.train()
    for p in base_model.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=lr)
    n_tr = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"trainable params: {n_tr:,}", flush=True)
    rng = torch.Generator().manual_seed(seed)
    n = len(data)
    t0 = time.time()
    step = 0
    while step < steps:
        idx = torch.randperm(n, generator=rng).tolist()
        for i in range(0, n, bs):
            batch = [data[j] for j in idx[i : i + bs]]
            loss = dpo_loss(policy, base_model, tok, batch, beta, max_prompt, max_resp, device)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % 50 == 0:
                print(f"  step {step}/{steps}  dpo_loss {loss.item():.4f}  ({time.time()-t0:.0f}s)", flush=True)
            if step >= steps:
                break
    return policy


# ---------------------------------------------------------------- scaled merge (同 remote_lora_test)
def scaled_lora_model(clean_base, peft_model, s):
    """返回 clean_base 的 deepcopy, 权重 = W0 + s·ΔW_lora。"""
    m = copy.deepcopy(clean_base)
    net = peft_model.base_model.model if hasattr(peft_model.base_model, "model") else peft_model.base_model
    for name, target in net.named_modules():
        if not hasattr(target, "lora_A") or not hasattr(target, "lora_B"):
            continue
        base_w = target.base_layer.weight.data
        A = target.lora_A["default"].weight.data
        B = target.lora_B["default"].weight.data
        scale = float(target.scaling["default"])
        delta = (B @ A).to(base_w.dtype) * scale
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
    ap.add_argument("--m", type=int, default=20)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--n-data", type=int, default=1000)
    ap.add_argument("--sweep", default="0,0.05,0.1,0.2,0.4,0.7,1.0")
    ap.add_argument("--out", default="../data/rl_test_results.json")
    args = ap.parse_args()

    dev = "cuda"
    dt = AutoTokenizer.from_pretrained(TARGET, trust_remote_code=True)
    dtok_set = set(dt.get_vocab().keys())

    print("=== load Qwen + shared-vocab probes + clean fingerprint ===", flush=True)
    aligner = Aligner(Q, layer=8, device=dev, dtype=torch.bfloat16)
    aligner.model.eval()
    shared = set(aligner.tok.get_vocab().keys()) & dtok_set
    aligner.restrict_vocab(shared)
    print(f"shared vocab size: {len(shared)}", flush=True)

    probes, probe_strs = [], []
    for ref in REFERENCES:
        ids = aligner.tok(ref, return_tensors="pt")["input_ids"].to(aligner.vocab.device)
        cands = aligner.sample_near_synonyms(ids, M=args.m, top_k=6, k_neighbors=200)
        probes += cands
        probe_strs += aligner.decode(cands)
    n = len(probes)
    print(f"probe count: {n}", flush=True)

    LP0 = torch.stack([next_token_logp(aligner.model, p) for p in probes])

    # ---- DPO train ----
    print("=== load preference data (rm-static subset) ===", flush=True)
    data = load_pref_data(args.n_data)
    print(f"pref pairs: {len(data)}", flush=True)

    print("=== DPO (LoRA r=16) on preference data ===", flush=True)
    peft_m = train_dpo(aligner.model, aligner.tok, data, steps=args.steps, bs=args.bs)
    peft_m.eval()

    # ---- adapter-scale sweep D_KL ----
    print("=== adapter-scale sweep D_KL ===", flush=True)
    sweep = [float(x) for x in args.sweep.split(",")]
    dpo_rows = []
    for s in sweep:
        ms = scaled_lora_model(aligner.model, peft_m, s)
        ms.eval()
        LPs = torch.stack([next_token_logp(ms, p) for p in probes])
        kl_full = mean_kl(LPs, LP0, "full")
        kl_trunc = mean_kl(LPs, LP0, "trunc", k=20)
        dpo_rows.append({"s": s, "KL_full": kl_full, "KL_trunc20": kl_trunc})
        print(f"  s={s:5.2f}  D_KL(full)={kl_full:9.4f}  D_KL(trunc20)={kl_trunc:9.4f}", flush=True)
        del ms
        torch.cuda.empty_cache()

    # ---- zero-noise sanity ----
    reLP0 = torch.stack([next_token_logp(aligner.model, p) for p in probes])
    zero_full = mean_kl(reLP0, LP0, "full")
    zero_trunc = mean_kl(reLP0, LP0, "trunc", k=20)
    print(f"zero-noise sanity: D_KL(full)={zero_full:.6f}  D_KL(trunc)={zero_trunc:.6f}", flush=True)

    res = {
        "model": Q, "layer": 8, "n_probes": n, "method": "LoRA-DPO (rm-static subset)",
        "dpo": {"steps": args.steps, "bs": args.bs, "n_data": len(data), "lora_r": 16},
        "zero_noise": {"KL_full": zero_full, "KL_trunc20": zero_trunc},
        "dpo_sweep": dpo_rows,
        "probes": probe_strs,
    }
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
