#!/usr/bin/env python3
"""
sweep_stats.py — 对比多种黑盒发散度统计量在权重噪声下的单调性/灵敏度

对 open-weight 模型注入 N(0,(rσ)²) 噪声, 对每个 r 计算多种指纹发散度:
  D_mean  : || μ_r − μ_0 || / || μ_0 ||          (观测 token 集 T 上均值向量 L2)
  D_cov   : || λ_r − λ_0 || / || λ_0 ||          (T 上协方差谱)
  D_KL    : mean_x KL( p_0(·|x) || p_r(·|x) )    (全词表 next-token 分布, 逐探针平均)
  D_TV    : mean_x TV( p_0, p_r )                (全词表 next-token 分布, 逐探针平均)
  D_rank  : 1 − mean_x Spearman( logp_0, logp_r )  (T 上的排序相关)

目的: 找出既单调又灵敏的黑盒统计量。KL/TV 是凸散度, 不随探针分布塌缩而收缩,
预期比 D_cov 更单调; 排序相关对 logp 幅值塌缩不敏感。

用法: python sweep_stats.py --model Qwen/Qwen3.5-0.8B --layer 8 --out sweep_qwen_l8.txt
"""
from __future__ import annotations
import argparse, copy
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import AutoTokenizer
from align import Aligner

TARGET_MODEL_NAME = "deepseek-ai/DeepSeek-V4-Flash-0731"
REFERENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "The committee approved the new budget proposal this morning.",
    "She walked slowly through the old town under the rain.",
]


@torch.no_grad()
def next_token_dist(model, input_ids):
    """next-token 分布 (prob, logprob), 各 [Vfull]。"""
    logits = model(input_ids).logits[0, -1].float()
    return torch.softmax(logits, dim=-1), torch.log_softmax(logits, dim=-1)


@torch.no_grad()
def top_token_set(model, probes, vsh_ids, K):
    T = set()
    for p in probes:
        logits = model(p).logits[0, -1].float()
        lv = logits[vsh_ids]
        T.update(vsh_ids[torch.topk(lv, K).indices].tolist())
    return torch.tensor(sorted(T), device=vsh_ids.device)


def covariance_spectrum(X):
    m = X.shape[0]
    Xc = X - X.mean(0, keepdim=True)
    gram = (Xc @ Xc.T) / max(m - 1, 1)
    eig = torch.linalg.eigvalsh(gram)
    return torch.flip(torch.clamp(eig, min=0.0), dims=(0,))


def perturb_weights(model, r, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    m2 = copy.deepcopy(model)
    with torch.no_grad():
        for p in m2.parameters():
            if p.requires_grad:
                std = p.data.std().clamp_min(1e-8)
                noise = torch.randn(p.shape, generator=g, device=p.device, dtype=p.dtype)
                p.data.add_(noise * (r * std))
    return m2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--m", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--noise", default="0,0.02,0.05,0.10,0.20,0.40,0.60,1.00")
    ap.add_argument("--out", default="../data/sweep.txt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    aligner = Aligner(args.model, layer=args.layer)
    aligner.model.eval()
    dt = set(AutoTokenizer.from_pretrained(TARGET_MODEL_NAME,
                                           trust_remote_code=True).get_vocab().keys())
    shared = set(aligner.tok.get_vocab().keys()) & dt
    aligner.restrict_vocab(shared)
    vsh_ids = aligner.shared_ids
    print(f"shared vocab: {len(vsh_ids)}", flush=True)

    probes = []
    for ref in REFERENCES:
        ids = aligner.tok(ref, return_tensors="pt")["input_ids"].to(aligner.vocab.device)
        probes.extend(aligner.sample_near_synonyms(ids, M=args.m, top_k=6, k_neighbors=200))
    print(f"probe count: {len(probes)}", flush=True)

    T_ids = top_token_set(aligner.model, probes, vsh_ids, args.top_k)
    print(f"T size: {len(T_ids)}", flush=True)

    # 干净模型: 每探针 next-token 分布 + T 上 logp 向量
    p0_list, lp0_list = [], []
    for p in probes:
        prob, logp = next_token_dist(aligner.model, p)
        p0_list.append(prob); lp0_list.append(logp[T_ids])
    P0 = torch.stack(p0_list)                       # [m, Vfull]
    X0 = torch.stack(lp0_list)                      # [m, |T|]
    mu0 = X0.mean(0)
    lam0 = covariance_spectrum(X0)

    noise_list = [float(x) for x in args.noise.split(",")]
    hdr = f"{'r':>8} {'D_mean':>9} {'D_cov':>9} {'D_KL':>9} {'D_TV':>9} {'D_rank':>9}"
    print("\n" + hdr, flush=True)
    print("-" * len(hdr), flush=True)
    rows = []
    for r in noise_list:
        m_r = perturb_weights(aligner.model, r, args.seed)
        pr_list, lpr_list = [], []
        for p in probes:
            prob, logp = next_token_dist(m_r, p)
            pr_list.append(prob); lpr_list.append(logp[T_ids])
        Pr = torch.stack(pr_list)
        Xr = torch.stack(lpr_list)
        mu_r = Xr.mean(0)
        lam_r = covariance_spectrum(Xr)

        D_mean = (mu_r - mu0).norm().item() / mu0.norm().clamp_min(1e-12).item()
        D_cov = (lam_r - lam0).norm().item() / lam0.norm().clamp_min(1e-12).item()
        # KL(p0 || pr) = sum_t p0 (log p0 - log pr); logp 下界 floor 防止 softmax 下溢→-inf
        logP0 = torch.log(P0).clamp(min=-30.0)
        logPr = torch.log(Pr).clamp(min=-30.0)
        kl = (P0 * (logP0 - logPr)).sum(-1).mean().item()
        tv = (0.5 * (P0 - Pr).abs().sum(-1)).mean().item()
        # Spearman: 逐探针在 T 上的 logp 排序相关
        rho = []
        for i in range(X0.shape[0]):
            rho.append(spearmanr(X0[i].cpu().numpy(), Xr[i].cpu().numpy()).correlation)
        D_rank = 1.0 - float(sum(rho) / len(rho))
        rows.append((r, D_mean, D_cov, kl, tv, D_rank))
        print(f"{r:>8.3f} {D_mean:>9.4f} {D_cov:>9.4f} {kl:>9.4f} {tv:>9.4f} {D_rank:>9.4f}",
              flush=True)
        del m_r
        torch.cuda.empty_cache()

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("r D_mean D_cov D_KL D_TV D_rank\n")
        for row in rows:
            f.write(" ".join(f"{x:.4f}" for x in row) + "\n")
    print(f"\n已写入 {args.out}", flush=True)


if __name__ == "__main__":
    main()
