#!/usr/bin/env python3
"""
calibrate_logp_cov.py — 黑盒(next-token logp 向量)协方差指纹的权重噪声标定

口径与 deepseek_covariance.py 一致: 指纹 = 观测 token 集合 T 上的 next-token logp 向量
  φ(x) = [ log p(t | x) ]_{t ∈ T}
其中 T = 干净模型各探针 top-K next-token 的并集 ∩ 共享词表 Vsh (K=20 匹配 API top-20)。
对 open-weight 模型注入 N(0,(rσ)²) 噪声(σ = 每张量参数尺度), 测:
  D_logp = || μ_r − μ_0 || / || μ_0 ||        (均值向量相对发散)
  D_cov  = || λ_r − λ_0 || / || λ_0 ||        (协方差谱相对发散; λ 用 m×m Gram 特征值,
                                              避免物化 |T|×|T| 协方差)

这是"本地白盒"口径(φ 在 T 上取真实 logp); 黑盒(API top-k + floor)已在 §5.2 /
deepseek_covariance.py 中。

用法:
  python calibrate_logp_cov.py --model Qwen/Qwen3.5-0.8B      --layer 8  --out scan_qwen_l8_logp.txt
  python calibrate_logp_cov.py --model Qwen/Qwen3.5-0.8B      --layer 16 --out scan_qwen_l16_logp.txt
  python calibrate_logp_cov.py --model unsloth/Llama-3.2-1B   --layer 8  --out scan_llama_l8_logp.txt
  python calibrate_logp_cov.py --model unsloth/gemma-3-1b-pt  --layer 8  --out scan_gemma_l8_logp.txt
  --full-vsh: 对照, 用全共享词表 Vsh 而非 top-K 观测集; --top-k K (默认 20)
"""
from __future__ import annotations
import argparse, copy
import torch
from transformers import AutoTokenizer
from align import Aligner

TARGET_MODEL_NAME = "deepseek-ai/DeepSeek-V4-Flash-0731"

REFERENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "The committee approved the new budget proposal this morning.",
    "She walked slowly through the old town under the rain.",
]


@torch.no_grad()
def logp_over(model, input_ids, T_ids):
    """φ(x) = [log p(t|x)]_{t∈T}, 取序列末尾位置的 next-token 分布, 在固定 token 集 T 上取真实 logp。"""
    logits = model(input_ids).logits                                # [1,T,Vfull]
    lp = torch.log_softmax(logits.float()[:, -1, :], dim=-1)        # [1,Vfull]
    return lp[0, T_ids]                                             # [|T|]


@torch.no_grad()
def top_token_set(model, probes, vsh_ids, K):
    """T = 干净模型各探针 top-K next-token 的并集 ∩ Vsh(固定 token 集)。"""
    T = set()
    for p in probes:
        logits = model(p).logits[0, -1].float()                     # [Vfull]
        lv = logits[vsh_ids]                                        # [|Vsh|]
        topk_local = torch.topk(lv, K).indices                      # 在 Vsh 内的下标
        T.update(vsh_ids[topk_local].tolist())
    return torch.tensor(sorted(T), device=vsh_ids.device)


def covariance_spectrum(X):
    """X [m,V]: 样本协方差 Σ=(XcᵀXc)/(m−1) 的谱(降序, ≥0)。用 m×m Gram 特征值。"""
    m = X.shape[0]
    Xc = X - X.mean(0, keepdim=True)
    gram = (Xc @ Xc.T) / max(m - 1, 1)                              # [m,m]
    eig = torch.linalg.eigvalsh(gram)
    return torch.flip(torch.clamp(eig, min=0.0), dims=(0,))         # 降序


def perturb_weights(model, r, seed=0):
    """每张量加 N(0,(r·std)²) 噪声(相对扰动 r)。返回深拷贝后的新模型。"""
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
    ap.add_argument("--m", type=int, default=20, help="每参考句的近邻候选数")
    ap.add_argument("--top-k", type=int, default=20, help="每探针取 top-K next-token(匹配 API)")
    ap.add_argument("--noise", default="0,0.02,0.05,0.10,0.20,0.40,0.60,1.00")
    ap.add_argument("--out", default="../data/cov_logp.txt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full-vsh", action="store_true", help="用全共享词表 Vsh 而非 top-K 观测集(对照)")
    ap.add_argument("--no-shared", action="store_true", help="不限制共享词表(用全词表)")
    args = ap.parse_args()

    aligner = Aligner(args.model, layer=args.layer)
    aligner.model.eval()

    # 共享词表 Vsh(与 DeepSeek 交集)
    if args.no_shared:
        aligner.restrict_vocab(None)
        vsh_ids = torch.arange(aligner.vocab.shape[0], device=aligner.vocab.device)
    else:
        dt = set(AutoTokenizer.from_pretrained(TARGET_MODEL_NAME,
                                               trust_remote_code=True).get_vocab().keys())
        shared = set(aligner.tok.get_vocab().keys()) & dt
        aligner.restrict_vocab(shared)
        vsh_ids = aligner.shared_ids
    print(f"shared vocab: {len(vsh_ids)}", flush=True)

    # 生成探针(embedding 邻居替换, 对齐 layer)
    probes = []
    for ref in REFERENCES:
        ids = aligner.tok(ref, return_tensors="pt")["input_ids"].to(aligner.vocab.device)
        cands = aligner.sample_near_synonyms(ids, M=args.m, top_k=6, k_neighbors=200)
        probes.extend(cands)
    print(f"probe count: {len(probes)}", flush=True)

    # 固定 token 集 T
    if args.full_vsh:
        T_ids = vsh_ids
    else:
        T_ids = top_token_set(aligner.model, probes, vsh_ids, args.top_k)
    print(f"T size: {len(T_ids)}", flush=True)

    # 干净模型指纹 (μ0, 谱 λ0)
    X0 = torch.stack([logp_over(aligner.model, p, T_ids) for p in probes])
    mu0 = X0.mean(0)
    lam0 = covariance_spectrum(X0)
    print(f"X0: {tuple(X0.shape)}  ||mu0||={mu0.norm().item():.3f}  "
          f"top3占比={lam0[:3].sum().item()/lam0.sum().item()*100:.0f}%", flush=True)

    noise_list = [float(x) for x in args.noise.split(",")]
    print(f"\n{'r(rel)':>8} {'D_logp':>10} {'D_cov':>10}", flush=True)
    print("-" * 32, flush=True)
    rows = []
    for r in noise_list:
        m_r = perturb_weights(aligner.model, r, args.seed)
        Xr = torch.stack([logp_over(m_r, p, T_ids) for p in probes])
        mu_r = Xr.mean(0)
        lam_r = covariance_spectrum(Xr)
        D_logp = (mu_r - mu0).norm().item() / mu0.norm().clamp_min(1e-12).item()
        D_cov = (lam_r - lam0).norm().item() / lam0.norm().clamp_min(1e-12).item()
        rows.append((r, D_logp, D_cov))
        print(f"{r:>8.3f} {D_logp:>10.4f} {D_cov:>10.4f}", flush=True)
        del m_r
        torch.cuda.empty_cache()

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("r(rel)  D_logp  D_cov\n")
        for r, dl, dc in rows:
            f.write(f"{r:.3f}  {dl:.4f}  {dc:.4f}\n")
    print(f"\n已写入 {args.out}", flush=True)

    rs = torch.tensor([x[0] for x in rows if x[0] <= 0.1], dtype=torch.float64)
    ds = torch.tensor([x[1] for x in rows if x[0] <= 0.1], dtype=torch.float64)
    if len(rs) >= 2:
        A = torch.stack([rs, torch.ones_like(rs)], dim=1)
        k, b = torch.linalg.lstsq(A, ds).solution.tolist()
        print(f"线性拟合 D_logp ≈ {k:.3f}·r + {b:.4f}  (小 r 区间)", flush=True)


if __name__ == "__main__":
    main()
