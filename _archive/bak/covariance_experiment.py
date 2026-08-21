"""
核心实验: 用近义句集合估算 Qwen 中间层表征协方差, 并以权重噪声 N(0,s^2) 标定指纹发散度。
目标: 建立 "权重扰动尺度 s ↔ 指纹发散度 D(s)" 曲线, 以便实测给定指纹差异时反推隐含权重差异,
      从而推断两个模型权重是否几乎相同。

两种指纹(可迁移到 DeepSeek):
  (1) logprob 指纹  f_M = [logp_M(S'_j)]_j   (DeepSeek API 可提供 top-20 logprobs, 与之可比)
  (2) 表征协方差    Σ_M 在近义邻域上的 pooled hidden 协方差 (Qwen 专属, 更细)

噪声: 每张量加 N(0, (r*sigma_W)^2), 即相对扰动 r(0=无噪声)。s = r*sigma_W 即 "N(0,s^2) 噪声"。
"""
from __future__ import annotations
import argparse, math, copy
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
TARGET_MODEL_NAME = "deepseek-ai/DeepSeek-V4-Flash-0731"

REFERENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "The committee approved the new budget proposal this morning.",
    "She walked slowly through the old town under the rain.",
]


def pool(h, mask, mode="mean"):
    if mode == "mean":
        return (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
    return h[:, 0]


@torch.no_grad()
def pooled_hidden(model, tok, ids, layer):
    out = model(ids, output_hidden_states=True)
    mask = (ids != tok.pad_token_id).to(out.hidden_states[layer].dtype)
    return pool(out.hidden_states[layer], mask)          # [B,D]


@torch.no_grad()
def seq_logprobs(model, tok, ids):
    """整个序列的对数概率 (exclude 首个 BOS 位置)。返回标量。"""
    logits = model(ids).logits                            # [B,T,V]
    lp = F.log_softmax(logits.float(), dim=-1)
    B, T, V = lp.shape
    # token t_i 的 logprob = lp[i-1, t_i]
    tok_ids = ids[:, 1:]
    gather = lp[:, :-1, :].gather(2, tok_ids.unsqueeze(-1)).squeeze(-1)  # [B,T-1]
    return gather.sum(1)                                   # [B]


def perturb_weights(model, r, seed=0):
    """按每张量 std 加 N(0,(r*sigma)^2) 噪声(相对扰动 r)。返回新模型(copy)。"""
    g = torch.Generator(device="cuda").manual_seed(seed)
    m2 = copy.deepcopy(model)
    with torch.no_grad():
        for name, p in m2.named_parameters():
            if p.requires_grad:
                std = p.data.std().clamp_min(1e-8)
                noise = torch.randn(p.shape, generator=g, device=p.device, dtype=p.dtype)
                p.data.add_(noise * (r * std))
    return m2


def fingerprint_logp(model, tok, candidates_batch):
    """candidates_batch: list of tensor ids. 返回 logprob 向量。"""
    return seq_logprobs(model, tok, candidates_batch)


def fingerprint_cov(model, tok, ids_batch, layer):
    """近义候选 pooled hidden 的中心化协方差 Σ ∈ [D,D]。"""
    H = pooled_hidden(model, tok, ids_batch, layer)        # [m,D]
    Hc = H - H.mean(0, keepdim=True)
    m = H.shape[0]
    return (Hc.T @ Hc) / max(m - 1, 1), H                 # (Sigma, H)


def rel_fro(A, B):
    """相对 Frobenius 发散度 ||A-B||_F / ||B||_F"""
    return (A - B).norm().item() / B.norm().clamp_min(1e-12).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--noise", type=str, default="0,0.01,0.03,0.06,0.10",
                    help="相对噪声 r 的逗号分隔列表")
    ap.add_argument("--pool", default="mean")
    args = ap.parse_args()

    import sys; sys.path.insert(0, "/root/autodl-tmp/work")
    from align import Aligner
    aligner = Aligner(MODEL_NAME, layer=args.layer, pool=args.pool)
    model, tok = aligner.model, aligner.tok
    model.eval()
    # 共享子词表限制(与可迁移前提一致: 候选 token 必须是两模型共享的)
    from transformers import AutoTokenizer
    qw = set(tok.get_vocab().keys())
    dt = set(AutoTokenizer.from_pretrained(TARGET_MODEL_NAME,
                                           trust_remote_code=True).get_vocab().keys())
    shared = qw & dt
    aligner.restrict_vocab(shared)
    print(f"shared vocab: {len(shared)}/{len(qw)} Qwen tokens")
    print(f"model layers={model.config.num_hidden_layers} hidden={model.config.hidden_size} "
          f"layer={args.layer}")

    cand_sets = []   # list of (ref_str, [candidate_id_tensors])
    for ref in REFERENCES:
        ids = tok(ref, return_tensors="pt")["input_ids"].to("cuda")
        cands = aligner.sample_near_synonyms(ids, M=20, top_k=6, k_neighbors=200)
        cand_sets.append((ref, cands))
        print(f"reference: {ref!r}  ({len(cands)} candidates)")
        for c in cands:
            print(f"    cand: {aligner.decode(c)[0]!r}")

    # ---- 无噪声基线指纹 ----
    base_logp, base_cov = {}, {}
    for ref, cands in cand_sets:
        B = torch.cat(cands, dim=0)
        base_logp[ref] = fingerprint_logp(model, tok, B)
        base_cov[ref], _ = fingerprint_cov(model, tok, B, args.layer)

    # ---- 噪声标定 ----
    noise_list = [float(x) for x in args.noise.split(",")]
    print(f"\n{'r(rel)':>8} {'D_logp':>10} {'D_cov':>10} {'D_logp_cos':>12}")
    print("-" * 46)
    rows = []
    for r in noise_list:
        m_r = perturb_weights(model, r)
        d_logp, d_cov, d_cos = 0.0, 0.0, 0.0
        n = 0
        for ref, cands in cand_sets:
            B = torch.cat(cands, dim=0)
            lp_r = fingerprint_logp(m_r, tok, B)
            cov_r, H_r = fingerprint_cov(m_r, tok, B, args.layer)
            d_logp += rel_fro(lp_r, base_logp[ref])
            d_cov += rel_fro(cov_r, base_cov[ref])
            # logprob 指纹的余弦相似度(方向性保持)
            a = (lp_r - lp_r.mean()).float(); b = (base_logp[ref] - base_logp[ref].mean()).float()
            d_cos += F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
            n += 1
        d_logp, d_cov, d_cos = d_logp / n, d_cov / n, d_cos / n
        rows.append((r, d_logp, d_cov, d_cos))
        print(f"{r:>8.3f} {d_logp:>10.4f} {d_cov:>10.4f} {d_cos:>12.4f}", flush=True)
        del m_r
        torch.cuda.empty_cache()

    # ---- 解读: 拟合 D_logp vs r 的近线性关系, 给出反查函数 ----
    print("\n校准解读: r 增大 -> D_logp/D_cov 单调增大。给定目标模型的实测指纹差异, 反查 r 即隐含权重相对扰动尺度。")
    if len(rows) >= 2:
        rs = [x[0] for x in rows]; dls = [x[1] for x in rows]
        # 简单线性斜率
        from numpy import polyfit
        k, b = polyfit(rs, dls, 1)
        print(f"线性拟合 D_logp ~= {k:.3f}*r + {b:.4f}   (反查 r ≈ (D_logp - {b:.4f})/{k:.3f})")


if __name__ == "__main__":
    main()
