#!/usr/bin/env python3
"""
baseline_topk.py — Win-point A: does the black-box top-k readout preserve the signal of
random-probe baselines as well as it preserves the paper's representation-aligned probes?

Aligned probes induce low-entropy next-token distributions (mass concentrated in the
top-k), random probes high-entropy (mass spread into the tail). The paper's fingerprint
is read through a top-k (top-20) truncated KL, so a method whose signal lives in the tail
loses most of it under truncation. We measure, for each probe condition and each r, the
mean per-probe top-20 truncated KL and its ratio to the full-vocabulary KL.

Usage:
  python baseline_topk.py --model Qwen/Qwen3.5-0.8B --layer 8 --out baseline_topk_qwen_l8.txt
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
FLOOR = -30.0
RND_SEED = 0


@torch.no_grad()
def next_token_logp(model, input_ids):
    return torch.log_softmax(model(input_ids).logits[0, -1].float(), dim=-1)


@torch.no_grad()
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


def full_kl(lp0, lpr):
    p0 = lp0.exp().clamp_min(1e-30)
    return (p0 * (lp0.clamp(min=FLOOR) - lpr.clamp(min=FLOOR))).sum()


def truncated_kl(lp0, lpr, k):
    """Top-k truncated KL: union of top-k supports + renormalized tail bucket."""
    top = torch.cat([torch.topk(lp0, k).indices, torch.topk(lpr, k).indices]).unique()
    q0 = lp0[top].exp().clamp_min(1e-30)
    qr = lpr[top].exp().clamp_min(1e-30)
    tail0 = (1.0 - q0.sum()).clamp_min(0.0)
    tailr = (1.0 - qr.sum()).clamp_min(0.0)
    kl = (q0 * (lp0[top] - lpr[top])).sum()
    if tail0 > 0:
        kl = kl + tail0 * (torch.log(tail0 + 1e-30) - torch.log(tailr + 1e-30))
    return kl


@torch.no_grad()
def random_probes(aligner, ref_ids, n_per_ref, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    ids = aligner.shared_ids
    out = []
    for ref in ref_ids:
        T = ref.shape[1]
        for _ in range(n_per_ref):
            idx = torch.randint(0, ids.numel(), (T,), generator=g, device=ids.device)
            out.append(ids[idx].unsqueeze(0))
    return out


def mean_entropy(lps):
    ps = lps.exp().clamp_min(1e-30)
    return (-(ps * lps).sum(-1)).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--m", type=int, default=20)
    ap.add_argument("--m-rnd", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--noise", default="0,0.02,0.05,0.10,0.20")
    ap.add_argument("--out", default="baseline_topk.txt")
    args = ap.parse_args()

    aligner = Aligner(args.model, layer=args.layer)
    aligner.model.eval()
    dt = set(AutoTokenizer.from_pretrained(TARGET_MODEL_NAME, trust_remote_code=True).get_vocab().keys())
    shared = set(aligner.tok.get_vocab().keys()) & dt
    aligner.restrict_vocab(shared)
    vsh = aligner.shared_ids

    ref_ids = [aligner.tok(r, return_tensors="pt")["input_ids"].to(vsh.device) for r in REFERENCES]
    alg_probes = []
    for ids in ref_ids:
        alg_probes.extend(aligner.sample_near_synonyms(ids, M=args.m, top_k=6, k_neighbors=200))
    rnd_probes = random_probes(aligner, ref_ids, args.m_rnd, seed=RND_SEED)

    def cond_readout(probes):
        return torch.stack([next_token_logp(aligner.model, p) for p in probes])

    LP0_alg = cond_readout(alg_probes)
    LP0_rnd = cond_readout(rnd_probes)
    print(f"alg_probes={len(alg_probes)} rnd_probes={len(rnd_probes)}", flush=True)
    print(f"entropy  alg={mean_entropy(LP0_alg):.2f} rnd={mean_entropy(LP0_rnd):.2f}", flush=True)

    noise_list = [float(x) for x in args.noise.split(",")]
    hdr = ["r", "full_alg", "tr20_alg", "ratio_alg", "full_rnd", "tr20_rnd", "ratio_rnd"]
    print(" ".join(f"{h:>11}" for h in hdr), flush=True)
    rows = []
    for r in noise_list:
        m_r = perturb_weights(aligner.model, r)
        LP1_alg = torch.stack([next_token_logp(m_r, p) for p in alg_probes])
        LP1_rnd = torch.stack([next_token_logp(m_r, p) for p in rnd_probes])
        na, la = len(alg_probes), len(rnd_probes)
        full_alg = sum(full_kl(LP0_alg[i], LP1_alg[i]).item() for i in range(na)) / na
        tr_alg = sum(truncated_kl(LP0_alg[i], LP1_alg[i], args.top_k).item() for i in range(na)) / na
        full_rnd = sum(full_kl(LP0_rnd[i], LP1_rnd[i]).item() for i in range(la)) / la
        tr_rnd = sum(truncated_kl(LP0_rnd[i], LP1_rnd[i], args.top_k).item() for i in range(la)) / la
        row = [r, full_alg, tr_alg, tr_alg / max(full_alg, 1e-12),
               full_rnd, tr_rnd, tr_rnd / max(full_rnd, 1e-12)]
        rows.append(row)
        print(" ".join(f"{x:>11.4f}" for x in row), flush=True)
        del m_r
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        f.write(" ".join(hdr) + "\n")
        for row in rows:
            f.write(" ".join(f"{x:.6f}" for x in row) + "\n")
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
