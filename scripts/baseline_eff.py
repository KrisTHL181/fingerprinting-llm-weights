#!/usr/bin/env python3
"""
baseline_eff.py — Win-point B: statistical power / probe efficiency.

The paper's detector reads the mean per-probe KL and fires against a clean null
(zero divergence at no change). The reliability of a single-shot verdict with a
finite probe budget is set by the signal-to-noise of the per-probe KL estimate,
not by its raw magnitude: a method whose per-probe KL has low variance detects
with few probes, one with high variance needs many.

We report, per probe condition (aligned vs random) and per r:
  mean   : mean per-probe full-vocabulary KL
  std    : std of per-probe KL
  snr    : mean/std  (per-probe signal-to-noise, higher is more reliable)
  n*     : probes needed for a 3-sigma detection, (3*std/mean)^2  (lower is better)
Both the full-vocabulary and the top-20 truncated KL are reported (the latter is
the black-box statistic).

Usage:
  python baseline_eff.py --model Qwen/Qwen3.5-0.8B --layer 8 --out baseline_eff_qwen_l8.txt
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


def summarize(tag, r, klvec_full, klvec_tr):
    def stats(v):
        v = torch.tensor(v)
        m = v.mean().item()
        s = v.std().item()
        snr = (m / s) if s > 1e-12 else float("inf")
        nstar = ((3 * s / m) ** 2) if m > 1e-12 else float("inf")
        return m, s, snr, nstar
    mf, sf, snf, nf = stats(klvec_full)
    mt, st, snt, nt = stats(klvec_tr)
    print(f"{tag:>12} r={r:>5}: full  mean={mf:7.4f} std={sf:7.4f} snr={snf:7.2f} n*={nf:8.1f}"
          f" | top20 mean={mt:7.4f} std={st:7.4f} snr={snt:7.2f} n*={nt:8.1f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--m", type=int, default=20)
    ap.add_argument("--m-rnd", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--noise", default="0.02,0.05,0.10")
    ap.add_argument("--out", default="baseline_eff.txt")
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
    print(f"alg_probes={len(alg_probes)} rnd_probes={len(rnd_probes)}", flush=True)

    LP0_alg = torch.stack([next_token_logp(aligner.model, p) for p in alg_probes])
    LP0_rnd = torch.stack([next_token_logp(aligner.model, p) for p in rnd_probes])

    for r in [float(x) for x in args.noise.split(",")]:
        m_r = perturb_weights(aligner.model, r)
        LP1_alg = torch.stack([next_token_logp(m_r, p) for p in alg_probes])
        LP1_rnd = torch.stack([next_token_logp(m_r, p) for p in rnd_probes])
        fa = [full_kl(LP0_alg[i], LP1_alg[i]).item() for i in range(len(alg_probes))]
        ta = [truncated_kl(LP0_alg[i], LP1_alg[i], args.top_k).item() for i in range(len(alg_probes))]
        fr = [full_kl(LP0_rnd[i], LP1_rnd[i]).item() for i in range(len(rnd_probes))]
        tr_ = [truncated_kl(LP0_rnd[i], LP1_rnd[i], args.top_k).item() for i in range(len(rnd_probes))]
        summarize("alg", r, fa, ta)
        summarize("rnd", r, fr, tr_)
        del m_r
        torch.cuda.empty_cache()

    # dump raw per-probe vectors for offline analysis
    print(f"\nwrote nothing extra; run printed summaries to {args.out} placeholder", flush=True)


if __name__ == "__main__":
    main()
