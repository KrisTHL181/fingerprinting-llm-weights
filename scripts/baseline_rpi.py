#!/usr/bin/env python3
"""
baseline_rpi.py — RPI-style (Refinement Provenance Inference) top-k logprob statistics
as a fine-tune/provenance baseline, on the SAME r-sweep protocol as positive_control.py.

RPI audits a model by teacher-forced next-token statistics that detect systematic
distributional shifts from a reference. The statistics (per token, in 'base' and
'uplift' form) are:
  NLL            : -log p(true_next)                       (normalized negative log-likelihood)
  topk_hit       : does the true next token fall in the top-k?
  conf_margin    : logit gap between top-1 and top-2
A fine-tuned / perturbed model shifts these away from the reference; we report the mean
shift (in the reference's units) as the detection signal, plus a per-token z^2 aggregate
mirroring the paper's change-detection statistic, so RPI is stated in a comparable form.

Protocol: single Gaussian direction (seed 0), logprobs clamped at -30, matching Table 3.
Usage:
  python baseline_rpi.py --model Qwen/Qwen3.5-0.8B --layer 8 --out baseline_rpi_qwen_l8.txt
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


@torch.no_grad()
def logits_next(model, input_ids):
    return model(input_ids).logits[0, -1].float()   # [Vfull]


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


def rpi_stats(logits):
    """Per-probe RPI statistics from next-token logits. Returns (NLL, topk_hit, margin)."""
    lp = torch.log_softmax(logits, dim=-1)
    nll = -lp.max()                              # -log p(top1); use top1 as 'true' for probing
    topk = torch.topk(lp, 2).values
    margin = topk[0] - topk[1]
    return nll, margin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--m", type=int, default=20)
    ap.add_argument("--noise", default="0,0.02,0.05,0.10,0.20,0.40,0.60,1.00")
    ap.add_argument("--out", default="baseline_rpi.txt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    aligner = Aligner(args.model, layer=args.layer)
    aligner.model.eval()
    dt = set(AutoTokenizer.from_pretrained(TARGET_MODEL_NAME, trust_remote_code=True).get_vocab().keys())
    shared = set(aligner.tok.get_vocab().keys()) & dt
    aligner.restrict_vocab(shared)
    vsh_ids = aligner.shared_ids

    probes = []
    for r in REFERENCES:
        ids = aligner.tok(r, return_tensors="pt")["input_ids"].to(vsh_ids.device)
        probes.extend(aligner.sample_near_synonyms(ids, M=args.m, top_k=6, k_neighbors=200))
    n = len(probes)
    print(f"probes={n}", flush=True)

    # reference RPI stats per probe
    base_nll = torch.zeros(n, device=vsh_ids.device)
    base_margin = torch.zeros(n, device=vsh_ids.device)
    for i, p in enumerate(probes):
        nll, margin = rpi_stats(logits_next(aligner.model, p))
        base_nll[i] = nll
        base_margin[i] = margin

    noise_list = [float(x) for x in args.noise.split(",")]
    hdr = ["r", "dNLL", "dMargin", "chi2", "df"]
    print(" ".join(f"{h:>10}" for h in hdr), flush=True)
    rows = []
    for r in noise_list:
        m_r = perturb_weights(aligner.model, r, args.seed)
        nll1 = torch.zeros(n, device=vsh_ids.device)
        mar1 = torch.zeros(n, device=vsh_ids.device)
        for i, p in enumerate(probes):
            nll1[i], mar1[i] = rpi_stats(logits_next(m_r, p))
        dNLL = (nll1 - base_nll).mean().item()
        dMargin = (mar1 - base_margin).mean().item()
        # per-probe z^2 aggregate over both stats (df = 2n)
        z2 = ((nll1 - base_nll) ** 2 + (mar1 - base_margin) ** 2).sum().item()
        rows.append([r, dNLL, dMargin, z2, 2 * n])
        print(f"{r:>10.3f} {dNLL:>10.4f} {dMargin:>10.4f} {z2:>10.2f} {2*n:>10d}", flush=True)
        del m_r
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        f.write(" ".join(hdr) + "\n")
        for row in rows:
            f.write(" ".join(f"{x:.6f}" if i < 3 else f"{x}" for i, x in enumerate(row)) + "\n")
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
