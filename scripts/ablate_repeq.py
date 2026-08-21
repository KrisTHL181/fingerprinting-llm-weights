#!/usr/bin/env python3
"""
ablate_repeq.py — Is representation-equivalence of the probes necessary?

Three probe conditions under the SAME weight-perturbation protocol (r grid, seed 0)
and the SAME full-vocabulary KL (clamp -30) used for Table tab:calib:

  eq  : representation-equivalent probes (sample_near_synonyms, cos~0.98 to h*)
  rnd : random token sequences, matched in length to each reference, tokens drawn
        uniformly from the shared subword vocab (same probe count) - NOT aligned
  ref : the reference sentences themselves, no synthesis (trivial baseline)

If D_KL(r) is markedly larger / more sensitive for eq than for rnd and ref at the
same r, the representation-equivalence property is load-bearing. If all coincide,
it is not. We also report mean cosine to the reference latent h* per condition to
confirm the manipulation actually separates the conditions in representation space.
"""
from __future__ import annotations
import argparse, copy
import torch
from transformers import AutoTokenizer
from align import Aligner, pool_hidden
TARGET_MODEL_NAME = "deepseek-ai/DeepSeek-V4-Flash-0731"
REFERENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "The committee approved the new budget proposal this morning.",
    "She walked slowly through the old town under the rain.",
]

@torch.no_grad()
def next_token_logp(model, input_ids):
    logits = model(input_ids).logits[0, -1].float()
    return torch.log_softmax(logits, dim=-1)      # [Vfull]

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

@torch.no_grad()
def full_kl(lp0, lpr, floor):
    p0 = lp0.exp()
    return (p0 * (lp0.clamp(min=floor) - lpr.clamp(min=floor))).sum()

@torch.no_grad()
def cos_to_ref(model, aligner, probes, ref_ids):
    """mean cosine between each probe's pooled hidden and the reference pooled hidden."""
    layer = aligner.layer
    mask = (ref_ids != aligner.tok.pad_token_id).to(model.dtype)
    h_star = pool_hidden(model(ref_ids, output_hidden_states=True).hidden_states[layer], mask, aligner.pool)
    cos = 0.0
    for p in probes:
        pmask = (p != aligner.tok.pad_token_id).to(model.dtype)
        hp = pool_hidden(model(p, output_hidden_states=True).hidden_states[layer], pmask, aligner.pool)
        cos += torch.cosine_similarity(hp, h_star, dim=-1).item()
    return cos / len(probes)

def build_probe_sets(aligner, m_per_ref, seed):
    """Return (sets dict cond->list[tensor[1,T]], mean_cos dict)."""
    rng = torch.Generator(device="cuda").manual_seed(seed)
    shared_ids = aligner.shared_ids
    sets = {"eq": [], "rnd": [], "ref": []}
    for ref in REFERENCES:
        ids = aligner.tok(ref, return_tensors="pt")["input_ids"].to(aligner.vocab.device)
        T = ids.shape[1]
        # eq: near-synonyms, drop the reference itself (pure synthesized set)
        syn = aligner.sample_near_synonyms(ids, M=m_per_ref, top_k=6, k_neighbors=200)
        syn = [s for s in syn if not torch.equal(s, ids)][:m_per_ref]
        sets["eq"].extend(syn)
        # rnd: random length-matched sequences over shared vocab (same count)
        for _ in range(m_per_ref):
            rnd = shared_ids[torch.randint(0, len(shared_ids), (1, T), generator=rng, device="cuda")]
            sets["rnd"].append(rnd)
        # ref: the reference itself (m_per_ref copies = same probe count, fair count control)
        sets["ref"].extend([ids.clone()] * m_per_ref)
    # mean cos to the reference latent for each condition
    mean_cos = {}
    for cond in ["eq", "rnd", "ref"]:
        c = 0.0
        for ref in REFERENCES:
            ids = aligner.tok(ref, return_tensors="pt")["input_ids"].to(aligner.vocab.device)
            sub = [p for p in sets[cond] if p.shape == ids.shape][:m_per_ref]
            c += cos_to_ref(aligner.model, aligner, sub, ids)
        mean_cos[cond] = c / len(REFERENCES)
    return sets, mean_cos

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--m", type=int, default=20)
    ap.add_argument("--floor", type=float, default=-30.0)
    ap.add_argument("--noise", default="0,0.02,0.05,0.10,0.20,0.40,0.60,1.00")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="../data/ablate_repeq.txt")
    args = ap.parse_args()

    aligner = Aligner(args.model, layer=args.layer)
    aligner.model.eval()
    dt = set(AutoTokenizer.from_pretrained(TARGET_MODEL_NAME, trust_remote_code=True).get_vocab().keys())
    shared = set(aligner.tok.get_vocab().keys()) & dt
    aligner.restrict_vocab(shared)
    print(f"model {args.model} layer {args.layer} shared-vocab {len(aligner.shared_ids)}", flush=True)

    sets, mean_cos = build_probe_sets(aligner, args.m, args.seed)
    for cond in ["eq", "rnd", "ref"]:
        print(f"cond {cond:>4}  n={len(sets[cond]):3d}  mean cos to h* = {mean_cos[cond]:.4f}", flush=True)

    noise_list = [float(x) for x in args.noise.split(",")]
    header = "cond n cos  " + " ".join(f"r={r:.2f}" for r in noise_list)
    print("\n" + header, flush=True)

    out_rows = []
    for cond in ["eq", "rnd", "ref"]:
        probes = sets[cond]
        LP0 = torch.stack([next_token_logp(aligner.model, p) for p in probes])
        kl = []
        for r in noise_list:
            m_r = perturb_weights(aligner.model, r, args.seed)
            LPr = torch.stack([next_token_logp(m_r, p) for p in probes])
            kl.append(sum(full_kl(LP0[i], LPr[i], args.floor).item() for i in range(len(probes))) / len(probes))
            del m_r; torch.cuda.empty_cache()
        line = f"{cond:>4} {len(probes):>3} {mean_cos[cond]:.4f} " + " ".join(f"{v:.4f}" for v in kl)
        print(line, flush=True)
        out_rows.append(line)

    with open(args.out, "w") as f:
        f.write(header + "\n")
        for rline in out_rows:
            f.write(rline + "\n")
    print(f"\nwrote {args.out}", flush=True)

if __name__ == "__main__":
    main()
