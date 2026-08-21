"""
deepseek_covariance — 探测并检测 DeepSeek 的协方差指纹(同 llm_fingerprint 的 record/check/show)。

DeepSeek API 不暴露 hidden state, 只给 top-20 logprobs。因此把一组近义句当探针:
每个近义句(上下文)诱导一个 next-token logprob 分布;这些分布在【共享子词表】上的
样本协方差 Σ 与逐 token 均值, 就是模型当前的可观测指纹。

record: 探测(每上下文采样 S 次)并存档指纹 {逐 token 均值/std/计数, 协方差谱, 上下文}。
check : 用存档的相同上下文重探, 做两样本检验(逐 token z² -> χ² p 值) + 协方差相对
        Frobenius 发散度 D_cov, 判定 CHANGED / UNCHANGED。
show  : 打印已存档指纹摘要。

用法:
  record: python deepseek_covariance.py record --out ds_cov.json [--samples 3]
  check : python deepseek_covariance.py check  --out ds_cov.json [--alpha 1e-3]
  show  : python deepseek_covariance.py show   --out ds_cov.json
"""
from __future__ import annotations
import argparse, json, math, time, urllib.request
import numpy as np
from scipy.stats import chi2
from transformers import AutoTokenizer
from align import Aligner

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
TARGET_MODEL_NAME = "deepseek-ai/DeepSeek-V4-Flash-0731"
API_URL = "https://api.deepseek.com/v1/chat/completions"
KEY = open("/root/.ds_key").read().strip()
DS_MODEL = "deepseek-v4-flash"

REFERENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "The committee approved the new budget proposal this morning.",
    "She walked slowly through the old town under the rain.",
]


# ---------------------------------------------------------------------------
# API + 共享词表
# ---------------------------------------------------------------------------
def ds_top_logprobs(prefix: str, top_n: int = 20):
    """对 prefix 请求 DeepSeek next-token top-n logprobs (thinking disabled, 无 temperature)。"""
    body = {"model": DS_MODEL, "max_tokens": 1,
            "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": prefix}],
            "logprobs": True, "top_logprobs": top_n}
    req = urllib.request.Request(API_URL, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json",
                                          "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        j = json.loads(resp.read())
    lp = j["choices"][0].get("logprobs") or {}
    arr = lp.get("content") or lp.get("reasoning_content") or []
    out = []
    if arr:
        for c in arr[0]["top_logprobs"]:
            if c["logprob"] > -9990:
                out.append((c["token"], c["logprob"]))
    return out


def build_probe_contexts(aligner, contexts_per_ref: int):
    """用近义候选句(共享词表内)构建探针上下文集。"""
    ctx = []
    for ref in REFERENCES:
        ids = aligner.tok(ref, return_tensors="pt")["input_ids"].to("cuda")
        cands = aligner.sample_near_synonyms(ids, M=contexts_per_ref, top_k=6,
                                             k_neighbors=200)
        for c in cands:
            ctx.append(aligner.decode(c)[0])
    return ctx


def probe(aligner, shared, contexts, samples, top_n):
    """对每个上下文采样 samples 次, 返回:
       - token_stats: {token: [obs_logp, ...]}  (只含观测到的共享 token)
       - ctx_means:   [{token: mean_logp}, ...] (每上下文对 samples 取均值)"""
    token_obs: dict[str, list[float]] = {}
    ctx_means = []
    for i, c in enumerate(contexts):
        runs = [dict(ds_top_logprobs(c, top_n)) for _ in range(samples)]
        # 观测到的共享 token 并集
        seen = {t for r in runs for t in r if t in shared}
        per = {}
        for t in seen:
            vals = [r[t] for r in runs if t in r]
            per[t] = float(np.mean(vals))
            token_obs.setdefault(t, []).extend(vals)
        ctx_means.append(per)
        print(f"  [{i+1}/{len(contexts)}] ctx={c!r} shared_top={len(seen)}", flush=True)
    return token_obs, ctx_means


def mean_std(xs):
    n = len(xs)
    if n == 0:
        return None, None, 0
    m = sum(xs) / n
    if n < 2:
        return m, None, n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(max(var, 0.0)), n


def covariance_from_ctx(ctx_means, floor=-15.0):
    """由每上下文均值向量构建 X 并算协方差。返回 (cov, V, X, means)。"""
    V = sorted({t for m in ctx_means for t in m})
    X = np.full((len(ctx_means), len(V)), floor, dtype=float)
    for i, m in enumerate(ctx_means):
        for t, v in m.items():
            X[i, V.index(t)] = v
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / max(len(ctx_means) - 1, 1)
    return cov, V, X, X.mean(axis=0)


# ---------------------------------------------------------------------------
# χ² 生存函数(用 scipy)
# ---------------------------------------------------------------------------
def chi2_sf(x2, df):
    return float(chi2.sf(x2, df))


# ---------------------------------------------------------------------------
# record / check / show
# ---------------------------------------------------------------------------
def do_record(args):
    aligner = Aligner(MODEL_NAME, layer=8)
    dt = set(AutoTokenizer.from_pretrained(TARGET_MODEL_NAME,
                                           trust_remote_code=True).get_vocab().keys())
    shared = set(aligner.tok.get_vocab().keys()) & dt
    aligner.restrict_vocab(shared)
    print(f"shared vocab: {len(shared)}", flush=True)
    contexts = build_probe_contexts(aligner, args.contexts_per_ref)
    print(f"probe contexts: {len(contexts)}  samples/ctx: {args.samples}", flush=True)
    token_obs, ctx_means = probe(aligner, shared, contexts, args.samples, args.top_n)
    cov, V, X, means = covariance_from_ctx(ctx_means, args.floor)
    eig = np.linalg.eigvalsh((cov + cov.T) / 2)
    eig = eig[eig > 1e-9][::-1]

    stats = {}
    for t, obs in token_obs.items():
        m, s, n = mean_std(obs)
        stats[t] = {"mean": m, "std": s, "count": n}

    fp = {
        "schema": 1, "model": DS_MODEL, "collected_at": int(time.time()),
        "contexts": contexts, "n_contexts": len(contexts),
        "samples": args.samples, "top_n": args.top_n, "floor": args.floor,
        "token_stats": stats, "cov_eigvals": [float(e) for e in eig],
        "cov_trace": float(cov.trace()), "cov_frob": float(np.linalg.norm(cov)),
        "mean_logp": {t: float(v) for t, v in zip(V, means)},
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(fp, f, ensure_ascii=False, indent=1)
    print(f"[record] 已存档指纹: {args.out}  (tokens={len(V)}, rank={len(eig)})")
    print(f"[record] cov_trace={cov.trace():.2f}  top3占比="
          f"{eig[:3].sum()/eig.sum()*100:.0f}%")


def do_check(args):
    with open(args.out, encoding="utf-8") as f:
        old = json.load(f)
    aligner = Aligner(MODEL_NAME, layer=8)
    dt = set(AutoTokenizer.from_pretrained(TARGET_MODEL_NAME,
                                           trust_remote_code=True).get_vocab().keys())
    shared = set(aligner.tok.get_vocab().keys()) & dt
    aligner.restrict_vocab(shared)
    contexts = old["contexts"]
    samples = args.samples or old["samples"]
    print(f"[check] 重探 {len(contexts)} 上下文 x {samples} 采样", flush=True)
    token_obs, ctx_means = probe(aligner, shared, contexts, samples, old["top_n"])
    cov_new, V_new, _, mean_new = covariance_from_ctx(ctx_means, old["floor"])
    eig_new = np.linalg.eigvalsh((cov_new + cov_new.T) / 2)
    eig_new = eig_new[eig_new > 1e-9][::-1]

    # ---- 逐 token 两样本 z² (仅比较两期都有真实数据的 token) ----
    zs, df = [], 0
    for t, st in old["token_stats"].items():
        if st["mean"] is None or t not in token_obs:
            continue
        m1, s1, n1 = st["mean"], st["std"], st["count"]
        m2, s2, n2 = mean_std(token_obs[t])
        if m2 is None or s1 is None or s2 is None or not (n1 and n2):
            continue
        se = math.sqrt(s1 * s1 / n1 + s2 * s2 / n2)
        if se <= 0:
            continue
        zs.append(((m2 - m1) / se) ** 2)
        df += 1
    p_logp = chi2_sf(sum(zs), df) if df else None

    # ---- 均值向量相对发散(公共 token 子集) ----
    Vcom = [t for t in V_new if t in old["mean_logp"]]
    d_mean = None
    if len(Vcom) >= 3:
        a = np.array([old["mean_logp"][t] for t in Vcom])
        b = np.array([mean_new[V_new.index(t)] for t in Vcom])
        d_mean = float(np.linalg.norm(a - b) / np.linalg.norm(a))

    # ---- 协方差谱相对发散 ----
    eig_old = np.array(old["cov_eigvals"])
    d_cov = (float(np.linalg.norm(eig_new[: len(eig_old)] - eig_old) /
                   np.linalg.norm(eig_old)) if len(eig_old) else None)

    # ---- 判定: 显著 p 值, 或均值/谱发散超过阈值 ----
    changed = (p_logp is not None and p_logp < args.alpha) or \
              (d_mean is not None and d_mean > args.thr) or \
              (d_cov is not None and d_cov > args.thr)
    print(f"\n[check] 逐 token z² 检验: chi2={sum(zs):.2f} df={df} p="
          f"{p_logp:.4g}" if p_logp is not None else "\n[check] z² 检验: n/a")
    print(f"[check] 均值向量相对发散 D_mean={d_mean:.3f}" if d_mean is not None else
          "[check] 均值发散: n/a")
    print(f"[check] 协方差谱相对发散 D_cov={d_cov:.3f}" if d_cov is not None else
          "[check] 协方差谱发散: n/a")
    verdict = "CHANGED" if changed else \
        ("LIKELY_CHANGED" if (p_logp is not None and p_logp < 0.05) else "UNCHANGED")
    print(f"[check] 结论: {verdict}  (alpha={args.alpha}, thr={args.thr})")


def do_show(args):
    with open(args.out, encoding="utf-8") as f:
        fp = json.load(f)
    print(json.dumps({k: fp[k] for k in
                      ("model", "collected_at", "n_contexts", "samples", "top_n",
                       "cov_trace", "cov_frob", "cov_eigvals", "mean_logp")},
                     ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser(prog="deepseek_covariance")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="{record,check,show}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", default="ds_cov.json")
    r = sub.add_parser("record", parents=[common])
    r.add_argument("--contexts-per-ref", type=int, default=7)
    r.add_argument("--samples", type=int, default=2)
    r.add_argument("--top-n", type=int, default=20)
    r.add_argument("--floor", type=float, default=-15.0)
    c = sub.add_parser("check", parents=[common])
    c.add_argument("--samples", type=int, default=None)
    c.add_argument("--alpha", type=float, default=1e-3)
    c.add_argument("--thr", type=float, default=0.30,
                   help="D_mean / D_cov 判定已改变的相对发散阈值")
    sub.add_parser("show", parents=[common])
    args = ap.parse_args()
    {"record": do_record, "check": do_check, "show": do_show}[args.cmd](args)


if __name__ == "__main__":
    main()
