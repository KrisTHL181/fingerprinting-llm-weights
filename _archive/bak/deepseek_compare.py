"""
最终应用: 在【共享子词表】上比较 Qwen 与 DeepSeek 的 next-token logprob, 推断权重相似性。

词表对齐(核心前提): 只在两模型词表交集(shared tokens)上比较。这些 token 在两模型中
是【完全相同的原始字符串】, 因此可逐 token 对齐比较 —— 避免之前不同 tokenizer 表示导致的假零重叠。
  - 上下文 = 参考句去掉末词(共享前缀)。
  - Qwen:    本地全词表 next-token logprob, 只取 shared token。
  - DeepSeek: 无 temperature + thinking disabled, top-20 logprobs, 只取 shared token。
  - 指纹/度量: shared top-k 的 Jaccard 重叠 + 共享 token 的 logp 相关/余弦。
结合 covariance_experiment 标定: 同 tokenizer 下 D_logp ~= 5.22*r(对照 Qwen vs Qwen+噪声)。
"""
from __future__ import annotations
import json, urllib.request
import torch, torch.nn.functional as F
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


@torch.no_grad()
def qwen_top_logprobs(model, tok, prefix_ids, top_n=20):
    logits = model(prefix_ids).logits[0, -1, :]          # [V]
    lp = F.log_softmax(logits.float(), dim=-1)
    vals, ids = lp.topk(top_n)
    return [(tok.convert_ids_to_tokens(int(i)), v.item()) for i, v in zip(ids, vals)]


def main():
    aligner = Aligner(MODEL_NAME, layer=8)
    model, tok = aligner.model, aligner.tok

    # ---- 共享子词表: 两 tokenizer 的原始字符串交集 ----
    qw = set(tok.get_vocab().keys())
    dt = set(AutoTokenizer.from_pretrained(TARGET_MODEL_NAME,
                                           trust_remote_code=True).get_vocab().keys())
    shared = qw & dt
    print(f"Qwen vocab={len(qw)}  DeepSeek vocab={len(dt)}  shared={len(shared)}")
    aligner.restrict_vocab(shared)

    cos_sum, corr_sum, jac_sum, sh_sum, n = 0.0, 0.0, 0.0, 0.0, 0
    for r, ref in enumerate(REFERENCES):
        ids = tok(ref, return_tensors="pt")["input_ids"].to("cuda")
        T = ids.shape[1]
        prefix_ids = ids[:, : T - 1]                       # 去掉末词
        prefix_text = aligner.decode(prefix_ids)[0]
        q = [(t, lp) for t, lp in qwen_top_logprobs(model, tok, prefix_ids, 40) if t in shared]
        d = [(t, lp) for t, lp in ds_top_logprobs(prefix_text, 20) if t in shared]
        q_set = {t for t, _ in q}
        d_set = {t for t, _ in d}
        shared_hit = q_set & d_set
        jac = len(shared_hit) / len(q_set | d_set) if (q_set | d_set) else 0.0
        dq = {t: lp for t, lp in q if t in shared_hit}
        dd = {t: lp for t, lp in d if t in shared_hit}
        cos = corr = float("nan")
        if len(shared_hit) >= 2:
            qq = torch.tensor([dq[t] for t in shared_hit], dtype=torch.float)
            ddt = torch.tensor([dd[t] for t in shared_hit], dtype=torch.float)
            cos = F.cosine_similarity(qq.unsqueeze(0), ddt.unsqueeze(0)).item()
            corr = torch.corrcoef(torch.stack([qq, ddt]))[0, 1].item()
        cos_sum += cos if cos == cos else 0.0
        corr_sum += corr if corr == corr else 0.0
        jac_sum += jac
        sh_sum += len(shared_hit)
        n += 1
        print(f"[{r}] ctx={prefix_text!r}\n"
              f"    shared next-tokens: Qwen={len(q_set)} DS={len(d_set)} hit={len(shared_hit)} "
              f"Jaccard={jac:.3f} cos={cos:.3f} corr={corr:.3f}",
              flush=True)
        for t in sorted(shared_hit)[:6]:
            print(f"      {t!r}  Qwen={dq[t]:.2f}  DS={dd[t]:.2f}")

    if n:
        print(f"\n=== 汇总 (共享词表对齐, {n} 个上下文) ===")
        print(f"shared top-k Jaccard 均值: {jac_sum/n:.3f}  平均命中数: {sh_sum/n:.1f}")
        print(f"命中 token logp 余弦均值: {cos_sum/n:.3f}  相关均值: {corr_sum/n:.3f}")
        print("解读: 越接近 1 表示两模型在共享词表上的预测越一致 -> 权重/预测越相似。")


if __name__ == "__main__":
    main()
