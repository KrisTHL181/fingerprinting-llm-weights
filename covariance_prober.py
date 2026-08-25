"""
covariance_prober — 通用 model endpoint 的协方差指纹探测 / 检测 CLI

deepseek_covariance.py 的通用化: 不硬编码任何 endpoint / model / api key, 改为
命令行参数(或环境变量)传入, 可对任意 OpenAI 兼容的 chat/completions 接口做
next-token logprob 协方差指纹探测。

指纹原理(同 deepseek_covariance): 一组近义句作为探针, 每个上下文诱导一个
next-token logprob 分布; 这些分布在【共享子词表】上的样本协方差 Σ 与逐 token
均值, 就是模型当前的可观测指纹。

record: 探测(每上下文采样 S 次)并存档指纹 {逐 token 均值/std/计数, 协方差谱, 上下文}。
check : 用存档的相同上下文重探, 做两样本检验(逐 token z² -> χ² p 值) + 协方差相对
        Frobenius 发散度 D_cov, 判定 CHANGED / UNCHANGED。
show  : 打印已存档指纹摘要。

用法:
  export COV_API_KEY=sk-...
  python covariance_prober.py record --endpoint https://api.deepseek.com/v1/chat/completions \\
      --model deepseek-v4-flash --target-tokenizer deepseek-ai/DeepSeek-V4-Flash-0731 \\
      --thinking disabled --out data/cov.json
  python covariance_prober.py check --endpoint <同上> --model <同上> --out data/cov.json
  python covariance_prober.py show  --out data/cov.json

参数传入(命令行优先, 环境变量兜底):
  --endpoint  接口 URL           环境变量 COV_ENDPOINT
  --model     请求时用的模型名    环境变量 COV_MODEL
  --api-key   Bearer token      环境变量 COV_API_KEY (无鉴权本地端点可省略)

约束:
  * 假定接口为 OpenAI 兼容格式 (logprobs: true, top_logprobs: N)。
  * --model 是发送给 API 的模型名; --target-tokenizer 是构建共享词表用的本地 HF
    仓库名(默认取 --model)。两者经常不同(如 API 用 "deepseek-v4-flash"、
    本地 tokenizer 用 "deepseek-ai/DeepSeek-V4-Flash-0731")。
  * --thinking 控制是否下发 DeepSeek 的 thinking 字段(disabled / enabled / none)。
  * 共享词表 + 近义上下文构建很贵(加载本地 aligner + GPU 同义句搜索), 但确定性 →
    首次构建后写入探针缓存(默认 data/probe_cache_<target-tokenizer>.json), 之后
    record/check 直接复用。缓存按 target-tokenizer 隔离, 换模型不互相污染。

依赖: torch, transformers, scipy, numpy(需 GPU 用于本地 aligner 的近义句搜索)。
align.py 的 Aligner 已内联进本文件。
"""
from __future__ import annotations
import argparse, json, math, os, time, urllib.request
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import chi2
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")

DEFAULT_ALIGNER_MODEL = "Qwen/Qwen3.5-0.8B"
DEFAULT_REFERENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "The committee approved the new budget proposal this morning.",
    "She walked slowly through the old town under the rain.",
]

# 探针构建参数(与 build_probe_contexts 内的调用一致, 用于缓存校验)
PROBE_TOP_K = 6
PROBE_K_NEIGHBORS = 200
PROBE_CACHE_SCHEMA = 1


# ===========================================================================
# 以下为 align.py 内联: 中间层表征对齐的同义句生成(近义上下文探针的构建核心)
# 思路: 给定参考句 S,取小模型中间层 L 的 hidden state h*。在受限子词表内离散
# 优化输入 tokens 构成 S',使 |h(S',L)-h*| 足够小(≈含义相同),同时强制 S' 与 S
# 表面不同(多样性正则)→ 得到"潜在同义句"。HotFlip 风格梯度引导束搜索。
# ===========================================================================

def pool_hidden(h: torch.Tensor, mask: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """h: [B,T,D], mask: [B,T] (1=有效,0=padding)。mode: mean|last|first"""
    if mode == "mean":
        return (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
    if mode == "last":
        last = mask.sum(1) - 1  # [B]
        return h[torch.arange(h.shape[0]), last]
    if mode == "first":
        return h[:, 0]
    raise ValueError(mode)


class Aligner:
    """在受限词表内搜索与参考句中间层表征对齐的离散 token 序列。"""

    def __init__(
        self,
        model_name: str,
        layer: int | list[int],           # 中间层编号(1-based,hidden_states[layer])或一组层
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        pool: str = "mean",
    ):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, device_map=device
        )
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.layers = [layer] if isinstance(layer, int) else list(layer)
        self.layer = self.layers[0]       # 主层(用于显示/兼容)
        self.pool = pool
        self.dtype = dtype
        self.vocab = self.model.get_input_embeddings().weight.detach()  # [V,D]
        self.shared_ids = None
        self.model.eval()

    # -- 共享子词表过滤 -----------------------------------------------------
    def restrict_vocab(self, shared_vocab: set[str] | None = None):
        """把非共享 token 的嵌入得分屏蔽。shared_vocab 为 tokenizer 原样 token 的集合。"""
        if shared_vocab is None:
            self.shared_mask = None
            self.shared_ids = None
            return
        ids = [i for i, t in enumerate(self.tok.get_vocab().keys()) if t in shared_vocab]
        # 屏蔽其余 token 的候选(不给候选集),但保留一个"不替换"选项
        self.shared_ids = torch.tensor(ids, device=self.vocab.device)
        # 用于快速 top-k: 只在这些 id 里选
        self._shared_id_set = ids

    @torch.no_grad()
    def hidden(self, input_ids: torch.Tensor, layer: int | None = None):
        """取指定层 hidden states [B,T,D] 及其 mask。"""
        layer = layer or self.layer
        out = self.model(input_ids, output_hidden_states=True)
        h = out.hidden_states[layer]
        mask = (input_ids != self.tok.pad_token_id).to(h.dtype)
        return h, mask

    @torch.no_grad()
    def reference(self, input_ids: torch.Tensor):
        """参考句在各对齐层的 pooled 表征 h* —— 返回 list[ [1,D] ](每层一个)"""
        out = self.model(input_ids, output_hidden_states=True)
        mask = (input_ids != self.tok.pad_token_id).to(self.vocab.dtype)
        return [pool_hidden(out.hidden_states[l], mask, self.pool) for l in self.layers]

    # -- 梯度引导候选生成 ---------------------------------------------------
    def candidate_scores(
        self, input_ids: torch.Tensor, h_star: torch.Tensor, keep_mask: torch.Tensor | None = None,
        div_ref: torch.Tensor | None = None, div_weight: float = 0.0,
    ) -> torch.Tensor:
        """一阶近似的每位置·每 token 的 loss 变化量。返回 [B,T,V]。
        keep_mask: [B,T] 可选,只对 keep==1 的位置产生候选(其它位置候选设 inf,表示不替换)。
        div_ref: [B,T] 参考 token id 序列; div_weight>0 时,梯度额外引导"远离 div_ref 对应 token
                 的嵌入"(多样性),解决从 S 出发时 mse 梯度≈0 导致候选是噪声的问题。"""
        B, T = input_ids.shape
        emb = self.model.get_input_embeddings()
        inputs_embeds = emb(input_ids).detach().clone().requires_grad_(True)
        out = self.model(inputs_embeds=inputs_embeds, output_hidden_states=True)
        mask = (input_ids != self.tok.pad_token_id).to(inputs_embeds.dtype)
        # 多层对齐: 对所有对齐层求 pooled hidden 的 MSE 之和
        mse_terms = [F.mse_loss(pool_hidden(out.hidden_states[l], mask, self.pool), h_l.detach())
                     for l, h_l in zip(self.layers, h_star)]
        loss = sum(mse_terms[1:], mse_terms[0])
        if div_weight > 0 and div_ref is not None:
            # 多样性: 使各位置嵌入远离 div_ref 对应 token 的嵌入(最大化负距离)
            E_full = emb.weight.detach()
            ref_emb = E_full[div_ref]                                   # [B,T,D]
            dist2 = (inputs_embeds - ref_emb).pow(2).mean(-1)           # [B,T]
            loss = loss - div_weight * dist2.masked_fill(mask == 0, 0).mean()
        self.model.zero_grad()
        loss.backward()
        grad = inputs_embeds.grad  # [B,T,D]
        # 一阶近似: 替换为 token t 引起的 loss 变化 ≈ grad_emb · e_t
        E = emb.weight.detach()  # [V,D]
        scores = torch.einsum("btd,vd->btv", grad, E)  # [B,T,V]
        # 屏蔽: 非共享 token + (若指定)非候选位置 → +inf (永远不被选为"替换")
        if self.shared_ids is not None:
            full = torch.full_like(scores, float("inf"))
            full[:, :, self.shared_ids] = scores[:, :, self.shared_ids]
            scores = full
        if keep_mask is not None:
            scores = scores.masked_fill(keep_mask.unsqueeze(-1) == 0, float("inf"))
        return scores

    def top_candidates(self, input_ids, h_star, top_k, keep_mask=None,
                       div_ref=None, div_weight=0.0):
        """每位置 top_k 个替换候选 token id [B,T,k]"""
        scores = self.candidate_scores(input_ids, h_star, keep_mask, div_ref, div_weight)
        return scores.topk(top_k, dim=-1, largest=False).indices

    # -- 精确打分(批量前向) -------------------------------------------------
    @torch.no_grad()
    def evaluate(self, seqs: torch.Tensor, h_star: list[torch.Tensor], chunk: int = 48):
        """对一批等长序列 [B,T] 前向,返回 (mse [B], cos [B])。分块避免显存 OOM。"""
        B, T = seqs.shape
        mse = torch.zeros(B, device=seqs.device)
        cos = torch.zeros(B, device=seqs.device)
        for s in range(0, B, chunk):
            bseq = seqs[s : s + chunk]
            out = self.model(bseq, output_hidden_states=True)
            mask = (bseq != self.tok.pad_token_id).to(self.vocab.dtype)
            Bb = bseq.shape[0]
            for l, h_star_l in zip(self.layers, h_star):
                h = out.hidden_states[l]                       # [Bb,T,D]
                hs = pool_hidden(h, mask, self.pool)           # [Bb,D]
                mse[s:s+Bb] += F.mse_loss(hs, h_star_l.expand(Bb, -1), reduction="none").mean(1)
                cos[s:s+Bb] += F.cosine_similarity(hs, h_star_l.expand_as(hs), dim=-1)
        return mse, cos / len(self.layers)

    # -- 贪心 HotFlip(带多样性) --------------------------------------------
    def greedy_search(
        self,
        S_ids: torch.Tensor,          # 参考句 token ids [1,T]
        n_steps: int = 25,
        top_k: int = 12,
        diversity: float = 0.5,       # 多样性正则强度 λ: 越高越鼓励与 S 表面不同
        diversity_grad: float = 1.0,  # 候选生成的多样性梯度强度: 引导换远离 S 的词
        min_changes: int = 1,         # 至少替换这么多 token 才算"同义新句"
        max_align: float | None = None,  # 对齐达标 MSE 阈值,达到即停
        verbose: bool = True,
    ):
        """返回 (best_seq [1,T], {mse, cos, changes, history})"""
        S = S_ids.detach().clone()
        h_star = self.reference(S)
        cur = S.clone()
        best_seq = None
        best_meta = None
        history = []

        T = cur.shape[1]
        best_seq = None
        best_meta = None
        for step in range(n_steps):
            # 多样性梯度引导候选: 让每个候选位置倾向换成"远离 S 对应 token"的词,
            # 否则从 S 出发 mse 梯度≈0,候选全是噪声,换词推不动。
            cands = self.top_candidates(
                cur, h_star, top_k, div_ref=S, div_weight=diversity_grad)
            # 组装所有 (pos, cand) 候选序列
            seqs = []
            for i in range(T):
                for j in range(top_k):
                    c = cands[0, i, j].item()
                    if c == cur[0, i].item():
                        continue
                    seqs.append(cur[0].clone())
                    seqs[-1][i] = c
            if not seqs:
                break
            seq_batch = torch.stack(seqs)
            mse, cos = self.evaluate(seq_batch, h_star)              # [N]
            overlap = (seq_batch == S).float().mean(1)               # 越大越像 S
            # 组合得分: 越小越好 = mse + λ * overlap (min overlap → 多样)
            combined = mse + diversity * overlap
            idx = combined.argmin()
            nxt = seq_batch[idx:idx + 1].clone()
            applied = bool((nxt != cur).any())
            if applied:
                cur = nxt
            cur_mse = mse[idx].item()
            cur_cos = cos[idx].item()
            changes = (cur[0] != S[0]).sum().item()
            m = {"mse": cur_mse, "cos": cur_cos, "changes": changes, "step": step}
            history.append(m)
            if verbose:
                print(f"  step {step:2d} | mse {cur_mse:.4f} cos {cur_cos:.3f} changes {changes}/{T}")
            # 达标: 已换足 token 且对齐足够
            if max_align is not None and changes >= min_changes and cur_mse <= max_align:
                if verbose:
                    print("  -> alignment target reached")
                break
            # 保留"换词数最多"里对齐最好的作为候选输出
            if changes >= min_changes:
                if best_meta is None or m["mse"] < best_meta["mse"]:
                    best_seq, best_meta = cur.clone(), dict(m)
            # 收敛: 无替换可做
            if not applied:
                break
            # 全替换完
            if changes >= T:
                break

        # 优先返回达到 min_changes 多样性的 best_seq;否则退回最终 cur
        out_seq, out_meta = best_seq, best_meta
        if out_seq is None:
            out_seq = cur
            final_mse, final_cos = self.evaluate(cur, h_star)
            out_meta = {"mse": final_mse.item(), "cos": final_cos.item(),
                        "changes": (cur[0] != S[0]).sum().item()}
        out_meta["history"] = history
        return out_seq, out_meta

    def decode(self, ids: torch.Tensor) -> list[str]:
        return [self.tok.decode(s, skip_special_tokens=True) for s in ids]

    @torch.no_grad()
    def sample_near_synonyms(self, S: torch.Tensor, M: int = 20, top_k: int = 6,
                             k_neighbors: int = 200):
        """返回 M 个不同的近义候选(含 S 自身),用于估算表征协方差。
        基于嵌入近邻替换, 再用潜表征对齐(mse)筛选出最接近参考的 M 个不同序列。"""
        h_star = self.reference(S)
        cands = self.neighbor_candidates(S, k_neighbors, top_k)
        seqs = []
        for i in range(S.shape[1]):
            for c in cands[i]:
                if int(c) == int(S[0, i].item()):
                    continue
                seqs.append(S[0].clone())
                seqs[-1][i] = int(c)
        if not seqs:
            return [S]
        seq_batch = torch.stack(seqs)
        mse, _ = self.evaluate(seq_batch, h_star)
        uniq = {}
        for idx in range(len(seqs)):
            key = tuple(seq_batch[idx].tolist())
            if key not in uniq:
                uniq[key] = mse[idx].item()
        best = sorted(uniq.items(), key=lambda kv: kv[1])[: M]
        out = [S] + [torch.tensor(k, device=S.device).unsqueeze(0) for k, _ in best]
        return out

    # -- 受限同义词候选(嵌入空间近邻,限定共享词表) -------------------------
    @torch.no_grad()
    def neighbor_candidates(self, S: torch.Tensor, k_neighbors: int = 200, top_k: int = 8):
        """对 S [1,T] 每个 token,取嵌入空间最近邻中属于共享词表的 top-k 候选 token id。
        返回 list[T] of list[token_id](不含原 token、不含特殊 token)。"""
        E = self.vocab                                    # [V,D] 已 detach
        emb = self.model.get_input_embeddings()(S)        # [1,T,D]
        # 归一化算余弦
        En = E / E.norm(dim=-1, keepdim=True)
        bn = emb / emb.norm(dim=-1, keepdim=True)
        sim = bn[0] @ En.T                               # [T,V]
        special = {self.tok.pad_token_id, self.tok.eos_token_id,
                   self.tok.bos_token_id, self.tok.unk_token_id}
        special = {i for i in special if i is not None}
        cands_all = []
        for i in range(S.shape[1]):
            orig = int(S[0, i].item())
            # 屏蔽原 token 与特殊 token
            mask_v = torch.ones(self.vocab.shape[0], dtype=torch.bool, device=sim.device)
            mask_v[orig] = False
            for sp in special:
                mask_v[sp] = False
            simi = sim[i].masked_fill(~mask_v, -1e9)
            if self.shared_ids is not None:
                full = torch.full_like(simi, -1e9)
                full[self.shared_ids] = simi[self.shared_ids]
                simi = full
            top = simi.topk(k_neighbors).indices
            cands_all.append(top.tolist())
        return cands_all

    # -- 受限同义词搜索(潜表征作为选择目标) --------------------------------
    def synonym_search(
        self, S_ids: torch.Tensor,
        n_steps: int = 12, top_k: int = 8, k_neighbors: int = 300,
        min_changes: int = 2, verbose: bool = True,
    ):
        """在每个位置嵌入近邻(共享词表)中挑替换,使中间层表征最接近参考句。
        候选受限 → 输出流畅; 潜对齐作选择目标 → 保留含义。
        返回 (best_seq [1,T], meta)。"""
        S = S_ids.detach().clone()
        T = S.shape[1]
        h_star = self.reference(S)
        cands = self.neighbor_candidates(S, k_neighbors, top_k)   # list[T] of list
        cur = S.clone()
        history = []
        best_seq, best_meta = None, None
        for step in range(n_steps):
            # 所有 (pos, cand) 候选(每个 pos 用其 top-k 近义候选)
            seqs = []
            for i in range(T):
                for c in cands[i]:
                    if c == cur[0, i].item():
                        continue
                    seqs.append(cur[0].clone())
                    seqs[-1][i] = c
            if not seqs:
                break
            seq_batch = torch.stack(seqs)
            mse, cos = self.evaluate(seq_batch, h_star)
            overlap = (seq_batch == S).float().mean(1)            # 越小越多样
            # 选择目标: 优先低 mse(保持含义),同 mse 下更偏好多样(改更多词)
            combined = mse - 0.05 * (1.0 - overlap)              # 轻微奖励换词
            idx = combined.argmin()
            nxt = seq_batch[idx:idx + 1].clone()
            applied = bool((nxt != cur).any())
            if applied:
                cur = nxt
            changes = (cur[0] != S[0]).sum().item()
            m = {"mse": mse[idx].item(), "cos": cos[idx].item(),
                 "changes": changes, "step": step}
            history.append(m)
            if verbose:
                print(f"  step {step:2d} | mse {m['mse']:.4f} cos {m['cos']:.3f} changes {changes}/{T}")
            if changes >= min_changes and (best_meta is None or m["mse"] < best_meta["mse"]):
                best_seq, best_meta = cur.clone(), dict(m)
            if not applied:
                break
        if best_seq is None:
            best_seq, best_meta = cur, dict(history[-1]) if history else {"mse": 0.0, "cos": 1.0, "changes": 0}
        best_meta["history"] = history
        return best_seq, best_meta
# ===========================================================================


@dataclass
class Config:
    """一次探测的全部模型/接口参数(由 CLI/环境变量/存档指纹解析而来)。"""
    endpoint: str
    model: str
    api_key: str = ""
    target_tokenizer: str = ""
    aligner_model: str = DEFAULT_ALIGNER_MODEL
    thinking: str = "none"                     # "disabled" | "enabled" | "none"
    references: list = field(default_factory=lambda: list(DEFAULT_REFERENCES))


# ---------------------------------------------------------------------------
# API 调用(OpenAI 兼容)
# ---------------------------------------------------------------------------
def build_body(cfg: Config, prefix: str, top_n: int) -> dict:
    body = {"model": cfg.model, "max_tokens": 1,
            "messages": [{"role": "user", "content": prefix}],
            "logprobs": True, "top_logprobs": top_n}
    if cfg.thinking in ("disabled", "enabled"):
        body["thinking"] = {"type": cfg.thinking}
    return body


def top_logprobs(cfg: Config, prefix: str, top_n: int = 20):
    """对 prefix 请求 next-token top-n logprobs (无 temperature)。"""
    req = urllib.request.Request(cfg.endpoint,
                                 data=json.dumps(build_body(cfg, prefix, top_n)).encode(),
                                 headers={"content-type": "application/json"})
    if cfg.api_key:
        req.add_header("Authorization", f"Bearer {cfg.api_key}")
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


# ---------------------------------------------------------------------------
# 探针缓存: 共享词表 + 近义上下文 只构建一次, 之后复用 (按 target-tokenizer 隔离)
# ---------------------------------------------------------------------------
def default_cache_path(cfg: Config) -> str:
    slug = cfg.target_tokenizer.replace("/", "__")
    return os.path.join(DATA_DIR, f"probe_cache_{slug}.json")


def probe_cache_path(args, cfg: Config) -> str:
    return args.probe_cache or default_cache_path(cfg)


def _build_probe_artifacts(cfg: Config, args):
    """现场构建共享词表与探针上下文集(较贵: 加载模型 + GPU 搜索)。"""
    aligner = Aligner(cfg.aligner_model, layer=8)
    dt = set(AutoTokenizer.from_pretrained(cfg.target_tokenizer,
                                           trust_remote_code=True).get_vocab().keys())
    shared = set(aligner.tok.get_vocab().keys()) & dt
    aligner.restrict_vocab(shared)
    print(f"shared vocab: {len(shared)}", flush=True)
    contexts = build_probe_contexts(aligner, cfg, args.contexts_per_ref)
    return contexts, shared


def load_probe_cache(args, cfg: Config, require_params: bool):
    """读取探针缓存; 校验失败(或不存在)返回 None。
    require_params=True 时校验完整构建参数, 保证缓存的上下文与当前参数一致。"""
    path = probe_cache_path(args, cfg)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        cache = json.load(f)
    if cache.get("schema") != PROBE_CACHE_SCHEMA:
        return None
    if require_params and (
        cache.get("aligner_model") != cfg.aligner_model
        or cache.get("target_model") != cfg.target_tokenizer
        or cache.get("references") != cfg.references
        or cache.get("contexts_per_ref") != args.contexts_per_ref
        or cache.get("top_k") != PROBE_TOP_K
        or cache.get("k_neighbors") != PROBE_K_NEIGHBORS
    ):
        return None
    return cache


def _write_probe_cache(args, cfg: Config, contexts, shared):
    path = probe_cache_path(args, cfg)
    cache = {
        "schema": PROBE_CACHE_SCHEMA,
        "aligner_model": cfg.aligner_model, "target_model": cfg.target_tokenizer,
        "references": cfg.references,
        "contexts_per_ref": args.contexts_per_ref,
        "top_k": PROBE_TOP_K, "k_neighbors": PROBE_K_NEIGHBORS,
        "built_at": int(time.time()),
        "contexts": contexts, "shared_vocab": sorted(shared),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)
    print(f"[cache] 已写入探针缓存: {path}", flush=True)


def load_or_build_probe(args, cfg: Config):
    """返回 (contexts, shared)。缓存有效直接复用(不加载 aligner),
    否则现场构建并写缓存。"""
    cache = load_probe_cache(args, cfg, require_params=True)
    if cache is not None:
        print(f"[cache] 复用探针缓存: {probe_cache_path(args, cfg)}  "
              f"({len(cache['contexts'])} 上下文, {len(cache['shared_vocab'])} 共享词表)",
              flush=True)
        return cache["contexts"], set(cache["shared_vocab"])
    contexts, shared = _build_probe_artifacts(cfg, args)
    _write_probe_cache(args, cfg, contexts, shared)
    return contexts, shared


def get_shared_vocab(args, cfg: Config):
    """取共享词表(probe 过滤用)。优先读缓存; 缓存缺失时现场构建一次。"""
    cache = load_probe_cache(args, cfg, require_params=False)
    if cache is not None:
        return set(cache["shared_vocab"])
    _, shared = _build_probe_artifacts(cfg, args)
    return shared


def build_probe_contexts(aligner, cfg: Config, contexts_per_ref: int):
    """用近义候选句(共享词表内)构建探针上下文集。"""
    ctx = []
    for ref in cfg.references:
        ids = aligner.tok(ref, return_tensors="pt")["input_ids"].to("cuda")
        cands = aligner.sample_near_synonyms(ids, M=contexts_per_ref, top_k=PROBE_TOP_K,
                                             k_neighbors=PROBE_K_NEIGHBORS)
        for c in cands:
            ctx.append(aligner.decode(c)[0])
    return ctx


def probe(cfg: Config, shared, contexts, samples, top_n):
    """对每个上下文采样 samples 次, 返回:
       - token_stats: {token: [obs_logp, ...]}  (只含观测到的共享 token)
       - ctx_means:   [{token: mean_logp}, ...] (每上下文对 samples 取均值)"""
    token_obs: dict[str, list[float]] = {}
    ctx_means = []
    for i, c in enumerate(contexts):
        runs = [dict(top_logprobs(cfg, c, top_n)) for _ in range(samples)]
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


def chi2_sf(x2, df):
    return float(chi2.sf(x2, df))


# ---------------------------------------------------------------------------
# 配置解析: CLI 优先, 环境变量兜底
# ---------------------------------------------------------------------------
def _env(name, default=None):
    return os.environ.get(name, default)


def resolve_config(args, fp=None):
    """从 args + 环境变量(必要时回退到已存档指纹 fp 里的模型身份)解析 Config。
    仅供 check: fp 提供记录的 target_tokenizer / aligner_model 兜底。"""
    endpoint = args.endpoint or _env("COV_ENDPOINT") or (fp or {}).get("endpoint")
    model = args.model or _env("COV_MODEL") or (fp or {}).get("model")
    api_key = args.api_key if args.api_key is not None else _env("COV_API_KEY", "") or ""
    target = args.target_tokenizer or (fp or {}).get("target_tokenizer") or model
    aligner = args.aligner_model or (fp or {}).get("aligner_model") or DEFAULT_ALIGNER_MODEL
    refs = list(args.references) if args.references else list(DEFAULT_REFERENCES)
    return Config(endpoint=endpoint, model=model, api_key=api_key,
                  target_tokenizer=target, aligner_model=aligner,
                  thinking=args.thinking, references=refs)


def require_endpoint_model(cfg: Config, ap):
    if not cfg.endpoint or not cfg.model:
        ap.error("需要 --endpoint 和 --model (或用环境变量 COV_ENDPOINT / COV_MODEL)")


# ---------------------------------------------------------------------------
# record / check / show
# ---------------------------------------------------------------------------
def do_record(args, ap):
    cfg = resolve_config(args)
    require_endpoint_model(cfg, ap)
    contexts, shared = load_or_build_probe(args, cfg)
    print(f"probe contexts: {len(contexts)}  samples/ctx: {args.samples}", flush=True)
    token_obs, ctx_means = probe(cfg, shared, contexts, args.samples, args.top_n)
    cov, V, X, means = covariance_from_ctx(ctx_means, args.floor)
    eig = np.linalg.eigvalsh((cov + cov.T) / 2)
    eig = eig[eig > 1e-9][::-1]

    stats = {}
    for t, obs in token_obs.items():
        m, s, n = mean_std(obs)
        stats[t] = {"mean": m, "std": s, "count": n}

    fp = {
        "schema": 1, "model": cfg.model, "endpoint": cfg.endpoint,
        "target_tokenizer": cfg.target_tokenizer, "aligner_model": cfg.aligner_model,
        "collected_at": int(time.time()),
        "contexts": contexts, "n_contexts": len(contexts),
        "samples": args.samples, "top_n": args.top_n, "floor": args.floor,
        "probe_cache": probe_cache_path(args, cfg),
        "token_stats": stats, "cov_eigvals": [float(e) for e in eig],
        "cov_trace": float(cov.trace()), "cov_frob": float(np.linalg.norm(cov)),
        "mean_logp": {t: float(v) for t, v in zip(V, means)},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(fp, f, ensure_ascii=False, indent=1)
    print(f"[record] 已存档指纹: {args.out}  (tokens={len(V)}, rank={len(eig)})")
    print(f"[record] cov_trace={cov.trace():.2f}  top3占比="
          f"{eig[:3].sum()/eig.sum()*100:.0f}%")


def do_check(args, ap):
    with open(args.out, encoding="utf-8") as f:
        old = json.load(f)
    cfg = resolve_config(args, fp=old)
    require_endpoint_model(cfg, ap)
    if cfg.target_tokenizer != old.get("target_tokenizer") and not args.target_tokenizer:
        print(f"[warn] 当前 target-tokenizer({cfg.target_tokenizer}) 与存档"
              f"({old.get('target_tokenizer')}) 不一致, 共享词表可能不匹配", flush=True)
    # 复用 record 时的探针缓存, 除非显式覆盖
    if not args.probe_cache and old.get("probe_cache"):
        args.probe_cache = old["probe_cache"]
    contexts = old["contexts"]
    shared = get_shared_vocab(args, cfg)
    samples = args.samples or old["samples"]
    print(f"[check] 重探 {len(contexts)} 上下文 x {samples} 采样", flush=True)
    token_obs, ctx_means = probe(cfg, shared, contexts, samples, old["top_n"])
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
                      ("model", "endpoint", "target_tokenizer", "collected_at",
                       "n_contexts", "samples", "top_n",
                       "cov_trace", "cov_frob", "cov_eigvals", "mean_logp")},
                     ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser(
        prog="covariance_prober",
        description="通用 model endpoint 的 next-token logprob 协方差指纹探测/检测。")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="{record,check,show}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", default=os.path.join(DATA_DIR, "cov.json"),
                        help="指纹存档文件(默认 data/cov.json)")
    common.add_argument("--probe-cache", default=None,
                        help="探针缓存文件(默认 data/probe_cache_<target-tokenizer>.json)")
    common.add_argument("--endpoint", default=None,
                        help="OpenAI 兼容接口 URL (或用环境变量 COV_ENDPOINT)")
    common.add_argument("--model", default=None,
                        help="请求时用的模型名 (或用环境变量 COV_MODEL)")
    common.add_argument("--api-key", default=None,
                        help="Bearer token (或用环境变量 COV_API_KEY; 无鉴权端点可省略)")
    common.add_argument("--target-tokenizer", default=None,
                        help="构建共享词表用的本地 HF 仓库名(默认取 --model)")
    common.add_argument("--aligner-model", default=None,
                        help="近义句搜索用的 aligner 模型(默认 Qwen/Qwen3.5-0.8B)")
    common.add_argument("--thinking", choices=["disabled", "enabled", "none"], default="none",
                        help="是否下发 DeepSeek 的 thinking 字段(默认 none 不发送)")
    common.add_argument("--reference", action="append", dest="references", default=None,
                        help="探针参考句(可多次); 默认用内置英文例句")

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
    {"record": do_record, "check": do_check, "show": do_show}[args.cmd](args, ap)


if __name__ == "__main__":
    main()
