"""
中间层表征对齐的同义句生成 — 核心库

思路: 给定参考句 S,取小模型中间层 L 的 hidden state h*。
在【受限子词表】内离散优化输入 tokens 构成 S',使 |h(S',L) - h*| 足够小(≈含义相同),
同时强制 S' 在表面与 S 不同(多样性正则)→ 得到"潜在同义句"。

方法: HotFlip 风格梯度引导束搜索。
  - 求 loss 关于各位置 input embedding 的梯度,一阶近似选出每位置 top-k 候选替换 token(仅限共享词表)
  - 批量精确打分所有候选,结合多样性正则挑选
  - 贪心 / 束搜索迭代
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# 池化方式: 把一层的 hidden states [B,T,D] 归约为句子向量 [B,D]
# ---------------------------------------------------------------------------
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
