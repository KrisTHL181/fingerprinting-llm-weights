# CLAUDE.md — 论文写作约束

本文件约束对 `main.tex` 及本目录其他稿子的所有改写，改完之后需要使用 `./compile.sh` 确认能编译通过。

## 硬性禁止

### 1. 破折号与连接号
- 正文中**尽可能不使用 em dash（`---` 或 `—`）和 en dash（`--` 或 `–`）**。
- 替换方式：句号、逗号、冒号、分号、括号，或重新断句。
- 例外（允许保留）：
  - 数学区间/范围，如 `$0.010$--$0.028$`、`$L8$--$L16$`、`$P \sim 10^9$` 中的 `--` 表示区间、`input--output` 这类复合词范围；
  - 代码注释里的分隔线（如 `% ----`）；
  - 公式、表格、引用内部。
- 修改后搜索 `---` 和 `--`，逐个确认每个出现在正文中的都是合理保留的。

### 2. 高发 AI 词
正文改写时避免堆叠以下词（单个使用可按语境判断，多个连用或滥用即违规）：

```
actually, additionally, align with, crucial/crucially, delve, emphasizing,
enduring, enhance, fostering, garner, highlight (作动词), interplay,
intricate/intricacies, key (作形容词, 如 "the key observation"), landscape
(抽象义), pivotal, showcase, tapestry (抽象义), testament, underscore,
valuable, vibrant, the key insight, turns out to be, it is important to note
```

用更直白、更具体的动词或直接陈述替换，例如：
- `The key observation is that A.` → 直接写 `A.`
- `Crucially, this bound...` → `This bound...`
- `which turns out to be more sensitive` → `which is more sensitive`

### 3. "不是-而是" 句法
避免 AI 偏好的对立式修辞结构。改写为直接陈述：

| 避免 | 改写为 |
|---|---|
| `not X but Y` / `is X, not Y` | 直接陈述 Y；X 的对比若确有必要，用非修辞方式（如 `in contrast to`、分号分句） |
| `not merely X but Y` / `does not just X, it Y` | 直接写 Y |
| `far from being X, it is Y` | 直接写 Y |
| `not a barrier to X but Y` | 直接写 Y 的正面表述 |
| `rather than X` | 用 `instead of`、`in contrast to`，或直接省略多余对比 |
| `not X but rather Y` | 直接写 Y |

**保留**：真正的技术性对比与数学必要条件，例如 `necessary but weak signal`、`sensitive to the weights but does not require recovering them`、`in shape but at reduced magnitude`。这些是学术表述，不是 AI 修辞，不要误删。

## 提醒（避免过度纠错）

- **保留学术语气**：数学论文使用正式词汇、被动语态、`we show / we propose / consequently` 是正常的，不算 AI 味。不要为了"去 AI 味"把论文改得像口语。
- **标题**：`\section`/`\subsection` 保持首字母大写（符合多数期刊格式），不要强行改 sentence case。
- 混淆来源/无出处、销售语气、emoji、聊天机器人腔等在本稿中不存在，不用处理。

## 完整参考

AI 写作模式的完整清单见 **humanizer** skill（`/humanizer`），以该 skill 中《Signs of AI writing》的模式为准。本文件是其针对本论文项目的浓缩约束。
