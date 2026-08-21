# representation-matching

中间层表征对齐（近义句生成 + 权重指纹）的实验仓库。每个实验可独立复现：脚本在 `scripts/`，
结果数据在 `data/`，论文图在 `figures/`，论文在 `paper/`。共享核心库 `align.py` 与脚本同目录，
必须随脚本一起拷贝。

```
representation-matching/
├── scripts/    # 所有实验/画图脚本 + 共享库 align.py + 远程编排 run_all_calib.sh
├── data/       # 各实验的结果数据 (.txt/.json)
├── figures/    # 论文图 PDF (paper/main.tex 通过 ../figures/ 引用)
├── paper/      # 论文 main.tex + refs.bib + compile.sh + CLAUDE.md
└── _archive/   # 过时/备份/重复内容，仅供追溯，不参与复现
```

## 前置依赖

```bash
pip install torch transformers peft datasets scipy matplotlib
# 国内拉取 HF 模型建议: export HF_ENDPOINT=https://hf-mirror.com
```

重 GPU 实验（E1–E3、E5–E9）需本地或远程 NVIDIA GPU；DeepSeek 实验（E4）需 API key。

## 实验清单

### E1 判别力：中间层表征能否区分「同义 / 随机」
- 脚本：`scripts/discriminative_check.py`（本地 GPU，Qwen-0.8B）
- 运行：`cd scripts && python discriminative_check.py`
- 产物：`data/disc.txt` → 论文图 `fig_discriminative.pdf`

### E2 权重噪声标定（主统计量 D_KL）
- 脚本：`scripts/sweep_stats.py`（本地 GPU）
- 运行（每个模型族/层一次）：
  ```bash
  cd scripts
  python sweep_stats.py --model Qwen/Qwen3.5-0.8B     --layer 8  --out ../data/sweep_qwen_l8_v2.txt
  python sweep_stats.py --model Qwen/Qwen3.5-0.8B     --layer 16 --out ../data/sweep_qwen_l16_v2.txt
  python sweep_stats.py --model unsloth/Llama-3.2-1B  --layer 8  --out ../data/sweep_llama_l8_v2.txt
  python sweep_stats.py --model unsloth/gemma-3-1b-pt --layer 8  --out ../data/sweep_gemma_l8_v2.txt
  ```
- 产物：`data/sweep_*_v2.txt` → 论文图 `fig_calibration.pdf`

### E3 黑盒 logp-协方差标定
- 脚本：`scripts/calibrate_logp_cov.py`；编排：`scripts/run_all_calib.sh`（远程 GPU）
- 运行：`cd scripts && ./run_all_calib.sh`
  - 远程机需把本仓库同步过去；路径用环境变量覆盖（`WORK_DIR`、`PY`、`HF_ENDPOINT`、`OUT_DIR`），默认值对应 autodl。
- 产物：`data/scan_*_logp.txt`

### E4 DeepSeek 协方差指纹（record / check / show）
- 脚本：`scripts/deepseek_covariance.py`（需 DeepSeek API key，填入文件内 `KEY` 处）
- 运行：
  ```bash
  cd scripts
  python deepseek_covariance.py record --out ../data/ds_cov.json --samples 3   # 探测并存档指纹
  python deepseek_covariance.py check  --out ../data/ds_cov.json               # 两样本检验
  python deepseek_covariance.py show   --out ../data/ds_cov.json               # 打印存档摘要
  ```
- 产物：`data/ds_cov.json` → 论文图 `fig_spectrum.pdf`

### E5 LoRA 微调 / 跨族偷换灵敏度
- 脚本：`scripts/remote_lora_test.py`（远程 GPU）+ `scripts/analyze_lora_results.py`（本地）
- 运行：
  ```bash
  cd scripts
  ALPACA_DIR=/root/autodl-tmp/work/alpaca_sub_2500 python remote_lora_test.py --out ../data/lora_test_results.json
  python analyze_lora_results.py data/lora_test_results.json
  ```
- 产物：`data/lora_test_results.json` → 论文图 `fig_lora_sensitivity.pdf`

### E6 量化灵敏度
- 脚本：`scripts/remote_quant_test.py`（远程 GPU）+ `scripts/analyze_quant_results.py`（本地）
- 运行：
  ```bash
  cd scripts
  python remote_quant_test.py --out ../data/quant_test_results.json
  python analyze_quant_results.py data/quant_test_results.json
  ```
- 产物：`data/quant_test_results.json` → 论文图 `fig_quant_sensitivity.pdf`

### E7 RL(DPO) 对齐微调灵敏度
- 脚本：`scripts/remote_rl_test.py`（远程 GPU）
- 运行：`cd scripts && python remote_rl_test.py --out ../data/rl_test_results.json`
- 产物：`data/rl_test_results.json` → 论文图 `fig_rl_sensitivity.pdf`

### E8 ablation：表征等价是否必要
- 脚本：`scripts/ablate_repeq.py`（本地 GPU）
- 运行：`cd scripts && python ablate_repeq.py`
- 产物：`data/ablate_repeq.txt`

### E9 ablation：表征等价的集中度 / 可迁移性
- 脚本：`scripts/ablate_concentration.py`（本地 GPU）
- 运行：`cd scripts && python ablate_concentration.py`
- 产物：`data/ablate_concentration.txt`（+ `data/ablate_concentration.txt.probes.json`）

## 图与论文

- 图（从 `data/` 重新生成 `figures/`，需 matplotlib）：
  ```bash
  cd scripts
  python make_figures.py     # fig_calibration / fig_discriminative / fig_spectrum
  python make_lora_fig.py    # fig_lora_sensitivity
  python make_quant_fig.py   # fig_quant_sensitivity
  python make_rl_fig.py      # fig_rl_sensitivity
  ```
- 论文（`main.tex` 经 `../figures/` 引用图）：`cd paper && ./compile.sh`

## 复现流程建议

E2 是主标定（D_KL vs 权重噪声尺度 r，对应论文 Table 3 / fig_calibration），
E5–E7 在远程机实测指纹方法对不同篡改（LoRA / 量化 / RL 对齐）的灵敏度并映射回等价噪声尺度 r_eq，
E8–E9 检验「表征等价」性质本身是否被需要、是否集中/可迁移。全部脚本输出统一落到 `data/`，
画图脚本从 `data/` 读取，论文图来自 `figures/`，从而任何一步都可单独重跑而不影响其他部分。
