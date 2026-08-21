#!/usr/bin/env bash
#
# compile.sh — 编译 LaTeX 论文（pdflatex + bibtex，自动处理交叉引用）
#
# 用法:
#   ./compile.sh            # 正常编译
#   ./compile.sh --clean    # 清理编译产生的临时文件后退出
#   ./compile.sh --run      # 编译成功后自动打开生成的 PDF
#
set -euo pipefail

# --- 配置 -----------------------------------------------------------------
TEX="main.tex"
MAIN="${TEX%.tex}"          # main
AUXEXTS="aux bbl blg log out toc lof lot"   # 需要清理的临时文件后缀
RUN_PDF=false

# --- 参数解析 -------------------------------------------------------------
for arg in "$@"; do
    case "$arg" in
        --clean)
            # 只删除编译生成的临时文件，绝不匹配源文件 main.tex
            for ext in $AUXEXTS; do
                [ -e "$MAIN.$ext" ] && rm -f "$MAIN.$ext"
            done
            echo "已清理临时文件 (保留 $TEX / $MAIN.pdf)"
            exit 0
            ;;
        --run)   RUN_PDF=true ;;
        *) echo "未知参数: $arg (仅支持 --clean / --run)" >&2; exit 1 ;;
    esac
done

# --- 编译 -----------------------------------------------------------------
# 1) 编译收集交叉引用; 2) bibtex 生成参考文献; 3-4) 重复编译解决引用
pdflatex -interaction=nonstopmode "$TEX" > /dev/null
if [ -f "$MAIN.bbl" ] || grep -q '\\bibliography' "$TEX" 2>/dev/null; then
    bibtex "$MAIN" > /dev/null
fi
pdflatex -interaction=nonstopmode "$TEX" > /dev/null
pdflatex -interaction=nonstopmode "$TEX" > /dev/null

# --- 检查错误 -------------------------------------------------------------
if grep -qE '^! ' "$MAIN.log"; then
    echo "编译失败，存在错误。请查看 $MAIN.log" >&2
    grep -nE '^! ' "$MAIN.log" | head >&2
    exit 1
fi

# --- 结果 -----------------------------------------------------------------
if grep -qi 'undefined citations' "$MAIN.log"; then
    echo "警告: 仍有未定义的引用，请检查 refs.bib 中的 citation key" >&2
fi

echo "编译完成: $MAIN.pdf"

if [ "$RUN_PDF" = true ]; then
    if command -v xdg-open > /dev/null; then
        xdg-open "$MAIN.pdf"
    elif command -v open > /dev/null; then
        open "$MAIN.pdf"
    else
        echo "未找到可用的 PDF 打开命令" >&2
    fi
fi
