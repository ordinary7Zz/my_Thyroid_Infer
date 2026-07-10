#!/bin/bash
# ============================================================
# 上传项目到 ModelScope（魔搭社区）
# 仓库: instincts/my_Thyroid_infer
# 排除: results, results_*, datasets, wheels, gptoss, gptoss_pth 等
# 用法: bash upload_modelscope.sh [--dry-run] [--sync]
# ============================================================
set -euo pipefail

REPO_ID="instincts/my_Thyroid_infer"
COMMIT_MSG="Update project (code + weights + env)"
DRY_RUN=false
SYNC_FLAG=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --sync)    SYNC_FLAG="--sync"; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "  上传到 ModelScope: ${REPO_ID}"
echo "  排除目录: results, results_*, datasets, wheels, gptoss*, zips, *.zip"
echo "============================================================"

# 1. 确保 modelscope 已安装
if ! command -v ms &>/dev/null; then
    echo "[1/3] 安装 modelscope CLI ..."
    pip install -q modelscope
else
    echo "[1/3] modelscope CLI 已安装"
fi

# 2. 检查登录状态
echo "[2/3] 检查 ModelScope 登录状态 ..."
if ! ms whoami &>/dev/null 2>&1; then
    echo ""
    echo "❌ 未登录 ModelScope!"
    echo ""
    echo "请先登录（获取 token: https://www.modelscope.cn/my/myaccesstoken）:"
    echo ""
    echo "  ms login --token YOUR_TOKEN"
    echo ""
    echo "或设置环境变量:"
    echo ""
    echo "  export MODELSCOPE_API_TOKEN=YOUR_TOKEN"
    echo ""
    exit 1
fi
echo "  已登录: $(ms whoami 2>/dev/null)"

# 3. Dry-run: 智能分析上传内容
if $DRY_RUN; then
    echo ""
    echo "[Dry-run] 上传文件分析"
    echo ""

    # 收集所有待上传文件（size_bytes \t path），只扫描一次
    FILE_LIST=$(find . \
        -type d \( -name results -o -name 'results_*' -o -name datasets -o -name wheels \
                   -o -name 'gptoss*' -o -name __pycache__ -o -name .git -o -name .codebuddy \
                   -o -name zips \) -prune \
        -o -type f \( -name '*.pyc' -o -name '*.pyo' -o -name '*.so' \
                     -o -name '.DS_Store' -o -name '*.zip' \) -prune \
        -o -type f -printf '%s\t%p\n')

    # --- 目录大小汇总（按顶级目录分组，大小降序）---
    echo "━━━ 目录大小汇总（按大小降序）━━━"
    echo "$FILE_LIST" | awk -F'\t' '{
        size = $1; path = $2
        sub(/^\.\//, "", path)
        n = split(path, parts, "/")
        topdir = (n > 1) ? parts[1] "/" : path
        dir_size[topdir] += size
        dir_count[topdir]++
        total_size += size
        total_count++
    }
    END {
        for (d in dir_size)
            printf "%d\t%d\t%s\n", dir_size[d], dir_count[d], d
        printf "%d\t%d\t__TOTAL__\n", total_size, total_count
    }' | sort -t$'\t' -k1,1 -rn | while IFS=$'\t' read -r sz cnt dir; do
        if [ "$dir" = "__TOTAL__" ]; then
            printf "  %-8s  %5d files  ── TOTAL ──\n" "$(numfmt --to=iec "$sz")" "$cnt"
        else
            printf "  %-8s  %5d files  %s\n" "$(numfmt --to=iec "$sz")" "$cnt" "$dir"
        fi
    done

    # --- 大文件 Top 20（≥ 10MB，按大小降序）---
    echo ""
    echo "━━━ 大文件 Top 20（≥ 10MB，按大小降序）━━━"
    LARGE_COUNT=0
    LARGE_SIZE=0
    while IFS=$'\t' read -r sz path; do
        if [ "$sz" -ge 10485760 ]; then
            printf "  %-8s  %s\n" "$(numfmt --to=iec "$sz")" "$path"
            LARGE_COUNT=$((LARGE_COUNT + 1))
            LARGE_SIZE=$((LARGE_SIZE + sz))
        fi
    done < <(echo "$FILE_LIST" | sort -t$'\t' -k1,1 -rn | head -20)

    if [ "$LARGE_COUNT" -eq 0 ]; then
        echo "  （无 ≥ 10MB 的大文件）"
    fi

    # --- 全部大文件统计（含未展示的）---
    ALL_LARGE=$(echo "$FILE_LIST" | awk -F'\t' '$1 >= 10485760 {c++; s+=$1} END {printf "%d\t%d", c, s}')
    ALL_LARGE_COUNT=$(echo "$ALL_LARGE" | cut -f1)
    ALL_LARGE_SIZE=$(echo "$ALL_LARGE" | cut -f2)

    # --- 汇总 ---
    TOTAL_COUNT=$(echo "$FILE_LIST" | wc -l)
    TOTAL_SIZE=$(echo "$FILE_LIST" | awk -F'\t' '{s+=$1} END {print s}')

    echo ""
    echo "━━━ 汇总 ━━━"
    printf "  总文件数:     %d\n" "$TOTAL_COUNT"
    printf "  总大小:       %s\n" "$(numfmt --to=iec "$TOTAL_SIZE")"
    printf "  大文件(≥10M): %s 个, 合计 %s\n" "$ALL_LARGE_COUNT" "$(numfmt --to=iec "$ALL_LARGE_SIZE")"
    echo ""
    echo "（dry-run 模式，未实际上传。去掉 --dry-run 执行上传）"
    exit 0
fi

# 3. 上传
echo "[3/3] 开始上传 ..."
echo ""

ms upload "${REPO_ID}" . \
    --exclude "results/**" \
    --exclude "results_pretrained/**" \
    --exclude "results_dino_mask_pretrained/**" \
    --exclude "datasets/**" \
    --exclude "wheels/**" \
    --exclude "gptoss/**" \
    --exclude "gptoss_pth/**" \
    --exclude "zips/**" \
    --exclude "__pycache__/**" \
    --exclude ".git/**" \
    --exclude ".codebuddy/**" \
    --exclude "*.pyc" \
    --exclude ".DS_Store" \
    --exclude "*.pyo" \
    --exclude "*.so" \
    --exclude "*.zip" \
    --commit-message "${COMMIT_MSG}" \
    --max-workers 4 \
    ${SYNC_FLAG}

echo ""
echo "✅ 上传完成!"
echo "查看: https://www.modelscope.cn/models/${REPO_ID}"
