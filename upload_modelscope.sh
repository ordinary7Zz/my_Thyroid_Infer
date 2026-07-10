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

# 3. Dry-run: 列出将上传的文件
if $DRY_RUN; then
    echo ""
    echo "[Dry-run] 将上传的文件（排除 results/datasets/wheels/gptoss 后）:"
    echo ""
    # 列出文件，排除不需要的目录
    find . \
        -type d \( -name results -o -name 'results_*' -o -name datasets -o -name wheels \
                   -o -name 'gptoss*' -o -name __pycache__ -o -name .git -o -name .codebuddy \
                   -o -name zips \) -prune \
        -o -type f \( -name '*.pyc' -o -name '*.pyo' -o -name '*.so' \
                     -o -name '.DS_Store' -o -name '*.zip' \) -prune \
        -o -type f -print | \
        sort | \
        while read -r f; do
            size=$(du -h "$f" | cut -f1)
            printf "  %8s  %s\n" "$size" "$f"
        done

    total=$(find . \
        -type d \( -name results -o -name 'results_*' -o -name datasets -o -name wheels \
                   -o -name 'gptoss*' -o -name __pycache__ -o -name .git -o -name .codebuddy \
                   -o -name zips \) -prune \
        -o -type f \( -name '*.pyc' -o -name '*.pyo' -o -name '*.so' \
                     -o -name '.DS_Store' -o -name '*.zip' \) -prune \
        -o -type f -print | wc -l)
    echo ""
    echo "共 ${total} 个文件"
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
