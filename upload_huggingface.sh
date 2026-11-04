#!/bin/bash
# ============================================================
# 上传项目到 Hugging Face（含代码 + 权重）
# 仓库: instincts7Zz/my_Thyroid_infer  (repo-type: model)
# 排除: results, datasets, wheels, gptoss 等（与 upload_modelscope.sh 对齐）
# 用法: bash upload_huggingface.sh [--dry-run]
# ============================================================
set -euo pipefail

REPO_ID="instincts7Zz/my_Thyroid_infer"
COMMIT_MSG="Update project (code + weights + env)"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "  上传到 Hugging Face: ${REPO_ID}"
echo "============================================================"

if ! command -v hf &>/dev/null; then
    pip install -q -U "huggingface_hub[cli]"
fi

if ! hf auth whoami &>/dev/null 2>&1; then
    echo "❌ 未登录 Hugging Face。请运行: hf auth login"
    exit 1
fi
echo "  已登录 HF: $(hf auth whoami 2>/dev/null | head -1)"

EXCLUDES=(
    --exclude "results/**"
    --exclude "results_pretrained/**"
    --exclude "results_dino_mask_pretrained/**"
    --exclude "results_*/**"
    --exclude "datasets/**"
    --exclude "wheels/**"
    --exclude "gptoss/**"
    --exclude "gptoss_pth/**"
    --exclude "zips/**"
    --exclude "__pycache__/**"
    --exclude ".git/**"
    --exclude ".codebuddy/**"
    --exclude "*.pyc"
    --exclude ".DS_Store"
    --exclude "*.pyo"
    --exclude "*.so"
    --exclude "*.tar.gz"
    --exclude "thyroid_infer_env.tar.gz"
)

if $DRY_RUN; then
    echo "（dry-run）将执行: hf upload ${REPO_ID} . --repo-type model ${EXCLUDES[*]} ..."
    exit 0
fi

hf upload "${REPO_ID}" . \
    --repo-type model \
    "${EXCLUDES[@]}" \
    --commit-message "${COMMIT_MSG}"

echo ""
echo "✅ 上传完成!"
echo "查看: https://huggingface.co/${REPO_ID}"
