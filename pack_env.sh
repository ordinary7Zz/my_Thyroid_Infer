#!/bin/bash
# ============================================================
# 重新打包 conda 环境 → thyroid_infer_env.tar.gz
# 用法: bash pack_env.sh
# ============================================================
set -euo pipefail

ENV_NAME="thyroid_infer"
OUTPUT="thyroid_infer_env.tar.gz"

echo "============================================================"
echo "  重新打包 conda 环境: ${ENV_NAME}"
echo "  输出文件: ${OUTPUT}"
echo "============================================================"

# 1. 确保 conda-pack 已安装
if ! command -v conda-pack &>/dev/null; then
    echo "[1/3] 安装 conda-pack ..."
    pip install -q conda-pack
else
    echo "[1/3] conda-pack 已安装"
fi

# 2. 打包环境
echo "[2/3] 打包环境（可能需要几分钟）..."
conda pack -n "${ENV_NAME}" -o "${OUTPUT}" --force

# 3. 显示结果
echo "[3/3] 打包完成"
echo ""
echo "文件信息:"
ls -lh "${OUTPUT}"
echo ""
echo "校验:"
sha256sum "${OUTPUT}" | cut -d' ' -f1

echo ""
echo "✅ 环境打包完成: ${OUTPUT}"
