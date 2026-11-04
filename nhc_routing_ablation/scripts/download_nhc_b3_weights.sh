#!/usr/bin/env bash
# 从 HuggingFace 或 ModelScope 只拉 NHC 七策略 B3 所需的 DINO 权重（不下载 lsnet/repvit 等）。
# 用法（在 my_Thyroid_Infer 根目录）:
#   bash nhc_routing_ablation/scripts/download_nhc_b3_weights.sh
#   WEIGHT_SOURCE=huggingface bash nhc_routing_ablation/scripts/download_nhc_b3_weights.sh
#   WITH_AUTOGLUON=0 bash ...   # 若只验 seg+cls 且跳过 static_rule 可关（不推荐）
#   FORCE_DOWNLOAD=1 bash ...   # 本地已有也强制重新拉取
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SOURCE="${WEIGHT_SOURCE:-modelscope}"
WITH_AUTOGLUON="${WITH_AUTOGLUON:-1}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"

HF_REPO="instincts7Zz/my_Thyroid_infer"
MS_REPO="instincts/my_Thyroid_infer"
AG_DIR="infer_autogluon/checkpoints/binary/autogluon_model_20260204_140346"
RADIOMICS_YAML="infer_autogluon/radiomics_2d.yaml"

mapfile -t FILES < <(cat <<'LIST'
infer_dinov3_unet/checkpoints/Nodule/dino_unet_train_all_datasets_epoch_40.pth
infer_dinov3_unet/checkpoints/Nodule/dino_unet_train_TN3K_epoch_50.pth
infer_dinov3_unet/checkpoints/Nodule/dino_unet_train_ThyroidXL_epoch_50.pth
infer_dinov3_unet/checkpoints/Nodule/dino_unet_train_PKTN_comprehensive_epoch_50.pth
infer_dinov3_unet/checkpoints/Nodule/dino_unet_train_TN5K_epoch_50.pth
infer_dinov3_unet_multitask/checkpoints/BM_2cls/dino_unet_train_multitask_all_datasets_epoch_40.pth
infer_dinov3_unet_multitask/checkpoints/BM_2cls/dino_unet_train_multitask_TN3K_epoch_50.pth
infer_dinov3_unet_multitask/checkpoints/BM_2cls/dino_unet_train_multitask_TN5K_epoch_50.pth
infer_dinov3_unet_multitask/checkpoints/BM_2cls/dino_unet_train_multitask_ThyroidXL_epoch_50.pth
infer_dinov3_unet/checkpoints/TGVideo/dino_unet_train_TGVideo_epoch_30.pth
LIST
)

mkdir -p infer_dinov3_unet/checkpoints/Nodule infer_dinov3_unet/checkpoints/TGVideo infer_dinov3_unet_multitask/checkpoints/BM_2cls

_nhc_weights_missing() {
  local f
  for f in "${FILES[@]}"; do
    [[ -f "$ROOT/$f" ]] || return 0
  done
  if [[ "$WITH_AUTOGLUON" == "1" ]]; then
    [[ -f "$ROOT/$RADIOMICS_YAML" ]] || return 0
    [[ -d "$ROOT/$AG_DIR" ]] || return 0
    # AutoGluon 目录非空且含 models（避免空目录误判为已齐全）
    [[ -d "$ROOT/$AG_DIR/models" ]] || return 0
  fi
  return 1
}

echo "仓库根: $ROOT"

if [[ "$FORCE_DOWNLOAD" != "1" ]] && ! _nhc_weights_missing; then
  echo "本地 NHC 权重已齐全，跳过下载（如需重拉: FORCE_DOWNLOAD=1 bash $0）"
  for f in "${FILES[@]}"; do
    echo "  OK $f"
  done
  if [[ "$WITH_AUTOGLUON" == "1" ]]; then
    echo "  OK $RADIOMICS_YAML"
    echo "  OK $AG_DIR/"
  fi
  echo "权重校验通过（未访问网络）。"
  exit 0
fi

echo "权重来源: $SOURCE（结节/分类 DINO + TGVideo 腺体 + AutoGluon）"

if [[ "$SOURCE" == "huggingface" ]]; then
  pip install -q -U "huggingface_hub[cli]"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  hf download "$HF_REPO" "${FILES[@]}" --repo-type model --local-dir "$ROOT"
  if [[ "$WITH_AUTOGLUON" == "1" ]]; then
    hf download "$HF_REPO" "$RADIOMICS_YAML" --repo-type model --local-dir "$ROOT"
    hf download "$HF_REPO" "$AG_DIR" --repo-type model --local-dir "$ROOT"
  fi
elif [[ "$SOURCE" == "modelscope" ]]; then
  pip install -q modelscope
  export INFER_ROOT="$ROOT"
  export MS_REPO="$MS_REPO"
  export WITH_AUTOGLUON="$WITH_AUTOGLUON"
  export FILE_LIST
  FILE_LIST="$(printf '%s\n' "${FILES[@]}")"
  python3 <<'PY'
import os
from pathlib import Path

from modelscope import snapshot_download

root = Path(os.environ["INFER_ROOT"]).resolve()
patterns = [ln.strip() for ln in os.environ["FILE_LIST"].splitlines() if ln.strip()]
patterns.append("infer_autogluon/radiomics_2d.yaml")
if os.environ.get("WITH_AUTOGLUON") == "1":
    patterns.append("infer_autogluon/checkpoints/binary/autogluon_model_20260204_140346/**")
snapshot_download(os.environ["MS_REPO"], local_dir=str(root), allow_file_pattern=patterns)
print("已写入:", root)
PY
else
  echo "WEIGHT_SOURCE 须为 modelscope 或 huggingface" >&2
  exit 1
fi

missing=0
for f in "${FILES[@]}"; do
  if [[ -f "$ROOT/$f" ]]; then
    echo "  OK $f"
  else
    echo "  缺失 $f" >&2
    missing=1
  fi
done
if [[ "$WITH_AUTOGLUON" == "1" ]]; then
  ag="$ROOT/$AG_DIR"
  [[ -d "$ag" ]] && echo "  OK $AG_DIR/" || {
    echo "  缺失 AutoGluon 目录（static_rule 需要）" >&2
    missing=1
  }
fi
[[ "$missing" == 0 ]] || exit 1
echo "权重下载校验通过。"
