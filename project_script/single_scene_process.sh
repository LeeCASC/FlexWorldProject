#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   ./single_scene_process.sh <scene_id>
# 作用：为指定场景准备 Gaussian Splatting 输入目录（images + sparse 软链）。
#      仅执行“第一步”准备，便于后续手动/批量调用 GS 训练。

SCENE_ID="${1:-001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f}"

# 路径配置（如有不同请修改）
SRC_IMG_ROOT="/home/lizihao/datasets/DL3DV10K/1K"
SRC_CACHE_ROOT="/home/lizihao/datasets/DL3DV10K-ColmapCache/1K"
WORK_ROOT="/home/lizihao/datasets/gs_input"
GS_OUTPUT_ROOT="/home/lizihao/datasets/DL3DV-gaussian-splatting/output"

IMG_SRC="${SRC_IMG_ROOT}/${SCENE_ID}/images_4"
SPARSE_SRC="${SRC_CACHE_ROOT}/${SCENE_ID}/colmap/sparse/0"
WORK_DIR="${WORK_ROOT}/${SCENE_ID}"

if [[ ! -d "${IMG_SRC}" ]]; then
  echo "❌ 找不到图像目录: ${IMG_SRC}"
  exit 1
fi
if [[ ! -d "${SPARSE_SRC}" ]]; then
  echo "❌ 找不到 sparse 目录: ${SPARSE_SRC}"
  exit 1
fi

mkdir -p "${WORK_DIR}"
ln -sfn "${IMG_SRC}" "${WORK_DIR}/images"
ln -sfn "${SPARSE_SRC}" "${WORK_DIR}/sparse"

echo "✅ 已准备完成：${WORK_DIR}"
echo "   images -> ${IMG_SRC}"
echo "   sparse -> ${SPARSE_SRC}"
echo
echo "后续可在 Gaussian Splatting 仓库执行训练生成点云，例如："
echo "  cd ~/projects/gaussian-splatting"
echo "  conda activate FlexWorld   # 或你的 GS 环境"
echo "  python train.py \\"
echo "    -s ${WORK_DIR} \\"
echo "    -m ${GS_OUTPUT_ROOT}/${SCENE_ID} \\"
echo "    --iterations 7000 --eval False --save-iteration 7000"