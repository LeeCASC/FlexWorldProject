#!/usr/bin/env bash
set -euo pipefail

# 批量：COLMAP 去畸变 → 缩放相机到 images_4 分辨率 → 生成 undistorted_scaled → 跑 Gaussian Splatting
# 默认使用分辨率 960x540（images_4，原始 3840x2160 的 1/4），并在 gsplat310 环境下训练。
#
# 用法：
#   ./batch_gs_reconstruct.sh [scene_id1 scene_id2 ...]
# 不传参数则按 RAW_ROOT 下的目录顺序处理全部场景。

SCALE=0.25  # 3840x2160 -> 960x540
RAW_ROOT="/home/lizihao/datasets/DL3DV10K/1K"
CACHE_ROOT="/home/lizihao/datasets/DL3DV10K-ColmapCache/1K"
GS_INPUT_ROOT="/home/lizihao/datasets/gs_input"
GS_OUT_ROOT="/home/lizihao/datasets/DL3DV-gaussian-splatting/output"
GS_REPO="/home/lizihao/projects/gaussian-splatting"
GS_ENV="gsplat310"
# 避免 conda activate 时 MKL 接口变量未定义导致 set -u 退出
export MKL_INTERFACE_LAYER=${MKL_INTERFACE_LAYER:-GNU}

ensure_conda() {
  # shellcheck disable=SC1091
  source ~/anaconda3/etc/profile.d/conda.sh
}

scale_cameras_txt() {
  local src_txt="$1" dst_txt="$2" scale="$3"
  python - "$src_txt" "$dst_txt" "$scale" <<'PY'
import os, sys
src_txt, dst_txt, scale = sys.argv[1], sys.argv[2], float(sys.argv[3])
os.makedirs(os.path.dirname(dst_txt), exist_ok=True)
with open(src_txt) as f, open(dst_txt, "w") as g:
    for line in f:
        if line.startswith("#") or not line.strip():
            g.write(line); continue
        parts = line.strip().split()
        cam_id, model = parts[0], parts[1]
        w, h = float(parts[2]), float(parts[3])
        params = list(map(float, parts[4:]))
        if model in ["PINHOLE"]:
            params[0] *= scale; params[1] *= scale; params[2] *= scale; params[3] *= scale
        elif model in ["SIMPLE_PINHOLE"]:
            params[0] *= scale; params[1] *= scale; params[2] *= scale
        elif model in ["OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE"]:
            params[0] *= scale; params[1] *= scale; params[2] *= scale; params[3] *= scale
        else:
            raise ValueError(f"Unsupported model {model}")
        w_new, h_new = int(w * scale), int(h * scale)
        g.write(f"{cam_id} {model} {w_new} {h_new} " + " ".join(f"{p:.8f}" for p in params) + "\n")
PY
}

process_scene() {
  local scene="$1"
  local raw_img="${RAW_ROOT}/${scene}/images_4"
  local raw_sparse="${CACHE_ROOT}/${scene}/colmap/sparse/0"
  local sparse_txt="${CACHE_ROOT}/${scene}/colmap/sparse_txt"
  local scaled_txt="${CACHE_ROOT}/${scene}/colmap/sparse_scaled_txt"
  local undist_sparse="${CACHE_ROOT}/${scene}/colmap/undistorted_sparse"
  local undist_out="${CACHE_ROOT}/${scene}/colmap/undistorted_scaled"
  local work="${GS_INPUT_ROOT}/${scene}"
  local gs_out="${GS_OUT_ROOT}/${scene}"

  echo "=== Scene: ${scene} ==="

  if [[ ! -d "${raw_sparse}" ]]; then
    echo "  [skip] sparse not found: ${raw_sparse}"
    return
  fi

  if [[ ! -d "${raw_img}" ]]; then
    echo "  [skip] images_4 not found: ${raw_img}"
    return
  fi

  # 检查 images_4 目录中是否有图像文件
  if [[ -z "$(find "${raw_img}" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) 2>/dev/null | head -1)" ]]; then
    echo "  [skip] images_4 is empty: ${raw_img}"
    return
  fi

  # 0) 如果已经有点云则跳过
  if [[ -f "${gs_out}/point_cloud/iteration_7000/point_cloud.ply" ]]; then
    echo "  [skip] point_cloud already exists."
    return
  fi

  mkdir -p "${sparse_txt}" "${scaled_txt}" "${undist_sparse}" "${undist_out}" "${work}"

  # 1) bin -> txt
  if [[ ! -f "${sparse_txt}/cameras.txt" ]]; then
    colmap model_converter --input_path "${raw_sparse}" --output_path "${sparse_txt}" --output_type TXT
  fi

  # 2) 缩放相机 txt
  scale_cameras_txt "${sparse_txt}/cameras.txt" "${scaled_txt}/cameras.txt" "${SCALE}"
  cp "${sparse_txt}/images.txt" "${scaled_txt}/images.txt"
  cp "${sparse_txt}/points3D.txt" "${scaled_txt}/points3D.txt"

  # 3) txt -> bin (scaled)
  colmap model_converter --input_path "${scaled_txt}" --output_path "${undist_sparse}" --output_type BIN

  # 4) 去畸变
  colmap image_undistorter \
    --image_path "${raw_img}" \
    --input_path "${undist_sparse}" \
    --output_path "${undist_out}" \
    --output_type COLMAP

  # 确保 sparse/0 结构
  mkdir -p "${undist_out}/sparse/0"
  cp "${undist_out}/sparse/"*.bin "${undist_out}/sparse/0/"

  # 5) 更新 gs_input 软链
  rm -f "${work}/images" "${work}/sparse"
  ln -s "${undist_out}/images" "${work}/images"
  ln -s "${undist_out}/sparse" "${work}/sparse"

  # 6) 跑 GS
  ensure_conda
  conda activate "${GS_ENV}"
  cd "${GS_REPO}"
  python train.py \
    -s "${work}" \
    -m "${gs_out}" \
    --iterations 7000 \
    --save_iterations 7000 \
    --disable_viewer
}

if [[ $# -gt 0 ]]; then
  scenes=("$@")
else
  mapfile -t scenes < <(ls "${RAW_ROOT}")
fi

for s in "${scenes[@]}"; do
  process_scene "$s"
done

