#!/usr/bin/env python3
"""
DL3DV10K 批量执行 video_generate.py 脚本

功能：
- 读取 dl3dv10k 目录下前 N 个子文件夹里的指定帧（默认 frame_00001.png）
- 按图片顺序分组分配轨迹：每 group_size 张图片使用一种轨迹，按 traj_schedule 顺序依次循环
- 支持设置 CUDA_VISIBLE_DEVICES
- 每次生成打印耗时

典型用法（你的需求）：
python batch_video_generate_dl3dv.py \
  --dl3dv_dir ./dl3dv10k \
  --max_folders 80 \
  --frame_name frame_00001.png \
  --group_size 10 \
  --traj_schedule up,down,left,right,forward,backward,rotate_left,rotate_right \
  --output_dir ./results-dl3dv10k
"""

import os
import subprocess
import argparse
import time
from pathlib import Path


VALID_TRAJS = ["up", "down", "left", "right", "forward", "backward", "rotate_left", "rotate_right"]


def run_command(input_image_path: Path, output_dir: Path, traj: str, cuda_visible_devices: str | None = None) -> bool:
    cmd = [
        "python",
        "video_generate.py",
        "--input_image_path",
        str(input_image_path),
        "--output_dir",
        str(output_dir),
        "--traj",
        traj,
    ]

    env = os.environ.copy()
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    print(f"\n{'=' * 60}")
    print("正在处理:")
    print(f"  输入图片: {input_image_path}")
    print(f"  输出目录: {output_dir}")
    print(f"  轨迹类型: {traj}")
    if cuda_visible_devices is not None:
        print(f"  CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
    print(f"{'=' * 60}\n")

    try:
        t0 = time.perf_counter()
        subprocess.run(cmd, check=True, cwd=os.getcwd(), env=env)
        dt = time.perf_counter() - t0
        print(f"✓ 成功完成: {input_image_path} - {traj} | 耗时: {dt:.2f}s\n")
        return True
    except subprocess.CalledProcessError as e:
        dt = time.perf_counter() - t0 if "t0" in locals() else None
        print(f"✗ 处理失败: {input_image_path} - {traj}")
        if dt is not None:
            print(f"  耗时: {dt:.2f}s")
        print(f"  错误信息: {e}\n")
        return False


def collect_dl3dv_frames(dl3dv_dir: Path, max_folders: int, frame_name: str) -> list[tuple[str, Path]]:
    """
    返回: [(folder_name, frame_path), ...]
    - folder_name: 子文件夹名（用于输出目录命名避免覆盖）
    - frame_path: 对应 frame 文件路径
    """
    if not dl3dv_dir.exists() or not dl3dv_dir.is_dir():
        raise FileNotFoundError(f"dl3dv_dir 不存在或不是目录: {dl3dv_dir}")

    subdirs = [p for p in dl3dv_dir.iterdir() if p.is_dir()]
    subdirs.sort(key=lambda p: p.name)
    subdirs = subdirs[:max_folders]

    results: list[tuple[str, Path]] = []
    missing: list[str] = []
    for d in subdirs:
        fp = d / frame_name
        if fp.is_file():
            results.append((d.name, fp))
        else:
            missing.append(str(fp))

    if missing:
        print(f"警告: 有 {len(missing)} 个文件夹缺少 {frame_name}，将跳过这些样本。")
        # 只打印前几个，避免刷屏
        for m in missing[:10]:
            print(f"  missing: {m}")
        if len(missing) > 10:
            print(f"  ... 还有 {len(missing) - 10} 个未显示")

    return results


def main():
    parser = argparse.ArgumentParser(description="DL3DV10K 批量执行 video_generate.py 脚本")

    parser.add_argument(
        "--dl3dv_dir",
        type=str,
        default="./dl3dv10k",
        help="DL3DV10K 本地目录（里面是很多子文件夹，每个子文件夹含 frame_*.png）",
    )

    parser.add_argument(
        "--max_folders",
        type=int,
        default=80,
        help="按子文件夹名排序后取前 N 个文件夹（默认 80）",
    )

    parser.add_argument(
        "--frame_name",
        type=str,
        default="frame_00001.png",
        help="每个子文件夹中要读取的帧文件名（默认 frame_00001.png）",
    )

    parser.add_argument(
        "--group_size",
        type=int,
        default=10,
        help="每组包含多少张图片（默认 10）：每组使用一种轨迹",
    )

    parser.add_argument(
        "--traj_schedule",
        type=str,
        default="up,down,left,right,forward,backward,rotate_left,rotate_right",
        help="轨迹列表（逗号分隔），按顺序每 group_size 张图片切换一次轨迹",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results-dl3dv10k",
        help="输出目录（默认 ./results-dl3dv10k）",
    )

    parser.add_argument(
        "--cuda_visible_devices",
        type=str,
        default=None,
        help="设置 CUDA_VISIBLE_DEVICES（例如 0 或 0,1）。不设置则沿用当前环境变量/默认行为。",
    )

    args = parser.parse_args()

    if args.max_folders <= 0:
        raise ValueError("--max_folders 必须为正整数")
    if args.group_size <= 0:
        raise ValueError("--group_size 必须为正整数")

    trajs = [t.strip() for t in args.traj_schedule.split(",") if t.strip()]
    if not trajs:
        raise ValueError("--traj_schedule 不能为空")
    for t in trajs:
        if t not in VALID_TRAJS:
            print(f"警告: 轨迹类型 '{t}' 不在支持列表中: {VALID_TRAJS}")

    dl3dv_dir = Path(args.dl3dv_dir)
    pairs = collect_dl3dv_frames(dl3dv_dir, args.max_folders, args.frame_name)
    if not pairs:
        print("错误: 没有找到任何可用的输入帧文件")
        return

    print("\n批量处理配置:")
    print(f"  DL3DV 目录: {dl3dv_dir}")
    print(f"  读取前 N 个文件夹: {args.max_folders}")
    print(f"  实际可用帧数量: {len(pairs)}")
    print(f"  frame_name: {args.frame_name}")
    print(f"  group_size: {args.group_size}")
    print(f"  traj_schedule: {trajs}")
    print(f"  output_dir: {args.output_dir}")
    if args.cuda_visible_devices is not None:
        print(f"  CUDA_VISIBLE_DEVICES: {args.cuda_visible_devices}")
    print("")

    success_count = 0
    fail_count = 0

    out_base = Path(args.output_dir)
    for idx, (folder_name, img_path) in enumerate(pairs):
        traj = trajs[(idx // args.group_size) % len(trajs)]
        output_subdir = out_base / f"{idx:04d}_{folder_name}_{traj}"
        ok = run_command(img_path, output_subdir, traj, cuda_visible_devices=args.cuda_visible_devices)
        if ok:
            success_count += 1
        else:
            fail_count += 1

    print(f"\n{'=' * 60}")
    print("批量处理完成!")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    print(f"  总计: {success_count + fail_count}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()


