#!/usr/bin/env python3
"""
批量执行 video_generate.py 脚本
支持批量处理多个输入图片和多个轨迹类型
"""
import os
import subprocess
import argparse
import time
from pathlib import Path

# 支持的轨迹类型
VALID_TRAJS = ["up", "down", "left", "right", "forward", "backward", "rotate_left", "rotate_right"]

def run_command(input_image_path, output_dir, traj, cuda_visible_devices=None):
    """执行单个视频生成命令"""
    cmd = [
        "python", "video_generate.py",
        "--input_image_path", str(input_image_path),
        "--output_dir", str(output_dir),
        "--traj", traj
    ]

    env = os.environ.copy()
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    
    print(f"\n{'='*60}")
    print(f"正在处理:")
    print(f"  输入图片: {input_image_path}")
    print(f"  输出目录: {output_dir}")
    print(f"  轨迹类型: {traj}")
    if cuda_visible_devices is not None:
        print(f"  CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
    print(f"{'='*60}\n")
    
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

def batch_process(args):
    """批量处理函数"""
    # 确定输入图片列表
    if args.input_image_path:
        # 单个图片或图片列表
        input_images = [Path(p.strip()) for p in args.input_image_path.split(',')]
    elif args.input_dir:
        # 从目录中读取所有图片
        input_dir = Path(args.input_dir)
        image_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
        input_images = [f for f in input_dir.iterdir() if f.is_file() and f.suffix in image_extensions]
        input_images.sort()
    else:
        # 默认使用 assets 目录
        assets_dir = Path("./assets")
        image_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
        input_images = [f for f in assets_dir.iterdir() if f.suffix in image_extensions]
        input_images.sort()
    
    # 确定轨迹列表
    if args.traj:
        trajs = [t.strip() for t in args.traj.split(',') if t.strip()]
    else:
        trajs = VALID_TRAJS
    
    # 验证所有轨迹是否有效
    for traj in trajs:
        if traj not in VALID_TRAJS:
            print(f"警告: 轨迹类型 '{traj}' 不在支持列表中: {VALID_TRAJS}")
    
    # 验证所有图片是否存在
    valid_images = []
    for img_path in input_images:
        if not img_path.exists():
            print(f"警告: 图片不存在，跳过: {img_path}")
        else:
            valid_images.append(img_path)
    
    if not valid_images:
        print("错误: 没有找到有效的输入图片")
        return

    print(f"\n批量处理配置:")
    print(f"  输入图片数量: {len(valid_images)}")
    print(f"  轨迹类型数量: {len(trajs)}")
    print(f"  总任务数: {len(valid_images) * len(trajs)}")
    print(f"  基础输出目录: {args.output_dir}\n")
    
    # 执行批量处理
    success_count = 0
    fail_count = 0

    for img_path in valid_images:
        for traj in trajs:
            # 生成输出目录名称
            # 例如: ./results-single-traj/room_backward
            img_name = img_path.stem  # 不带扩展名的文件名
            output_subdir = Path(args.output_dir) / f"{img_name}_{traj}"

            success = run_command(img_path, output_subdir, traj, cuda_visible_devices=args.cuda_visible_devices)
            if success:
                success_count += 1
            else:
                fail_count += 1
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"批量处理完成!")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    print(f"  总计: {success_count + fail_count}")
    print(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(
        description='批量执行 video_generate.py 脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
示例用法:

1. 处理单个图片的所有轨迹类型:
   python batch_video_generate.py --input_image_path ./assets/room.png

2. 处理单个图片的指定轨迹类型:
   python batch_video_generate.py --input_image_path ./assets/room.png --traj backward,forward

3. 处理目录下所有图片的所有轨迹类型:
   python batch_video_generate.py --input_dir ./assets

4. 处理多个图片的多个轨迹类型:
   python batch_video_generate.py --input_image_path ./assets/room.png,./assets/house.png --traj backward,forward

支持的轨迹类型: {', '.join(VALID_TRAJS)}
        """
    )
    
    parser.add_argument(
        '--input_image_path',
        type=str,
        help='输入图片路径（单个或多个，用逗号分隔）。例如: ./assets/room.png 或 ./assets/room.png,./assets/house.png'
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        help='输入图片目录（将处理目录下所有图片文件）'
    )
    
    parser.add_argument(
        '--traj',
        type=str,
        help=f'轨迹类型（单个或多个，用逗号分隔）。如果不指定，将处理所有轨迹类型。支持: {", ".join(VALID_TRAJS)}'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results-single-traj',
        help='输出目录（默认为 ./results-single-traj）'
    )

    parser.add_argument(
        '--cuda_visible_devices',
        type=str,
        default=None,
        help='设置 CUDA_VISIBLE_DEVICES（例如 0 或 0,1）。不设置则沿用当前环境变量/默认行为。'
    )
    
    args = parser.parse_args()
    
    # 如果既没有指定图片路径也没有指定目录，默认使用 assets 目录
    if not args.input_image_path and not args.input_dir:
        print("未指定输入图片或目录，将使用默认的 ./assets 目录\n")
    
    batch_process(args)

if __name__ == "__main__":
    main()
