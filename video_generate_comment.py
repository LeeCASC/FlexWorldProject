"""
Video Generation Pipeline for FlexWorld

整体思路：
==========

这个 pipeline 实现了从单张图像生成新视角视频的完整流程：

1. 【输入处理】加载输入图像，解析相机轨迹参数
2. 【3D重建】使用 Dust3R 从单张图像重建3D点云场景
   - 加载图像并初始化 Dust3R 模型
   - 运行 Dust3R 进行深度估计和3D重建
   - 生成初始点云（Point Cloud）
3. 【点云渲染】根据相机轨迹渲染点云，生成粗糙的视角序列
   - 使用点云渲染器沿着轨迹渲染每一帧
   - 生成初始的视频帧序列（可能包含空洞和伪影）
4. 【视频扩散】使用视频扩散模型（CogVideo/ViewCrafter）增强渲染结果
   - 将粗糙的渲染帧输入到扩散模型
   - 扩散模型生成高质量、连贯的新视角视频帧
   - 保持第一帧为原始输入图像以确保一致性
5. 【后处理与输出】保存生成的视频

Pipeline 特点：
- 支持多种相机轨迹：简单指令（forward/backward/left/right等）或视频文件
- 使用扩散模型提升渲染质量，生成更真实的新视角
- 基于3D点云，保证几何一致性
"""

from PIL import Image
import numpy as np
import os
from omegaconf import OmegaConf
import argparse

from ops.cam_utils import CamPlanner, Mcam
from ops.utils.general import seed_everything, easy_save_video
from ops.utils.all_traj import get_traj
from ops.dust3r import Dust3rWrapper
from ops.PcdMgr import PcdMgr
from pipe.view_extend import Video_Tool
from ops.utils.logger import Ulog


def dust3r_pipe(opt, input_image_path, output_dir, name):
    """
    3D重建 Pipeline：从单张图像重建3D点云
    
    流程：
    1. 初始化 Dust3R 模型（用于单目深度估计和3D重建）
    2. 加载输入图像并预处理
    3. 运行 Dust3R 进行深度估计和3D点云生成
    4. 转换点云到世界坐标系
    5. 保存初始点云文件
    
    Args:
        opt: 配置对象
        input_image_path: 输入图像路径
        output_dir: 输出目录
        name: 场景名称
    
    Returns:
        pcd: PcdMgr 对象，包含重建的3D点云
        dust3r: Dust3rWrapper 对象，用于后续处理
    """
    # 初始化 Dust3R 模型
    dust3r = Dust3rWrapper(opt.dust3r)
    
    # 加载并预处理输入图像
    dust3r.load_initial_images([input_image_path], opt)
    global H, W
    H, W = dust3r.images[0]["img"].shape[2:]
    print(f"Input image size: {H}x{W}")
    
    # 运行 Dust3R 进行3D重建
    # background_mask=None 表示不使用背景掩码（可以后续添加分割功能）
    background_mask = None
    dust3r.run_dust3r_init(bg_mask=background_mask)
    
    # 获取初始点云（包含位置和颜色信息）
    bg_pm, pm = dust3r.get_inital_pm()
    cam = dust3r.get_cams()[-1]  # 获取参考相机
    
    # 将点云从相机坐标系转换到世界坐标系
    pm = pm.cpu().numpy()
    pm[..., :3] = pm[..., :3] @ cam.getW2C()[:3, :3].T + cam.getW2C()[:3, 3].T
    
    # 设置默认焦距（用于后续相机设置）
    Mcam.set_default_f(cam.f)
    print(f"Default focal length: {Mcam.default_f}")
    
    # 创建点云管理器对象
    pcdtmp = PcdMgr(pts3d=pm.reshape(-1, 6))  # 6维：xyz + rgb
    
    # 保存初始点云（用于调试和可视化）
    os.makedirs(output_dir, exist_ok=True)
    pcdtmp.save_ply(f"{output_dir}/{name}_pcd_ori.ply")
    print(f"Initial point cloud saved to: {output_dir}/{name}_pcd_ori.ply")
    
    return pcdtmp, dust3r


def video_generate(image, traj, pcd, opt, dust3r, output_dir, name, prompts=None):
    """
    视频生成 Pipeline：从点云和相机轨迹生成新视角视频
    
    流程：
    1. 使用点云沿着相机轨迹渲染粗糙的视频帧
    2. 使用视频扩散模型增强渲染结果，生成高质量视频
    
    Args:
        image: 输入图像 (numpy array, [H, W, 3], 0-255)
        traj: 相机轨迹列表
        pcd: PcdMgr 对象，包含重建的3D点云
        opt: 配置对象
        dust3r: Dust3rWrapper 对象
        output_dir: 输出目录
        name: 场景名称
        prompts: 可选的文本提示（用于引导扩散模型）
    
    Returns:
        video: 生成的视频帧序列 (torch.Tensor, [T, H, W, 3], 0-1)
    """
    # 初始化视频生成工具（包含扩散模型）
    video_tool = Video_Tool(opt, dust3r)
    
    # 为所有相机设置统一的焦距
    for cam in traj:
        cam.set_cam(f=Mcam.default_f)
    
    # 可选：移除点云中的离群点（当前注释掉，可根据需要启用）
    # pcd.remove_outliers_near()
    
    # 运行视频生成 pipeline
    # 内部流程：
    # 1. 使用点云渲染器沿着轨迹渲染每一帧 -> render_results
    # 2. 将渲染结果输入扩散模型 -> diffusion_results
    # 3. 确保第一帧保持为原始输入图像
    video = video_tool.run(traj, image, pcd, prompts=prompts, logger=False)
    
    return video


def main_pipeline(opt, input_image_path, output_dir, name, traj_instruction):
    """
    主 Pipeline：完整的视频生成流程
    
    Args:
        opt: 配置对象
        input_image_path: 输入图像路径
        output_dir: 输出目录
        name: 场景名称
        traj_instruction: 相机轨迹指令（如 "forward", "backward" 或视频文件路径）
    """
    # ========== 步骤1: 输入处理 ==========
    print("\n" + "="*50)
    print("Step 1: Loading input image and trajectory")
    print("="*50)
    
    # 加载并预处理输入图像
    image = Image.open(input_image_path)
    # 调整图像尺寸（可根据需要修改）
    image = image.resize((1024, 576))
    image = np.array(image.convert("RGB"))
    print(f"Input image loaded: {image.shape}")
    
    # 解析相机轨迹
    # 支持：简单指令（"forward", "backward", "left", "right"等）或视频文件路径
    traj = get_traj(traj_instruction)
    print(f"Camera trajectory loaded: {len(traj)} frames")
    
    # ========== 步骤2: 3D重建 ==========
    print("\n" + "="*50)
    print("Step 2: 3D Reconstruction with Dust3R")
    print("="*50)
    
    pcd, dust3r = dust3r_pipe(opt, input_image_path, output_dir, name)
    print(f"Point cloud reconstructed: {len(pcd.pts)} points")
    
    # ========== 步骤3: 视频生成 ==========
    print("\n" + "="*50)
    print("Step 3: Video Generation with Diffusion Model")
    print("="*50)
    
    # 可选：添加文本提示引导扩散模型
    prompts = None  # 可以设置为列表，如 ["a beautiful room", "modern interior"]
    
    video = video_generate(image, traj, pcd, opt, dust3r, output_dir, name, prompts=prompts)
    print(f"Video generated: {video.shape}")
    
    # ========== 步骤4: 保存输出 ==========
    print("\n" + "="*50)
    print("Step 4: Saving output video")
    print("="*50)
    
    output_video_path = f"{output_dir}/{name}.mp4"
    easy_save_video(video, output_video_path)
    print(f"Video saved to: {output_video_path}")
    
    return video


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Generation Pipeline for FlexWorld")
    parser.add_argument('--name', type=str, default="test", 
                       help='Name of the configuration and output files')
    parser.add_argument('--traj', type=str, default="forward", 
                       help='Camera trajectory instruction (e.g., "forward", "backward", "left", "right", "rotate_left", "rotate_right") or path to a video file')
    parser.add_argument('--basic_opt', type=str, default='configs/basic.yaml', 
                       help='Path to basic configuration file')
    parser.add_argument('--config', type=str, default='configs/examples/test.yaml',
                       help='Path to example configuration file')
    parser.add_argument('--output_dir', type=str, default="./results-single-traj", 
                       help='Output directory for generated videos')
    parser.add_argument('--input_image_path', type=str, default="./assets/room.png", 
                       help='Path to input image')
    
    args = parser.parse_args()
    
    # 加载配置
    basic_opt = OmegaConf.load(args.basic_opt)
    config_opt = OmegaConf.load(args.config)
    opt = OmegaConf.merge(basic_opt, config_opt)
    opt.name = args.name
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置随机种子（保证可复现性）
    seed_everything(opt)
    
    # 可选：初始化日志系统
    # os.makedirs('./cache', exist_ok=True)
    # Ulog.create(f"video_generate_{args.name}", rootdir="./cache")
    # Ulog().add_code(__file__)
    
    # 运行主 pipeline
    video = main_pipeline(
        opt=opt,
        input_image_path=args.input_image_path,
        output_dir=args.output_dir,
        name=args.name,
        traj_instruction=args.traj
    )
    
    print("\n" + "="*50)
    print("Pipeline completed successfully!")
    print("="*50)

