"""
从 Hugging Face 下载 FlexWorld 模型的所有文件
"""
import os
from huggingface_hub import snapshot_download

def download_flexworld():
    """下载 GSAI-ML/FlexWorld 模型的所有文件到当前目录"""
    repo_id = "GSAI-ML/FlexWorld"
    local_dir = "."  # 当前文件夹
    
    print(f"开始下载 {repo_id} 的所有文件...")
    print(f"保存位置: {os.path.abspath(local_dir)}")
    print("这可能需要一些时间，请耐心等待...\n")
    
    try:
        # 下载仓库的所有文件
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # 不使用符号链接，完整下载文件
            resume_download=True  # 支持断点续传
        )
        print(f"\n✓ 下载完成！文件已保存到: {os.path.abspath(local_dir)}")
    except Exception as e:
        print(f"\n✗ 下载失败: {str(e)}")
        print("\n提示:")
        print("1. 请确保已安装 huggingface_hub: pip install huggingface_hub")
        print("2. 如果模型需要认证，请先登录: huggingface-cli login")
        raise

if __name__ == "__main__":
    download_flexworld()

