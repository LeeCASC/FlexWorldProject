#!/bin/bash
# 批量执行 video_generate.py 脚本的简单版本
# 用法: ./batch_video_generate.sh [输入图片路径] [轨迹类型]

# 默认值
INPUT_IMAGE="${1:-./assets/room.png}"
OUTPUT_DIR="${2:-./results-single-traj}"
TRAJ="${3:-backward}"

# 支持的轨迹类型
TRAJS=("up" "down" "left" "right" "forward" "backward" "rotate_left" "rotate_right")

# 如果传入的轨迹是 "all"，则处理所有轨迹类型
if [ "$TRAJ" == "all" ]; then
    echo "将处理所有轨迹类型: ${TRAJS[*]}"
    for traj in "${TRAJS[@]}"; do
        output_subdir="${OUTPUT_DIR}_${traj}"
        echo "=========================================="
        echo "处理: $INPUT_IMAGE - 轨迹: $traj"
        echo "输出目录: $output_subdir"
        echo "=========================================="
        python video_generate.py \
            --input_image_path "$INPUT_IMAGE" \
            --output_dir "$output_subdir" \
            --traj "$traj"
        
        if [ $? -eq 0 ]; then
            echo "✓ 成功完成: $traj"
        else
            echo "✗ 处理失败: $traj"
        fi
        echo ""
    done
else
    # 处理单个轨迹类型
    echo "处理: $INPUT_IMAGE"
    echo "轨迹类型: $TRAJ"
    echo "输出目录: $OUTPUT_DIR"
    echo ""
    
    python video_generate.py \
        --input_image_path "$INPUT_IMAGE" \
        --output_dir "$OUTPUT_DIR" \
        --traj "$TRAJ"
fi
