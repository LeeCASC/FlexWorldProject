# Dust3R 模型原理与 Pipeline 使用详解

## 一、概述

Dust3R (DUSt3R) 是一个用于从多视角图像进行3D重建的深度学习模型。它能够从单张或多张图像中重建出密集的3D点云，并估计相机参数（姿态、焦距等）。

## 二、Pipeline 中的使用流程

### 2.1 整体流程

在 `main_3dgs.py` 的 `dust3r_pipe()` 函数中，Dust3R 的使用流程如下：

```python
# 1. 初始化模型
dust3r = Dust3rWrapper(opt.dust3r)

# 2. 加载图像
dust3r.load_initial_images([input_image_path], opt)

# 3. 运行3D重建
dust3r.run_dust3r_init(bg_mask=background_mask)

# 4. 获取点云和相机参数
pm = dust3r.get_inital_pm()  # 点云 [H, W, 6] (xyz + rgb)
cam = dust3r.get_cams()[-1]  # 相机参数
```

### 2.2 核心函数详解

#### `run_dust3r_init()` - 主要重建流程

```python
def run_dust3r_init(self, input_images=None, clean_pc=True, bg_mask=None):
    # 步骤1: 构建图像对
    pairs = make_pairs(input_images, scene_graph='complete', ...)
    
    # 步骤2: 模型推理 - 获取成对的预测结果
    output = inference(pairs, self.dust3r, self.device, ...)
    
    # 步骤3: 创建全局对齐器
    scene = global_aligner(output, device=self.device, mode=PointCloudOptimizer)
    
    # 步骤4: 全局优化对齐
    loss = scene.compute_global_alignment(init='mst', niter=self.opts.niter, ...)
    
    # 步骤5: 获取深度图和点云
    self.depth = scene.get_depthmaps()[-1].detach()
    self.scene = scene.clean_pointcloud()
```

## 三、输入输出

### 3.1 输入

**图像输入格式：**
- 类型：`List[Dict]`，每个字典包含：
  - `'img'`: `[1, 3, H, W]` torch.Tensor，图像张量，值域 `[-1, 1]`
  - `'true_shape'`: `[H, W]` 原始图像尺寸
  - `'idx'`: 图像索引
  - `'instance'`: 图像路径/标识符
  - `'img_ori'`: 原始图像张量

**图像预处理：**
- 图像会被 resize 到 512x512（或配置的尺寸）
- 归一化到 `[-1, 1]` 范围
- 支持中心裁剪以适应模型输入

### 3.2 输出

**主要输出：**

1. **点云 (Point Cloud)**
   - 格式：`List[torch.Tensor]`，每个元素为 `[H, W, 6]`
   - 前3维：3D坐标 `(x, y, z)`（在相机坐标系或世界坐标系）
   - 后3维：RGB颜色 `(r, g, b)`，值域 `[0, 1]`
   - 通过 `get_pm()` 或 `get_inital_pm()` 获取

2. **深度图 (Depth Maps)**
   - 格式：`List[torch.Tensor]`，每个元素为 `[H, W]`
   - 表示每个像素的深度值
   - 通过 `scene.get_depthmaps()` 获取

3. **相机参数 (Camera Parameters)**
   - **姿态 (Pose)**: `c2w` 矩阵 `[4, 4]`，相机到世界的变换
   - **焦距 (Focal Length)**: 标量值
   - **主点 (Principal Point)**: `(cx, cy)` 坐标
   - 通过 `get_cams()` 获取 `Mcam` 对象列表

4. **置信度图 (Confidence Maps)**
   - 格式：`[H, W]`，表示每个像素预测的置信度
   - 用于过滤低质量点

## 四、模型架构与原理

### 4.1 模型架构：AsymmetricCroCo3DStereo

Dust3R 基于 **CroCo (Cross-view Completion)** 架构，是一个**非对称的双编码器-双解码器**结构：

```
输入图像对 (view1, view2)
    ↓
┌─────────────────────────────────────┐
│  编码器 (Encoder)                    │
│  - 两个共享权重的编码器              │
│  - 提取图像特征                      │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  解码器 (Decoder)                    │
│  - 两个独立的解码器 (dec_blocks)     │
│  - 交叉注意力机制                    │
│  - 融合两个视图的信息                │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  输出头 (Head)                       │
│  - 直接预测3D点坐标 (pts3d)          │
│  - 预测置信度 (conf)                 │
└─────────────────────────────────────┘
```

**关键特点：**
- **非对称性**：两个视图的3D点都预测在 view1 的坐标系中
- **端到端**：直接输出3D坐标，不需要先预测深度再转换
- **对称化训练**：推理时通过 `symmetrize=True` 增强鲁棒性

### 4.2 核心原理

#### 1. **成对图像处理 (Image Pairing)**

```python
pairs = make_pairs(input_images, scene_graph='complete', ...)
```

- **完全图模式** (`scene_graph='complete'`)：所有图像两两配对
- 对于 N 张图像，生成 `N*(N-1)/2` 个图像对
- 每个图像对独立通过模型推理

#### 2. **模型推理 (Inference)**

```python
output = inference(pairs, self.dust3r, self.device, batch_size=8)
```

**推理过程：**
- 对每个图像对 `(img_i, img_j)`：
  - 输入到模型：`pred1, pred2 = model(view1, view2)`
  - 输出：
    - `pred1['pts3d']`: view1 中每个像素的3D坐标（在view1坐标系）
    - `pred2['pts3d_in_other_view']`: view2 中每个像素的3D坐标（在view1坐标系）
    - `pred1['conf']`, `pred2['conf']`: 置信度图

**关键点：**
- 两个视图的3D点都预测在**同一个坐标系**（view1的坐标系）中
- 这使得后续的全局对齐成为可能

#### 3. **全局对齐 (Global Alignment)**

这是 Dust3R 的核心创新点：

```python
scene = global_aligner(output, device=self.device, mode=PointCloudOptimizer)
loss = scene.compute_global_alignment(init='mst', niter=self.opts.niter, ...)
```

**对齐目标：**
- 将多对图像的预测结果统一到**全局坐标系**
- 优化相机姿态、焦距、深度图，使得：
  - 同一3D点在多个视图中的投影一致
  - 不同视图预测的3D点对齐

**优化变量：**
- `im_poses`: 每张图像的相机姿态 `[4, 4]` (c2w)
- `im_focals`: 每张图像的焦距
- `im_depthmaps`: 每张图像的深度图 `[H, W]`
- `im_pp`: 主点坐标（可选）

**优化方法：**
- 使用 **MST (Minimum Spanning Tree)** 初始化
- 迭代优化（通常 300-500 次迭代）
- 损失函数：基于3D点的一致性约束

**对齐过程：**
1. **初始化**：使用 MST 构建初始相机图，估计相对姿态
2. **迭代优化**：
   - 从深度图和相机参数计算3D点
   - 计算不同视图间对应点的3D距离
   - 反向传播优化相机参数和深度
3. **收敛**：当损失不再下降时停止

#### 4. **点云生成**

```python
pts = scene.get_pts3d()  # 从深度图和相机参数计算3D点
```

对于每张图像：
- 使用深度图 `depth[H, W]` 和相机内参 `(f, cx, cy)`
- 通过反投影计算每个像素的3D坐标：
  ```
  x = (u - cx) * depth / f
  y = (v - cy) * depth / f
  z = depth
  ```
- 转换到世界坐标系：`pts3d_world = c2w @ pts3d_camera`

### 4.3 坐标系转换

Dust3R 使用 **RDF (Right-Down-Forward)** 坐标系：
- R: 右 (X+)
- D: 下 (Y+)
- F: 前 (Z+)

而 OpenGL 使用 **RUB (Right-Up-Back)** 坐标系，因此需要转换：

```python
# dust3r -> OpenGL
pts_tmp[:,:, [1,2]] = -pts_tmp[:,:, [1,2]]  # 翻转Y和Z轴
```

## 五、Pipeline 中的具体应用

### 5.1 单图像重建流程

```python
# 1. 加载图像
images, img_ori = dust3r.load_initial_images([input_image_path], opt)

# 2. 运行重建（单图像会复制一份形成图像对）
dust3r.run_dust3r_init(bg_mask=None)

# 3. 获取点云（包含颜色）
pm = dust3r.get_inital_pm()  # [H, W, 6]

# 4. 获取相机参数
cam = dust3r.get_cams()[-1]  # Mcam对象

# 5. 转换到世界坐标系
pm[...,:3] = pm[...,:3] @ cam.getW2C()[:3, :3].T + cam.getW2C()[:3, 3].T
```

### 5.2 多视图扩展

在 `view_expand_pipe()` 中，Dust3R 用于处理新视角：

```python
# 对于新的相机轨迹，使用预设的相机参数
dust3r.run_dust3r_preset(input_images, cams)
```

这样可以：
- 利用已知的相机参数
- 只优化深度图
- 快速处理新视角

### 5.3 背景替换

当有背景掩码时：

```python
dust3r.run_dust3r_init(bg_mask=background_mask)
```

- 为背景创建单独的相机姿态
- 缩放深度到特定范围 `[0.1, 0.4]`
- 使用 `refine_depth()` 融合前景和背景深度

## 六、关键参数

### 6.1 模型参数

- `model_path`: 预训练模型路径
- `batch_size`: 推理批次大小（默认8）
- `device`: 计算设备（'cuda' 或 'cpu'）

### 6.2 优化参数

- `niter`: 全局对齐迭代次数（通常300-500）
- `lr`: 学习率（通常0.01-0.001）
- `schedule`: 学习率调度策略
- `min_conf_thr`: 最小置信度阈值（过滤低质量点）

### 6.3 图像处理参数

- `size`: 输入图像尺寸（默认512）
- `force_1024`: 是否强制使用1024尺寸
- `square_ok`: 是否允许正方形图像

## 七、优势与特点

1. **端到端学习**：直接预测3D坐标，无需中间表示
2. **全局一致性**：通过全局对齐保证多视图一致性
3. **单图像支持**：可以从单张图像重建（通过自配对）
4. **密集重建**：每个像素都有3D坐标和颜色
5. **相机参数估计**：同时估计相机内参和外参

## 八、注意事项

1. **内存消耗**：完全图模式会生成大量图像对，内存需求大
2. **计算时间**：全局对齐需要多次迭代，可能较慢
3. **坐标系**：注意 RDF 和 RUB 坐标系的转换
4. **深度范围**：深度值可能需要根据场景调整
5. **置信度过滤**：使用 `min_conf_thr` 过滤低质量点

## 九、相关文件

- `ops/dust3r.py`: Dust3rWrapper 封装类
- `tools/dust3r/dust3r/model.py`: 模型定义
- `tools/dust3r/dust3r/inference.py`: 推理函数
- `tools/dust3r/dust3r/cloud_opt/optimizer.py`: 全局对齐优化器
- `tools/dust3r/dust3r/image_pairs.py`: 图像对生成

