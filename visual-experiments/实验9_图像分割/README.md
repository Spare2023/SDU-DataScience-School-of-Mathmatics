```text
======================================================================
Copyright (c) 2026 Spare2023
Repository: https://github.com/Spare2023/SDU-DataScience-School-of-Mathematics
Licensed under the BSD 3-Clause License

【使用限制说明】
1. 本代码仅用于本校课程学习参考，禁止无修改直接复制提交课程大作业；
2. 禁止未经许可将本代码用于商业项目、对外发表、冒充个人原创成果；
3. 若修改后使用，需保留完整版权声明，不得删除本段注释；
4. 因违规使用代码产生的抄袭、处分等全部后果由使用者自行承担。
======================================================================
```

# 实验9：图像分割

> 📌 本文件夹仅含源码 `.m`。部分脚本依赖第三方工具箱（BM3D）与训练好的模型（`.mat`），且 3 个 U-Net 训练脚本需要自行下载公开医学数据集（见文末）。请在使用前准备好相应数据并保证 MATLAB 路径正确。

## 文件清单

### `experiment9_task1.m`
- **作用**：任务1 遥感图控点分割。用 HSV 颜色阈值提取红色 L 型标记线，结合 Canny 边缘检测、Hough 直线检测、直线贪心组合拼接、Harris 角点检测定位控点中心；含形态学后处理。
- **使用方法**：直接运行。需同目录提供 `control.jpg`。
- **依赖**：Image Processing Toolbox。

### `experiment9_task2.m`
- **作用**：任务2 CNV（脉络膜新生血管）OCT 图像分割。Part A 传统方法（多阈值 Otsu / K-means / GrabCut / 主动轮廓）；Part B 调用 `task2_partB_unet.m` 的 U-Net 深度学习分割。
- **使用方法**：直接运行。需同目录提供 `cnv.png`（OCT 图像）与 `mask_cnv.png`（金标准掩膜）。
- **依赖**：Image Processing Toolbox；U-Net 部分需 Deep Learning Toolbox + 已训练模型 `unet_cnv_model.mat`（由 `task2_partB_unet.m` 生成）。

### `experiment9_task3.m`
- **作用**：任务3 眼底血管分割。对比传统自适应阈值方法（绿色通道 + CLAHE + 自适应二值化 + 形态学后处理）与 U-Net 深度学习方法。
- **使用方法**：直接运行。需同目录提供 `vessels.tif`（眼底图像）；若启用 U-Net 路径，还需预训练模型 `unet_task3_model.mat`（由 `experiment9_fives_unet.m` 训练生成）。
- **依赖**：Image Processing Toolbox；Deep Learning Toolbox（U-Net 路径）。

### `CoyeFilter.m`
- **作用**：Tyler L. Coye (2015) 的视网膜血管分割算法（RGB→PCA 灰度 + CLAHE + 多尺度 top-hat + 阈值），作为传统血管分割的参考/对比实现。**注意：本文件保留原作者版权声明，请勿删除。**
- **使用方法**：作为函数/脚本调用，输入图像路径在文件内硬编码（如 `13_right.jpeg`），需按需修改。
- **依赖**：Image Processing Toolbox。

### `isodata.m`
- **作用**：函数 `level = isodata(I)`，用 ISODATA（Ridler-Calvard 迭代）法计算全局图像分割阈值，供分割脚本调用。
- **使用方法**：作为函数调用，如 `level = isodata(gray); BW = im2bw(I, level);`。
- **依赖**：无（纯 MATLAB）。

### `experiment9_chasedb1_unet.m`
- **作用**：训练 U-Net 分割 **CHASEDB1** 视网膜血管（实验9 扩展）。Dice+CE 损失、CosineAnnealing 学习率、早停；输出可用于血管分割的 U-Net 模型。
- **使用方法**：直接运行。需准备好 CHASEDB1 数据集（见下方“训练网络所需数据库”），组织为 `chasedb1/train/{input,label}` 与 `chasedb1/val/{input,label}`（脚本中 `DATA_ROOT` 指向同目录下的 `chasedb1/`）。建议 GPU（RTX 4060 8GB 或同级）。
- **依赖**：Deep Learning Toolbox；GPU 强烈推荐。

### `experiment9_fives_unet.m`
- **作用**：训练 U-Net 分割 **FIVES** 眼底血管，并在 `vessels.tif` 上测试（优化版 v2，预加载到内存 + 自定义训练循环，显著加速）。输出 `unet_task3_model.mat`（被 `experiment9_task3.m` 加载）。
- **使用方法**：直接运行。需准备好 FIVES 数据集与 `vessels.tif`、`mask_vessels.gif`（GT），数据集目录在脚本中 `FIVES_DIR` 指向同目录下的 “FIVES A Fundus Image Dataset for AI-based Vessel Segmentation”。建议 GPU（脚本按 RTX 3050 4GB 调过 batch=16）。
- **依赖**：Deep Learning Toolbox；GPU 强烈推荐。

### `task2_partB_unet.m`
- **作用**：任务2 Part B 的 U-Net CNV 分割独立训练/推理脚本。支持两种模式：`'single'`（仅用 `cnv.png` 单图 patch 训练，CPU 约 20 分钟）；`'transfer'`（AMD-SD 预训练 + `cnv.png` 微调，需 AMD-SD 数据集）。输出 `unet_cnv_model.mat`。
- **使用方法**：直接运行。方案A 需同目录 `cnv.png`、`mask_cnv.png`；方案B 需额外准备 AMD-SD 数据集（见下方）。`TRAIN_MODE` 变量切换模式。
- **依赖**：Deep Learning Toolbox；方案B 需 GPU。

### `tune_unet_params_groupA.m`
- **作用**：U-Net 后处理参数网格搜索（σ 高斯平滑 × 阈值二值化），先训练/加载 `unet_cnv_model.mat` 得到 score_map，再快速扫描得出最佳组合并绘制热力图。
- **使用方法**：直接运行（会自动调用 `task2_partB_unet.m` 训练若模型不存在）。需同目录 `cnv.png`、`mask_cnv.png`。
- **依赖**：同 `task2_partB_unet.m`。

---

## ⚠️ 训练神经网络所需数据库（重点）

实验9 中 3 个 U-Net 训练脚本需要不同的公开医学图像数据集。请在使用前自行下载并按规定目录结构放置（**数据集不随本仓库分发**，使用时请遵守各数据集的原始许可协议）：

### 1. CHASEDB1（视网膜血管分割，对应 `experiment9_chasedb1_unet.m`）
- **内容**：~28 张眼底彩色图像（20 张训练 / 8 张验证），附血管标注掩膜。
- **获取**：从原始论文/公开镜像获取；亦可搜索 “CHASEDB1” 在 Kaggle 或学校镜像下载。
- **目录结构**（放在脚本同目录下的 `chasedb1/`）：
  ```
  chasedb1/
  ├── train/
  │   ├── input/   *.png   (原始眼底图)
  │   └── label/   *.png   (血管 GT 掩膜)
  └── val/
      ├── input/
      └── label/
  ```

### 2. FIVES（眼底血管分割，对应 `experiment9_fives_unet.m`）
- **内容**：FIVES – *A Fundus Image Dataset for AI-based Vessel Segmentation*，含大量眼底图与血管标注，用于训练 `vessels.tif` 的血管分割 U-Net。
- **获取**：IEEE DataPort 等公开渠道下载（注意其使用许可，仅限研究/课程学习）。
- **目录结构**：脚本 `FIVES_DIR` 指向同目录下的文件夹 `FIVES A Fundus Image Dataset for AI-based Vessel Segmentation`（内部按原数据集组织）。
- **附带文件**：还需同目录提供测试图 `vessels.tif` 与其金标准 `mask_vessels.gif`（随本实验原始材料提供）。

### 3. AMD-SD（AMD 标准数据库，对应 `task2_partB_unet.m` 方案B 迁移学习）
- **内容**：年龄相关性黄斑变性（AMD）OCT 图像库，用于 CNV 分割的预训练/微调。
- **获取**：从 AMD-SD 官方发布渠道下载，并按脚本指示预处理。
- **目录结构**（脚本默认路径，可修改）：
  ```
  E:\视觉与数据计算\数据集\AMD-SD\preprocessed\
  ├── train/
  └── val/
  ```
  若已运行 `prepare_amd_sd.m` 预处理，则指向上一步的输出目录；路径不对请自行修改脚本中的 `AMD_SD_TRAIN_DIR` / `AMD_SD_VAL_DIR`。

### 4. 单图自训练（无需额外数据库，`task2_partB_unet.m` 方案A）
- 直接用随本实验提供的 `cnv.png` + `mask_cnv.png` 做单图 patch 训练，CPU 即可（约 20 分钟），不需要下载任何外部数据库。

### 训练硬件与环境建议
- **环境**：MATLAB R2025a + Deep Learning Toolbox。
- **硬件**：建议使用 NVIDIA GPU（脚本针对 RTX 4060 8GB / RTX 3050 4GB 调参；无 GPU 时 CPU 可训练但极慢）。
- **数据合规**：上述数据集仅用于本校课程学习，请遵守各数据集的原始许可协议，不得用于商业或对外发表。
