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


# 实验7：彩色图像处理

> 📌 本文件夹仅含源码 `.m`。运行所需输入图片与第三方工具箱（BM3D/CBM3D）不在此目录，需从原项目根目录 `../BM3D` 获取并保证 MATLAB 路径包含它们。

## 文件清单

### `experiment7.m`
- **作用**：彩色图像处理主脚本。① RGB↔YCbCr/Lab/HSV/XYZ 颜色空间变换与通道显示；② 直接法（RGB 三通道独立）vs 间接法（变换域仅处理亮度通道）彩色去噪，比较 PSNR/SSIM；③ 色偏校正与雾霾去除；④ 超像素分割（SLIC）与彩色分割（IoU）。
- **使用方法**：直接运行。脚本会 `addpath('../BM3D')` 以使用 CBM3D；需同目录提供 `image_House256rgb.png`、`image_lena512rgb.png`、`yellowlily.jpg`。
- **依赖**：Image Processing Toolbox；BM3D 工具箱（`../BM3D`）。

### `experiment7_optimized.m`
- **作用**：实验7 的“课堂演示优化版”。提取 7 个辅助函数消除重复代码，支持 `DEMO_MODE`（1=步进暂停演示，0=全自动），每个任务可 Ctrl+Enter 单独运行，参数扫描分离到 `experiment7_tuning.m`。
- **使用方法**：直接运行；如需课堂分步演示将 `DEMO_MODE` 设为 1。依赖与 `experiment7.m` 相同。
- **依赖**：同上。

### `experiment7_tuning.m`
- **作用**：实验7 的参数敏感性调优脚本（独立运行）。覆盖任务3 的去噪参数（中值窗口 / Lab 高斯 sigma / YCbCr 双边强度 / HSV-V sigma / PCA 百分比）与任务4 的分割参数（H/S/V 阈值、形态学、双空间融合）。
- **使用方法**：直接运行，不依赖主文件变量，所有参数范围与实验要求一致。需同目录 `image_lena512rgb.png`、`yellowlily.jpg` 及 GT `yellowlily_gt.png`。
- **依赖**：同上。
