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


# 实验3：图像增强

> 📌 本文件夹仅含源码 `.m`。运行所需输入图片与第三方工具箱（BM3D 等）不在此目录，需从原项目根目录 `../BM3D` 获取并保证 MATLAB 路径包含它们。

## 文件清单

### `experiment3.m`
- **作用**：图像增强综合对比。① 眼底图像增强：`imadjust` / `histeq` / `adapthisteq`(CLAHE)，用 NIQE 客观评价；② 锐化：拉普拉斯算子与反锐化掩模，NIQE 对比；③ 去噪：对 lena/cameraman/house 加高斯噪声后，对比 高斯卷积、中值滤波、双边滤波、NLM、TV、DnCNN、BM3D 共 7 类（含深度学习方法），输出 PSNR/SSIM。
- **使用方法**：直接运行。脚本会自动 `addpath('../BM3D')` 加载 BM3D 工具箱（见仓库根目录 `BM3D/`），请确保该路径存在；需同目录提供 `fundus.png`、`lena.png`、`cameraman.png`、`house.png`。
- **依赖**：Image Processing Toolbox；BM3D 工具箱（第三方，`../BM3D`）；DnCNN 需要 Deep Learning Toolbox 或自带的 `denoiseImage` 支持。
