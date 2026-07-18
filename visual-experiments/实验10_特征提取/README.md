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


# 实验10：特征提取与表示

> 📌 本文件夹仅含源码 `.m`。运行所需输入图片不在此目录，需从原项目根目录获取。

## 文件清单

### `experiment10.m`
- **作用**：图像特征提取与表示。① 对平坦/边缘/纹理三类区域用灰度共生矩阵（GLCM）提取对比度、相关性、能量、同质性等纹理特征；② PCA 主成分分析与图像重建；③ SVD 低秩近似；④ WNNM 去噪（调用 `WNNM_denoising.m`）。
- **使用方法**：直接运行。需同目录提供 `lena.png`。
- **依赖**：Image Processing Toolbox；Statistics and Machine Learning Toolbox（`pca`）。

### `WNNM_denoising.m`
- **作用**：函数 `[denoised, PSNR_vec] = WNNM_denoising(noisy, clean, sigma, params)`，实现 Gu et al. CVPR 2014 的加权核范数最小化（WNNM）去噪，对每个图像块在相似块群组上做加权奇异值软阈值。
- **使用方法**：作为函数调用，如 `[denoised, psnr_vec] = WNNM_denoising(noisy_img, clean_img, 30/255, struct());`；`clean` 可留空（仅用于中间 PSNR 计算）。通常由 `experiment10.m` 调用。
- **依赖**：无（纯 MATLAB，部分实现可能用到 Image Processing Toolbox）。
