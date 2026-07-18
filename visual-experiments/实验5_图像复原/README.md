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

# 实验5：图像复原与重建

> 📌 本文件夹仅含源码 `.m`。运行所需输入图片不在此目录，需从原项目根目录获取。

## 文件清单

### `experiment5.m`
- **作用**：图像复原与 CT 重建。① 高斯模糊+噪声退化后，用 `deconvwnr` / `deconvreg` / `deconvlucy` / `deconvblind` 四种方法复原并比较 PSNR/SSIM；② 运动模糊+噪声图像盲去卷积；③ `phantom` 仿真脑图 + Radon 变换 / 逆 Radon 重建（18/36/90 个角度）。
- **使用方法**：直接运行。需同目录提供 `lena.png`、`cameraman.png`、`cameraman256_b_n.png`。
- **依赖**：Image Processing Toolbox。

### `multi_scale_deconvblind.m`
- **作用**：函数 `[img_restored, PSF_est] = multi_scale_deconvblind(img, init_psf, noise_std, n_scales, base_iter, base_damp_factor)`，实现多尺度由粗到精（Coarse-to-Fine）盲去卷积，参考 Fergus et al. SIGGRAPH 2006，逐层估计 PSF 并精调，最后用 `deconvreg` 非盲复原，避免局部最优。
- **使用方法**：作为函数调用，通常由 `experiment5.m` 调用；也可单独对退化图像做盲复原。
- **依赖**：Image Processing Toolbox。
