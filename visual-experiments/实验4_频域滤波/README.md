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


# 实验4：频域滤波

> 📌 本文件夹仅含源码 `.m`。运行所需输入图片不在此目录，需从原项目根目录获取。

## 文件清单

### `experiment4.m`
- **作用**：基于傅里叶变换的频域分析。① 计算 lena 频谱/相角，仅用幅度或仅用相位重建对比；② 高斯卷积（空域）vs 高斯低通（频域）去噪，比较 PSNR/SSIM；③ 三维频谱可视化；④ DCT 变换矩阵与反变换；⑤ 3D-DCT 彩色图像硬阈值去噪。
- **使用方法**：直接运行。需同目录提供 `lena.png`（`rgb2gray` 后与彩色 lena 均用到）。
- **依赖**：Image Processing Toolbox。

### `tune_parameters.m`
- **作用**：参数调优辅助脚本。采用“粗扫 + 细扫”分级策略，以 PSNR/SSIM/EPR（边缘保留度）多指标 + Knee-point 拐点检测 + 加权综合评分，自动搜索最优参数，结果保存到 `optimal_params.mat` 供 `experiment4.m` 加载。
- **使用方法**：可独立运行（不依赖 `experiment4.m`）。确保同目录有 `lena.png`，运行后查看控制台输出与曲线，最优参数写入 `optimal_params.mat`。
- **依赖**：Image Processing Toolbox。
