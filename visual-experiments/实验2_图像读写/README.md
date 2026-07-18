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

# 实验2：图像读写与基本操作

> 📌 本文件夹仅含源码 `.m`。运行所需输入图片与第三方工具箱不在此目录，需从原项目根目录获取。

## 文件清单

### `experiment2.m`
- **作用**：图像读写、裁剪、减采样/放大、插值方法对比与质量评价。包含：① 读入 lena 并裁剪子图、绘制第 100 行灰度曲线；② 4 倍减采样后用 5 种方法（最近邻 / 双线性 / 双三次 / Lanczos3 / IBP 超分辨率）放大回原尺寸；③ 计算 PSNR/SSIM 并显示差图像；④ 直方图均衡增强与对比。
- **使用方法**：直接运行。需同目录提供 `lena.png`。脚本内部引用减采样结果 `lena4s`/`lena4l`，首次运行应先执行减采样任务生成这两个变量再跑后续放大任务。
- **依赖**：MATLAB Image Processing Toolbox；`ibp_upsample.m`（同目录）。

### `ibp_upsample.m`
- **作用**：函数 `HR = ibp_upsample(low_res, target_size, interp_method, nIter, lambda)`，实现 IBP（Iterative Back Projection，迭代反投影）超分辨率重建。以插值结果为初值，反复执行“下采样 → 残差 → 上采样修正”，使重建图的下采样版本逼近低分辨率输入，从而恢复高频细节。
- **使用方法**：作为函数调用，例如 `HR = ibp_upsample(lena4s, size(I), 'bicubic', 60, 0.6);`。通常由 `experiment2.m` 自动调用，也可单独使用。
- **依赖**：无（纯 MATLAB）。
