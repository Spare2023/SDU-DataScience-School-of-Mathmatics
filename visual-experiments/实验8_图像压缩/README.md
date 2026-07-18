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


# 实验8：图像压缩与数字水印

> 📌 本文件夹仅含源码 `.m`。运行所需输入图片与第三方工具箱（DIPUM）不在此目录，需从原项目根目录 `../dipum_toolbox_2.0.2` 获取并保证 MATLAB 路径包含它们。

## 文件清单

### `experiment8.m`
- **作用**：图像压缩与数字水印。① JPEG（`im2jpeg`/`jpeg2im`）压缩重建；② JPEG2000（`im2jpeg2k`/`jpeg2k2im`）压缩重建；③ 视频帧显示（`montage`）；④ 频域数字水印（DCT/DFT/SVD 三种嵌入与提取）。支持 `DEMO_MODE` 步进演示。
- **使用方法**：直接运行。需要 **DIPUM 工具箱**（`dipum_toolbox_2.0.2/`，提供 `im2jpeg`/`im2jpeg2k` 等函数）已加入 MATLAB 路径；需同目录提供 `lena.png`、`house.png`、`watermark.bmp`（及可选的 `gsalesman` 视频用于任务3）。
- **依赖**：Image Processing Toolbox；**DIPUM 工具箱（第三方）**。
