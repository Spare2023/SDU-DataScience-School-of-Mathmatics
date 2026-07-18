% ======================================================================
% Copyright (c) 2026 Spare2023
% Repository: https://github.com/Spare2023/SDU-DataScience-School-of-Mathematics
% Licensed under the BSD 3-Clause License
%
% 【使用限制说明】
% 1. 本代码仅用于本校课程学习参考，禁止无修改直接复制提交课程大作业；
% 2. 禁止未经许可将本代码用于商业项目、对外发表、冒充个人原创成果；
% 3. 若修改后使用，需保留完整版权声明，不得删除本段注释；
% 4. 因违规使用代码产生的抄袭、处分等全部后果由使用者自行承担。
% ======================================================================

%% 实验10: 特征提取与表示
% 课程: 视觉与数据计算
% 重点函数: graycomatrix, graycoprops, pca, svd
clear all;
close all;
clc;
% 创建图片保存目录
fig_dir = fullfile(fileparts(mfilename('fullpath')), 'figs');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
fprintf('========================================\n');
fprintf('       实验10: 特征提取与表示\n');
fprintf('========================================\n\n');

%% 读入lena图像
img_path = 'lena.png';
img = imread(img_path);
if size(img, 3) == 3
    img_gray = rgb2gray(img);
else
    img_gray = img;
end
img_gray = im2double(img_gray);
fprintf('使用图像: %s\n', img_path);
fprintf('图像大小: %d x %d\n\n', size(img_gray, 1), size(img_gray, 2));

%% 任务1: 纹理度量计算
fprintf('【任务1】纹理度量计算\n');
fprintf('----------------------------------------\n');
% 定义三个不同区域的坐标
regions = struct();
regions(1).name = '平坦区域';
regions(1).row = 400;
regions(1).col = 300;
regions(1).size = 31;
regions(2).name = '边缘区域';
regions(2).row = 100;
regions(2).col = 400;
regions(2).size = 31;
regions(3).name = '纹理区域';
regions(3).row = 300;
regions(3).col = 100;
regions(3).size = 31;
% 存储结果
results = struct();
figure('Name', '任务1: 纹理区域选择', 'NumberTitle', 'off', 'Position', [50, 50, 1200, 400]);
imshow(img_gray, []);
hold on;
for i = 1:length(regions)
    r = regions(i);
    % 提取区域
    block = img_gray(r.row:r.row+r.size-1, r.col:r.col+r.size-1);
    % 绘制矩形
    rectangle('Position', [r.col, r.row, r.size, r.size], ...
              'EdgeColor', 'r', 'LineWidth', 2);
    text(r.col, r.row-5, r.name, 'Color', 'r', 'FontSize', 12);
    % 计算纹理度量
    % 1. 均值
    mean_val = mean2(block);
    % 2. 标准差
    std_val = std2(block);
    % 3. 平滑度
    smoothness = 1 - 1/(1 + std_val^2);
    % 4. 三阶矩（偏度）
    skewness_val = skewness(block(:));
    % 5. 一致性（均匀性）
    L = 256;
    block_uint8 = im2uint8(block);
    p = imhist(block_uint8) / numel(block);
    uniformity = sum(p.^2);
    % 6. 熵
    p_nonzero = p(p > 0);
    entropy_val = -sum(p_nonzero .* log2(p_nonzero));
    % 存储结果
    results(i).name = r.name;
    results(i).mean = mean_val;
    results(i).std = std_val;
    results(i).smoothness = smoothness;
    results(i).skewness = skewness_val;
    results(i).uniformity = uniformity;
    results(i).entropy = entropy_val;
    % GLCM特征
    glcm = graycomatrix(im2uint8(block), 'NumLevels', 8, 'Offset', [0 1]);
    stats = graycoprops(glcm, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
    results(i).contrast = stats.Contrast;
    results(i).correlation = stats.Correlation;
    results(i).energy = stats.Energy;
    results(i).homogeneity = stats.Homogeneity;
end
hold off;
title('三个纹理区域位置');
saveas(gcf, fullfile(fig_dir, 'exp10_fig1.png'));
fprintf('图片已保存: exp10_fig1.png\n');
% 显示结果表格
fprintf('\n纹理度量结果:\n');
fprintf('%-12s %8s %8s %10s %10s %10s %8s\n', ...
        '区域', '均值', '标准差', '平滑度', '偏度', '一致性', '熵');
fprintf('--------------------------------------------------------------------------------\n');
for i = 1:length(results)
    fprintf('%-12s %8.4f %8.4f %10.4f %10.4f %10.4f %8.4f\n', ...
            results(i).name, results(i).mean, results(i).std, ...
            results(i).smoothness, results(i).skewness, ...
            results(i).uniformity, results(i).entropy);
end
fprintf('\nGLCM特征:\n');
fprintf('%-12s %10s %12s %8s %12s\n', ...
        '区域', '对比度', '相关性', '能量', '同质性');
fprintf('----------------------------------------------------------------\n');
for i = 1:length(results)
    fprintf('%-12s %10.4f %12.4f %8.4f %12.4f\n', ...
            results(i).name, results(i).contrast, results(i).correlation, ...
            results(i).energy, results(i).homogeneity);
end

%% 任务2: 主分量变换
fprintf('\n【任务2】主分量变换\n');
fprintf('----------------------------------------\n');
% 以图像的列为向量进行PCA
% 每列是一个样本，每行是一个特征
X = img_gray;  % M x N 矩阵
% 计算均值并中心化
mean_col = mean(X, 1);
X_centered = X - mean_col;
% 使用SVD进行PCA（避免协方差矩阵特征分解的数值问题）
% U*S*V' = X_centered，V的列就是PCA主分量方向（按方差降序）
[~, S, V] = svd(X_centered, 'econ');
eigenvalues = diag(S).^2 / (size(X, 1) - 1);  % 各主分量对应的方差
% 计算主分量图像
% 第k个主分量 = X_centered * V(:, k)
pc_images = cell(3, 1);
pc_indices = [1, 100, 500];
for i = 1:length(pc_indices)
    k = pc_indices(i);
    if k <= size(V, 2)
        pc = X_centered * V(:, k);
        pc_images{i} = reshape(pc, size(X, 1), 1);
        % 扩展到图像大小以便显示
        pc_images{i} = repmat(pc_images{i}, 1, size(X, 2));
    end
end
% 显示结果
figure('Name', '任务2: 主分量图像', 'NumberTitle', 'off', 'Position', [50, 50, 1200, 900]);
subplot(2, 2, 1);
imshow(img_gray, []);
title('原图像');
for i = 1:length(pc_indices)
    subplot(2, 2, i+1);
    imshow(pc_images{i}, []);
    title(sprintf('第%d个主分量', pc_indices(i)));
end
fprintf('主分量分析完成\n');
fprintf('  第1个主分量: 包含最大方差，代表主要结构\n');
fprintf('  第100个主分量: 中等频率信息\n');
fprintf('  第500个主分量: 高频细节和噪声\n');
saveas(gcf, fullfile(fig_dir, 'exp10_fig2.png'));
fprintf('图片已保存: exp10_fig2.png\n');

%% 任务3: 奇异值分解与奇异值分布
fprintf('\n【任务3】奇异值分解与奇异值分布\n');
fprintf('----------------------------------------\n');
figure('Name', '任务3: 奇异值分布', 'NumberTitle', 'off', 'Position', [50, 50, 1400, 400]);
for i = 1:length(regions)
    r = regions(i);
    % 提取区域
    block = img_gray(r.row:r.row+r.size-1, r.col:r.col+r.size-1);
    % SVD分解
    [U, S, V_svd] = svd(block, 'econ');
    singular_values = diag(S);
    % 绘制奇异值分布
    subplot(1, 3, i);
    plot(singular_values, 'b-', 'LineWidth', 2);
    xlabel('奇异值序号');
    ylabel('奇异值');
    title({r.name, '奇异值分布'});
    grid on;
    fprintf('\n%s 奇异值统计:\n', r.name);
    fprintf('  最大奇异值: %.4f\n', max(singular_values));
    fprintf('  奇异值总和: %.4f\n', sum(singular_values));
    fprintf('  前5个奇异值占比: %.2f%%\n', sum(singular_values(1:5))/sum(singular_values)*100);
end
fprintf('\n奇异值分析:\n');
fprintf('  平坦区域: 奇异值快速衰减，能量集中在少数几个值\n');
fprintf('  边缘区域: 奇异值衰减较慢，包含更多方向信息\n');
fprintf('  纹理区域: 奇异值分布较均匀，包含丰富的高频信息\n');
saveas(gcf, fullfile(fig_dir, 'exp10_fig3.png'));
fprintf('图片已保存: exp10_fig3.png\n');

%% 任务4: 奇异值阈值收缩去噪 — 从基础SVD到WNNM
fprintf('\n【任务4】奇异值阈值收缩去噪 — SVD → WNNM 增强\n');
fprintf('----------------------------------------\n');
% 添加高斯噪声
sigma_noise = 30;
rng(0);
img_noisy = img_gray + (sigma_noise/255) * randn(size(img_gray));
img_noisy = max(0, min(1, img_noisy));
psnr_noisy = psnr(img_noisy, img_gray);
ssim_noisy = ssim(img_noisy, img_gray);

%% 4.1 基础方法: 分块SVD + 统一软阈值
fprintf('\n[4.1] 分块SVD + 统一软阈值:\n');
block_size = 16;
[M, N] = size(img_gray);
img_svd = zeros(size(img_gray));
threshold_factor = 1.5;
for i = 1:block_size:M
    for j = 1:block_size:N
        i_end = min(i+block_size-1, M);
        j_end = min(j+block_size-1, N);
        block = img_noisy(i:i_end, j:j_end);
        [U, S, V_svd] = svd(block, 'econ');
        singular_values = diag(S);
        threshold = threshold_factor * (sigma_noise/255) * sqrt(max(size(block)));
        singular_values_denoised = max(singular_values - threshold, 0);
        S_denoised = diag(singular_values_denoised);
        block_denoised = U(:, 1:size(S_denoised, 1)) * S_denoised * V_svd(:, 1:size(S_denoised, 1))';
        img_svd(i:i_end, j:j_end) = block_denoised;
    end
end
img_svd = max(0, min(1, img_svd));
psnr_svd = psnr(img_svd, img_gray);
ssim_svd = ssim(img_svd, img_gray);
fprintf('  PSNR = %.4f dB, SSIM = %.4f\n', psnr_svd, ssim_svd);

%% 4.2 增强方法: WNNM (加权核范数最小化)
%  WNNM 在 SVD 阈值收缩的基础上引入两项关键改进:
%    ① 非局部相似块匹配 — 利用图像的自相似性，将相似块组成低秩矩阵
%    ② 加权自适应阈值 — 大奇异值(结构)少收缩，小奇异值(噪声)多收缩
fprintf('\n[4.2] WNNM 加权核范数最小化 (SVD增强):\n');
[img_wnnm, psnr_hist] = WNNM_denoising(img_noisy, img_gray, sigma_noise/255, ...
    struct('iter_max', 12, 'K', 70, 'patch_size', 7, 'step', 4));
psnr_wnnm = psnr(img_wnnm, img_gray);
ssim_wnnm = ssim(img_wnnm, img_gray);
fprintf('  PSNR = %.4f dB, SSIM = %.4f\n', psnr_wnnm, ssim_wnnm);
fprintf('\n  ┌─────────────────────────────────────────────────────────────┐\n');
fprintf('  │  方法                      PSNR        SSIM      提升      │\n');
fprintf('  ├─────────────────────────────────────────────────────────────┤\n');
fprintf('  │  噪声图像                  %.4f    %.4f     —        │\n', psnr_noisy, ssim_noisy);
fprintf('  │  ① 分块SVD + 软阈值       %.4f    %.4f    +%.2f dB  │\n', psnr_svd, ssim_svd, psnr_svd-psnr_noisy);
fprintf('  │  ② WNNM (SVD增强)         %.4f    %.4f    +%.2f dB  │\n', psnr_wnnm, ssim_wnnm, psnr_wnnm-psnr_noisy);
fprintf('  └─────────────────────────────────────────────────────────────┘\n');
% 显示结果
figure('Name', '任务4: SVD → WNNM 去噪对比', 'NumberTitle', 'off', 'Position', [50, 50, 1400, 700]);
subplot(2, 3, 1);
imshow(img_gray, []);
title('原图像');
subplot(2, 3, 2);
imshow(img_noisy, []);
title({sprintf('噪声图像 (σ=%d)', sigma_noise), sprintf('PSNR=%.4f, SSIM=%.4f', psnr_noisy, ssim_noisy)});
subplot(2, 3, 3);
imshow(img_svd, []);
title({sprintf('① 分块SVD + 软阈值'), sprintf('PSNR=%.4f, SSIM=%.4f', psnr_svd, ssim_svd)});
subplot(2, 3, 4);
imshow(img_wnnm, []);
title({sprintf('② WNNM (SVD增强)'), sprintf('PSNR=%.4f, SSIM=%.4f', psnr_wnnm, ssim_wnnm)});
subplot(2, 3, 5);
imshow(img_noisy - img_wnnm, []);
title('WNNM残差图');
subplot(2, 3, 6);
plot(1:length(psnr_hist), psnr_hist, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 4);
xlabel('迭代次数'); ylabel('PSNR (dB)');
title('WNNM 迭代收敛曲线');
grid on;
saveas(gcf, fullfile(fig_dir, 'exp10_fig4.png'));
fprintf('图片已保存: exp10_fig4.png\n');

%% 结果分析
fprintf('\n【分析】\n');
fprintf('1. 纹理度量: 不同区域的统计特征有明显差异，可用于纹理分类\n');
fprintf('2. GLCM特征: 对比度、相关性等特征能有效描述纹理的空间关系\n');
fprintf('3. PCA: 主分量变换将图像信息按重要性排序，可用于降维和压缩\n');
fprintf('4. SVD: 奇异值分解揭示了图像的能量分布，适合去噪和压缩\n');
fprintf('5. SVD去噪对比:\n');
fprintf('   ① 分块SVD+软阈值: 各块独立处理, 统一阈值, 存在块效应\n');
fprintf('   ② WNNM(SVD增强): 非局部相似块匹配+加权阈值, PSNR显著提升\n');
fprintf('\n========================================\n');
fprintf('        实验10 完成!\n');
fprintf('========================================\n');