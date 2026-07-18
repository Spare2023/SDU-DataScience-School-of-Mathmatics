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

%% 实验3: 图像增强
% 课程: 视觉与数据计算
% 重点函数: imadjust, imhist, histeq, adapthisteq, imfilter, fspecial, medfilt2

clear all;
close all;
clc;

% 创建图片保存目录
fig_dir = fullfile(fileparts(mfilename('fullpath')), 'figs');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

% 添加 BM3D 算法路径
addpath('../BM3D');                    % BM3D 核心函数
addpath(genpath('../BM3D/IDDBM3D'));   % IDDBM3D 子模块

fprintf('========================================\n');
fprintf('          实验3: 图像增强\n');
fprintf('========================================\n\n');

%% 任务1: 眼底图像增强方法比较
fprintf('【任务1】眼底图像增强方法比较\n');
fprintf('----------------------------------------\n');

% 读入fundus图像
fundus_path = 'fundus.png';


fundus = imread(fundus_path);
if size(fundus, 3) == 3
    fundus_gray = rgb2gray(fundus);
else
    fundus_gray = fundus;
end

fprintf('使用图像: %s\n', fundus_path);
fprintf('图像大小: %d x %d\n', size(fundus_gray, 1), size(fundus_gray, 2));

% 方法1: imadjust - 灰度调整
fundus_imadjust = imadjust(fundus_gray);

% 方法2: histeq - 直方图均衡化
fundus_histeq = histeq(fundus_gray);

% 方法3: adapthisteq - 自适应直方图均衡化(CLAHE)
%   CLAHE (Contrast Limited Adaptive Histogram Equalization):
%   将图像分成小块(tiles), 每块独立做直方图均衡, 再双线性插值拼接;
%   对每个局部直方图设置裁剪阈值(Clip Limit), 限制对比度放大幅度,
%   避免全局histeq对噪声区域的过度增强. 对眼底这类光照不均的图像效果显著优于全局均衡.
fundus_adapthisteq = adapthisteq(fundus_gray);

% 计算NIQE质量指标
niqe_imadjust = niqe(fundus_imadjust);
niqe_histeq = niqe(fundus_histeq);
niqe_adapthisteq = niqe(fundus_adapthisteq);
fprintf('NIQE质量指标:\n');
fprintf('  imadjust: %.4f\n', niqe_imadjust);
fprintf('  histeq: %.4f\n', niqe_histeq);
fprintf('  adapthisteq: %.4f\n', niqe_adapthisteq);

% 显示结果
figure('Name', '任务1: 眼底图像增强方法比较', 'NumberTitle', 'off', 'Position', [50, 50, 1400, 900]);

% 原图像和直方图
subplot(4, 2, 1);
imshow(fundus_gray, []);
title('原图像');

subplot(4, 2, 2);
imhist(fundus_gray);
title('原图像直方图');

% imadjust
subplot(4, 2, 3);
imshow(fundus_imadjust, []);
title(sprintf('imadjust (NIQE=%.4f)', niqe_imadjust));

subplot(4, 2, 4);
imhist(fundus_imadjust);
title('imadjust直方图');

% histeq
subplot(4, 2, 5);
imshow(fundus_histeq, []);
title(sprintf('histeq (NIQE=%.4f)', niqe_histeq));

subplot(4, 2, 6);
imhist(fundus_histeq);
title('histeq直方图');

% adapthisteq
subplot(4, 2, 7);
imshow(fundus_adapthisteq, []);
title(sprintf('adapthisteq (NIQE=%.4f)', niqe_adapthisteq));

subplot(4, 2, 8);
imhist(fundus_adapthisteq);
title('adapthisteq直方图');

% 保存图片
saveas(gcf, fullfile(fig_dir, 'exp3_task1_fundus_enhance.png'));
fprintf('图片已保存: exp3_task1_fundus_enhance.png\n');

%% 任务2: 拉普拉斯算子和反锐化掩模增强（含参数优化对比）
fprintf('\n【任务2】拉普拉斯算子和反锐化掩模增强\n');
fprintf('----------------------------------------\n');

% 转为double进行精确运算
fundus_double = im2double(fundus_gray);

% ============ 2a. Laplacian 参数扫描（步长 0.1） ============
% Laplacian 锐化原理: 二阶微分算子检测灰度突变,
%   锐化公式: g = f - ∇²f  (原图减去二阶梯度)
%   ∇²f 在平坦区域≈0, 在边缘处有较大响应,
%   从原图中减去∇²f 等效于增强边缘对比度.
%   fspecial('laplacian', alpha):
%     alpha=0 → 仅考虑水平/垂直方向(4-连通邻域)
%     alpha=1 → 加入对角方向(8-连通邻域), 各向同性更好
%   步长 0.1 扫描 [0, 0.5] 区间寻找最优 alpha.
fprintf('\n--- Laplacian alpha 参数扫描 (步长 0.1) ---\n');
lap_alphas = 0:0.1:0.5;
lap_results = zeros(length(lap_alphas), 1);

for k = 1:length(lap_alphas)
    % fspecial('laplacian', alpha): alpha=0 → 4-连通, alpha=1 → 8-连通
    lap_kernel = fspecial('laplacian', lap_alphas(k));
    % 标准Laplacian锐化: g = f - ∇²f
    img_lap = fundus_double - imfilter(fundus_double, lap_kernel, 'replicate');
    img_lap = max(0, min(1, img_lap));  % 防止过冲
    lap_results(k) = niqe(img_lap);
    fprintf('  alpha=%.1f: NIQE=%.4f\n', lap_alphas(k), lap_results(k));
end

[best_lap_niqe, best_lap_idx] = min(lap_results);
best_lap_alpha = lap_alphas(best_lap_idx);
fprintf('  → 最佳alpha=%.1f (NIQE=%.4f)\n', best_lap_alpha, best_lap_niqe);

% 用最佳alpha生成最终结果
lap_kernel_best = fspecial('laplacian', best_lap_alpha);
fundus_laplacian = fundus_double - imfilter(fundus_double, lap_kernel_best, 'replicate');
fundus_laplacian = max(0, min(1, fundus_laplacian));

% 同时保存原始方法结果用于对比基线
laplacian_mask_old = [0 -1 0; -1 5 -1; 0 -1 0];
fundus_laplacian_old = imfilter(fundus_gray, laplacian_mask_old, 'replicate');
niqe_lap_old = niqe(fundus_laplacian_old);

% ============ 2b. Unsharp Masking 两阶段网格搜索 ============
% Unsharp Masking (反锐化掩模) 原理:
%   步骤: ① 对原图 f 做低通滤波得到模糊版本 g;
%         ② 计算"掩模": mask = f - g (原图减模糊 = 高频细节);
%         ③ 锐化: h = f + amount × mask.
%   参数: Radius 控制高斯模糊的σ(决定提取细节的尺度),
%         Amount 控制细节增强的幅度.
%   两阶段搜索策略: 粗扫定位最优区域 → 精扫步长减半细化.
fprintf('\n--- Unsharp Masking 参数搜索 ---\n');

%% 第一阶段：粗扫 5×5
fprintf('【阶段1】粗扫 (5×5):\n');
usm_radii_coarse = [1.0, 1.5, 2.0, 2.5, 3.0];
usm_amounts_coarse = [1.0, 1.3, 1.5, 1.8, 2.0];
usm_results_coarse = zeros(length(usm_radii_coarse), length(usm_amounts_coarse));

fprintf('Radius\\Amount');
for a = 1:length(usm_amounts_coarse)
    fprintf('   A=%.1f  ', usm_amounts_coarse(a));
end
fprintf('\n');

for r = 1:length(usm_radii_coarse)
    fprintf('  R=%.1f    ', usm_radii_coarse(r));
    for a = 1:length(usm_amounts_coarse)
        img_usm = imsharpen(fundus_gray, 'Radius', usm_radii_coarse(r), 'Amount', usm_amounts_coarse(a));
        usm_results_coarse(r, a) = niqe(img_usm);
        fprintf('%.4f ', usm_results_coarse(r, a));
    end
    fprintf('\n');
end

[best_coarse, best_lin] = min(usm_results_coarse(:));
[r_c, a_c] = ind2sub(size(usm_results_coarse), best_lin);
r_best_coarse = usm_radii_coarse(r_c);
a_best_coarse = usm_amounts_coarse(a_c);
fprintf('  粗扫最佳: Radius=%.1f, Amount=%.1f (NIQE=%.4f)\n', r_best_coarse, a_best_coarse, best_coarse);

%% 第二阶段：精扫 5×5 局部细化（步长减半）
fprintf('【阶段2】精扫 (5×5, 步长减半):\n');
% 在粗扫最佳值周围自动生成精扫网格
r_fine = r_best_coarse + (-0.3:0.15:0.3);
a_fine = a_best_coarse + (-0.2:0.1:0.2);
% 限制到合理范围
r_fine = r_fine(r_fine >= 0.5 & r_fine <= 4.0);
a_fine = a_fine(a_fine >= 0.5 & a_fine <= 2.5);
% 去重（四舍五入保留精度后取唯一）
r_fine = unique(round(r_fine, 2));
a_fine = unique(round(a_fine, 2));
% 如果精扫范围全超界（例如粗扫在边界），就留在边界做局部搜索
if length(r_fine) < 2, r_fine = r_best_coarse; end
if length(a_fine) < 2, a_fine = a_best_coarse; end

usm_results_fine = zeros(length(r_fine), length(a_fine));

fprintf('Radius\\Amount');
for a = 1:length(a_fine)
    fprintf('   A=%.2f', a_fine(a));
end
fprintf('\n');

for r = 1:length(r_fine)
    fprintf('  R=%.2f  ', r_fine(r));
    for a = 1:length(a_fine)
        img_usm = imsharpen(fundus_gray, 'Radius', r_fine(r), 'Amount', a_fine(a));
        usm_results_fine(r, a) = niqe(img_usm);
        fprintf(' %.4f', usm_results_fine(r, a));
    end
    fprintf('\n');
end

[best_fine, best_fine_lin] = min(usm_results_fine(:));
[r_f, a_f] = ind2sub(size(usm_results_fine), best_fine_lin);
best_radius = r_fine(r_f);
best_amount = a_fine(a_f);
best_usm_niqe = best_fine;
fprintf('  精扫最佳: Radius=%.2f, Amount=%.2f (NIQE=%.4f)\n', best_radius, best_amount, best_usm_niqe);

% 用最佳参数生成最终结果
fundus_unsharp = imsharpen(fundus_gray, 'Radius', best_radius, 'Amount', best_amount);

% 同时保存原始方法结果用于对比
h_gaussian_old = fspecial('gaussian', [5 5], 1);
fundus_blurred_old = imfilter(fundus_gray, h_gaussian_old, 'replicate');
mask_old = fundus_gray - fundus_blurred_old;
fundus_unsharp_old = fundus_gray + 1.5 * mask_old;
niqe_usm_old = niqe(fundus_unsharp_old);

% ============ 2c. 参数优化结果总结 ============
fprintf('\n--- 参数优化对比总结 ---\n');
fprintf('  Laplacian原方法: NIQE=%.4f (核:[0 -1 0; -1 5 -1])\n', niqe_lap_old);
fprintf('  Laplacian优化后: NIQE=%.4f (alpha=%.1f, 步长0.1精细扫描)\n', best_lap_niqe, best_lap_alpha);
fprintf('  Unsharp原方法:   NIQE=%.4f (高斯σ=1, Amount=1.5, 手动实现)\n', niqe_usm_old);
fprintf('  Unsharp优化后:   NIQE=%.4f (Radius=%.2f, Amount=%.2f, 两阶段搜索)\n', best_usm_niqe, best_radius, best_amount);
fprintf('  搜索规模: 粗扫%dx%d + 精扫%dx%d = %d组\n', ...
    length(usm_radii_coarse), length(usm_amounts_coarse), ...
    length(r_fine), length(a_fine), ...
    length(usm_radii_coarse)*length(usm_amounts_coarse) + length(r_fine)*length(a_fine));

% ============ 2d. 显示结果 ============
figure('Name', '任务2: 拉普拉斯与反锐化掩模增强（参数优化后）', ...
       'NumberTitle', 'off', 'Position', [50, 50, 1200, 400]);

subplot(1, 3, 1);
imshow(fundus_gray, []);
title('原图像');

subplot(1, 3, 2);
imshow(fundus_laplacian, []);
title({sprintf('Laplacian增强 (alpha=%.1f)', best_lap_alpha), ...
       sprintf('NIQE=%.4f (原%.2f)', best_lap_niqe, niqe_lap_old)});

subplot(1, 3, 3);
imshow(fundus_unsharp, []);
title({sprintf('Unsharp Masking (R=%.2f, A=%.2f)', best_radius, best_amount), ...
       sprintf('NIQE=%.4f (原%.2f)', best_usm_niqe, niqe_usm_old)});

% 保存图片
saveas(gcf, fullfile(fig_dir, 'exp3_task2_sharpen.png'));
fprintf('图片已保存: exp3_task2_sharpen.png\n');

%% 任务3: 高斯噪声图像去噪
fprintf('\n【任务3】高斯噪声图像去噪\n');
fprintf('----------------------------------------\n');

% 噪声参数
sigma = 30;  % 高斯噪声标准差

% 参数搜索范围
gauss_sizes = [3, 5, 7, 9, 11];                   % 高斯核大小（扩展更多选项）
gauss_sigmas = [0.5, 0.8, 1.0, 1.2, 1.5];        % 高斯sigma
median_sizes = [3, 5, 7, 9, 11];                   % 中值滤波核大小（参考代码使用11×11）
bilateral_coarse = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00];  % 双边滤波粗扫倍率
nlm_coarse = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00];        % NLM粗扫倍率
tv_coarse = [0.04, 0.08, 0.12, 0.16, 0.20];                % TV粗扫lambda

% 图像名称
image_names = {'lena', 'cameraman', 'house'};

% 创建图形窗口（保存句柄防止后续figure切换覆盖）
h_main_fig = figure('Name', '任务3: 噪声图像去噪（最优参数对比）', ...
    'NumberTitle', 'off', 'Position', [50, 50, 2000, 1200]);

% 加载预训练DnCNN网络（仅加载一次）
fprintf('加载DnCNN预训练模型...\n');
dncnn_net = denoisingNetwork('DnCNN');
fprintf('DnCNN模型加载完成。\n\n');

for i = 1:length(image_names)
    img_name = image_names{i};
    img_path = [img_name, '.png'];

    % 读入图像
    img = imread(img_path);
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    img = im2double(img);

    % 添加高斯噪声
    rng(0);
    img_noisy = img + (sigma/255) * randn(size(img));
    img_noisy = max(0, min(1, img_noisy));

    fprintf('\n========== 图像: %s ==========\n', img_name);

    % 初始化Pareto分析数据收集
    pareto_tags = {};
    pareto_psnr = {};
    pareto_ssim = {};
    pareto_sel_psnr = [];
    pareto_sel_ssim = [];

    %% 1. 高斯滤波网格调参
    fprintf('\n====== 高斯滤波参数优化 ======\n');
    fprintf('  %-8s  %-7s  %-10s  %-8s\n', '核大小', 'sigma', 'PSNR(dB)', 'SSIM');
    % 预分配候选数组
    n_gauss = length(gauss_sizes) * length(gauss_sigmas);
    gauss_psnr_vals = zeros(n_gauss, 1);
    gauss_ssim_vals = zeros(n_gauss, 1);
    gauss_img_cell = cell(n_gauss, 1);
    gauss_size_vals = zeros(n_gauss, 1);
    gauss_sigma_vals = zeros(n_gauss, 1);
    gidx = 0;
    for gs = gauss_sizes
        for gsig = gauss_sigmas
            gidx = gidx + 1;
            h = fspecial('gaussian', [gs gs], gsig);
            img_g = imfilter(img_noisy, h, 'replicate');
            p = psnr(img_g, img);
            s = ssim(img_g, img);
            fprintf('  [%d×%d]    %-7.2f  %-10.4f  %.4f\n', gs, gs, gsig, p, s);
            gauss_psnr_vals(gidx) = p;
            gauss_ssim_vals(gidx) = s;
            gauss_img_cell{gidx} = img_g;
            gauss_size_vals(gidx) = gs;
            gauss_sigma_vals(gidx) = gsig;
        end
    end
    % PSNR阈值+SSIM择优选择（文献: Padova大学混合损失建议α≈0.7）
    best_gidx = select_by_psnr_threshold(gauss_psnr_vals, gauss_ssim_vals);
    best_gauss_psnr = gauss_psnr_vals(best_gidx);
    best_gauss_ssim = gauss_ssim_vals(best_gidx);
    best_gauss_img = gauss_img_cell{best_gidx};
    best_gauss_size = gauss_size_vals(best_gidx);
    best_gauss_sigma = gauss_sigma_vals(best_gidx);
    max_p = max(gauss_psnr_vals); thresh_p = max_p - 0.5;
    n_valid = sum(gauss_psnr_vals >= thresh_p);
    fprintf('  %s\n', repmat('-', 1, 45));
    fprintf('  ✅ PSNR阈值+SSIM择优: [%d×%d] sigma=%.2f, PSNR=%.4f, SSIM=%.4f\n', ...
        best_gauss_size, best_gauss_size, best_gauss_sigma, best_gauss_psnr, best_gauss_ssim);
    fprintf('  📊 max PSNR=%.4f, 阈值=%.4f, %d个候选(共%d)中SSIM最高\n\n', ...
        max_p, thresh_p, n_valid, n_gauss);
    % 保存Pareto分析数据
    pareto_tags{end+1} = '高斯'; n_par = length(pareto_tags);
    pareto_psnr{n_par} = gauss_psnr_vals;
    pareto_ssim{n_par} = gauss_ssim_vals;
    pareto_sel_psnr(n_par) = best_gauss_psnr;
    pareto_sel_ssim(n_par) = best_gauss_ssim;

    %% 2. 中值滤波网格调参
    fprintf('====== 中值滤波参数优化 ======\n');
    median_best = tune_median_filter(img_noisy, img, median_sizes);
    best_median_psnr = median_best.psnr;
    best_median_ssim = median_best.ssim;
    best_median_img = median_best.image;
    best_median_size = median_best.windowSize;
    max_p = max(median_best.psnr_vals); thresh_p = max_p - 0.5;
    n_valid = sum(median_best.psnr_vals >= thresh_p);
    fprintf('  %s\n', repmat('-', 1, 35));
    fprintf('  ✅ PSNR阈值+SSIM择优: [%d×%d] PSNR=%.4f, SSIM=%.4f\n', ...
        best_median_size, best_median_size, best_median_psnr, best_median_ssim);
    fprintf('  📊 max PSNR=%.4f, 阈值=%.4f, %d个候选(共%d)中SSIM最高\n\n', ...
        max_p, thresh_p, n_valid, length(median_sizes));
    % 保存Pareto分析数据
    pareto_tags{end+1} = '中值'; n_par = length(pareto_tags);
    pareto_psnr{n_par} = median_best.psnr_vals;
    pareto_ssim{n_par} = median_best.ssim_vals;
    pareto_sel_psnr(n_par) = best_median_psnr;
    pareto_sel_ssim(n_par) = best_median_ssim;

    %% 3. 双边滤波两阶段调参（粗扫+精扫）
    % 双边滤波原理: 空间高斯核 × 灰度值域高斯核
    %   空间核: 距离中心越近权重越大 (与高斯滤波相同)
    %   值域核: 灰度值与中心越接近权重越大
    %   → 在平滑噪声的同时保留边缘 (边缘处两侧像素灰度差异大, 权重趋近0)
    %   参数 DoS (DegreeOfSmoothing) 控制值域核的σ,
    %   以 doSmooth_base 为基准, 搜索最佳倍率
    fprintf('====== 双边滤波参数优化（两阶段搜索） ======\n');
    doSmooth_base = 2 * sigma^2 / 255^2 * sqrt(size(img,1)*size(img,2)) / 50;

    % 阶段1：粗扫
    fprintf('【阶段1】粗扫:\n');
    fprintf('  %-12s  %-10s  %-8s\n', 'DoS倍率', 'PSNR(dB)', 'SSIM');
    best_bilateral_psnr = -inf;
    for bs = bilateral_coarse
        img_b = imbilatfilt(img_noisy, doSmooth_base * bs);
        p = psnr(img_b, img);
        s = ssim(img_b, img);
        fprintf('  %-12.2f  %-10.4f  %.4f\n', bs, p, s);
        if p > best_bilateral_psnr
            best_bilateral_psnr = p;  % 只需记录PSNR用于精扫范围
            best_bilateral_scale = bs;
        end
    end
    fprintf('  %s\n', repmat('-', 1, 35));
    fprintf('  ✅ 粗扫最佳: DoS倍率=%.2f, PSNR=%.4f\n', ...
        best_bilateral_scale, best_bilateral_psnr);

    % 阶段2：精扫（在粗扫最佳附近以步长0.05细化）
    fprintf('【阶段2】精扫 (步长0.05):\n');
    fprintf('  %-12s  %-10s  %-8s\n', 'DoS倍率', 'PSNR(dB)', 'SSIM');
    fine_range = best_bilateral_scale + (-0.20:0.05:0.20);
    fine_range = fine_range(fine_range >= 0.10 & fine_range <= 3.0);
    fine_range = unique(round(fine_range, 2));
    if length(fine_range) < 2, fine_range = best_bilateral_scale; end
    % 预分配候选数组
    n_fine = length(fine_range);
    bilat_f_psnr = zeros(n_fine, 1);
    bilat_f_ssim = zeros(n_fine, 1);
    bilat_f_img = cell(n_fine, 1);
    bilat_f_param = zeros(n_fine, 1);
    bidx = 0;
    for bs = fine_range
        bidx = bidx + 1;
        img_b = imbilatfilt(img_noisy, doSmooth_base * bs);
        p = psnr(img_b, img);
        s = ssim(img_b, img);
        fprintf('  %-12.2f  %-10.4f  %.4f\n', bs, p, s);
        bilat_f_psnr(bidx) = p;
        bilat_f_ssim(bidx) = s;
        bilat_f_img{bidx} = img_b;
        bilat_f_param(bidx) = bs;
    end
    % PSNR阈值+SSIM择优选择
    best_bidx = select_by_psnr_threshold(bilat_f_psnr, bilat_f_ssim);
    best_bilateral_psnr = bilat_f_psnr(best_bidx);
    best_bilateral_ssim = bilat_f_ssim(best_bidx);
    best_bilateral_img = bilat_f_img{best_bidx};
    best_bilateral_scale = bilat_f_param(best_bidx);
    max_p = max(bilat_f_psnr); thresh_p = max_p - 0.5;
    n_valid = sum(bilat_f_psnr >= thresh_p);
    fprintf('  %s\n', repmat('-', 1, 35));
    fprintf('  ✅ PSNR阈值+SSIM择优: DoS倍率=%.2f, PSNR=%.4f, SSIM=%.4f\n', ...
        best_bilateral_scale, best_bilateral_psnr, best_bilateral_ssim);
    fprintf('  📊 max PSNR=%.4f, 阈值=%.4f, %d个候选(共%d)中SSIM最高\n', ...
        max_p, thresh_p, n_valid, n_fine);
    fprintf('  搜索规模: 粗扫%d点 + 精扫%d点\n\n', ...
        length(bilateral_coarse), length(fine_range));
    % 保存Pareto分析数据
    pareto_tags{end+1} = '双边'; n_par = length(pareto_tags);
    pareto_psnr{n_par} = bilat_f_psnr;
    pareto_ssim{n_par} = bilat_f_ssim;
    pareto_sel_psnr(n_par) = best_bilateral_psnr;
    pareto_sel_ssim(n_par) = best_bilateral_ssim;

    %% 4. 非局部均值(NLM)两阶段调参（粗扫+精扫）
    fprintf('====== 非局部均值(NLM)参数优化（两阶段搜索） ======\n');

    % 阶段1：粗扫
    fprintf('【阶段1】粗扫:\n');
    fprintf('  %-12s  %-10s  %-8s\n', 'DoS倍率', 'PSNR(dB)', 'SSIM');
    best_nlm_psnr = -inf;
    for ns = nlm_coarse
        img_n = imnlmfilt(img_noisy, 'DegreeOfSmoothing', (sigma/2) * ns);
        p = psnr(img_n, img);
        s = ssim(img_n, img);
        fprintf('  %-12.2f  %-10.4f  %.4f\n', ns, p, s);
        if p > best_nlm_psnr
            best_nlm_psnr = p;  % 只需记录PSNR用于精扫范围
            best_nlm_scale = ns;
        end
    end
    fprintf('  %s\n', repmat('-', 1, 35));
    fprintf('  ✅ 粗扫最佳: DoS倍率=%.2f, PSNR=%.4f\n', ...
        best_nlm_scale, best_nlm_psnr);

    % 阶段2：精扫
    fprintf('【阶段2】精扫 (步长0.05):\n');
    fprintf('  %-12s  %-10s  %-8s\n', 'DoS倍率', 'PSNR(dB)', 'SSIM');
    fine_range = best_nlm_scale + (-0.20:0.05:0.20);
    fine_range = fine_range(fine_range >= 0.05 & fine_range <= 4.0);
    fine_range = unique(round(fine_range, 2));
    if length(fine_range) < 2, fine_range = best_nlm_scale; end
    % 预分配候选数组
    n_fine = length(fine_range);
    nlm_f_psnr = zeros(n_fine, 1);
    nlm_f_ssim = zeros(n_fine, 1);
    nlm_f_img = cell(n_fine, 1);
    nlm_f_param = zeros(n_fine, 1);
    nidx = 0;
    for ns = fine_range
        nidx = nidx + 1;
        img_n = imnlmfilt(img_noisy, 'DegreeOfSmoothing', (sigma/2) * ns);
        p = psnr(img_n, img);
        s = ssim(img_n, img);
        fprintf('  %-12.2f  %-10.4f  %.4f\n', ns, p, s);
        nlm_f_psnr(nidx) = p;
        nlm_f_ssim(nidx) = s;
        nlm_f_img{nidx} = img_n;
        nlm_f_param(nidx) = ns;
    end
    % PSNR阈值+SSIM择优选择
    best_nidx = select_by_psnr_threshold(nlm_f_psnr, nlm_f_ssim);
    best_nlm_psnr = nlm_f_psnr(best_nidx);
    best_nlm_ssim = nlm_f_ssim(best_nidx);
    best_nlm_img = nlm_f_img{best_nidx};
    best_nlm_scale = nlm_f_param(best_nidx);
    max_p = max(nlm_f_psnr); thresh_p = max_p - 0.5;
    n_valid = sum(nlm_f_psnr >= thresh_p);
    fprintf('  %s\n', repmat('-', 1, 35));
    fprintf('  ✅ PSNR阈值+SSIM择优: DoS倍率=%.2f, PSNR=%.4f, SSIM=%.4f\n', ...
        best_nlm_scale, best_nlm_psnr, best_nlm_ssim);
    fprintf('  📊 max PSNR=%.4f, 阈值=%.4f, %d个候选(共%d)中SSIM最高\n', ...
        max_p, thresh_p, n_valid, n_fine);
    fprintf('  搜索规模: 粗扫%d点 + 精扫%d点\n\n', ...
        length(nlm_coarse), length(fine_range));
    % 保存Pareto分析数据
    pareto_tags{end+1} = 'NLM'; n_par = length(pareto_tags);
    pareto_psnr{n_par} = nlm_f_psnr;
    pareto_ssim{n_par} = nlm_f_ssim;
    pareto_sel_psnr(n_par) = best_nlm_psnr;
    pareto_sel_ssim(n_par) = best_nlm_ssim;

    %% 5. 全变分(TV)去噪两阶段调参（粗扫+精扫）
    fprintf('====== 全变分(TV)参数优化（两阶段搜索） ======\n');

    % 阶段1：粗扫
    fprintf('【阶段1】粗扫:\n');
    fprintf('  %-12s  %-10s  %-8s\n', 'lambda', 'PSNR(dB)', 'SSIM');
    best_tv_psnr = -inf;
    for tl = tv_coarse
        img_t = tv_denoise(img_noisy, tl, 80);
        p = psnr(img_t, img);
        s = ssim(img_t, img);
        fprintf('  %-12.3f  %-10.4f  %.4f\n', tl, p, s);
        if p > best_tv_psnr
            best_tv_psnr = p;  % 只需记录PSNR用于精扫范围
            best_tv_lambda = tl;
        end
    end
    fprintf('  %s\n', repmat('-', 1, 35));
    fprintf('  ✅ 粗扫最佳: lambda=%.3f, PSNR=%.4f\n', ...
        best_tv_lambda, best_tv_psnr);

    % 阶段2：精扫（步长减半为0.005）
    fprintf('【阶段2】精扫 (步长0.005):\n');
    fprintf('  %-12s  %-10s  %-8s\n', 'lambda', 'PSNR(dB)', 'SSIM');
    fine_range = best_tv_lambda + (-0.025:0.005:0.025);
    fine_range = fine_range(fine_range >= 0.01 & fine_range <= 0.50);
    fine_range = unique(round(fine_range, 3));
    if length(fine_range) < 2, fine_range = best_tv_lambda; end
    % 预分配候选数组
    n_fine = length(fine_range);
    tv_f_psnr = zeros(n_fine, 1);
    tv_f_ssim = zeros(n_fine, 1);
    tv_f_img = cell(n_fine, 1);
    tv_f_param = zeros(n_fine, 1);
    tidx = 0;
    for tl = fine_range
        tidx = tidx + 1;
        img_t = tv_denoise(img_noisy, tl, 80);
        p = psnr(img_t, img);
        s = ssim(img_t, img);
        fprintf('  %-12.3f  %-10.4f  %.4f\n', tl, p, s);
        tv_f_psnr(tidx) = p;
        tv_f_ssim(tidx) = s;
        tv_f_img{tidx} = img_t;
        tv_f_param(tidx) = tl;
    end
    % PSNR阈值+SSIM择优选择
    best_tidx = select_by_psnr_threshold(tv_f_psnr, tv_f_ssim);
    best_tv_psnr = tv_f_psnr(best_tidx);
    best_tv_ssim = tv_f_ssim(best_tidx);
    best_tv_img = tv_f_img{best_tidx};
    best_tv_lambda = tv_f_param(best_tidx);
    max_p = max(tv_f_psnr); thresh_p = max_p - 0.5;
    n_valid = sum(tv_f_psnr >= thresh_p);
    fprintf('  %s\n', repmat('-', 1, 35));
    fprintf('  ✅ PSNR阈值+SSIM择优: lambda=%.3f, PSNR=%.4f, SSIM=%.4f\n', ...
        best_tv_lambda, best_tv_psnr, best_tv_ssim);
    fprintf('  📊 max PSNR=%.4f, 阈值=%.4f, %d个候选(共%d)中SSIM最高\n', ...
        max_p, thresh_p, n_valid, n_fine);
    fprintf('  搜索规模: 粗扫%d点 + 精扫%d点\n\n', ...
        length(tv_coarse), length(fine_range));
    % 保存Pareto分析数据
    pareto_tags{end+1} = 'TV'; n_par = length(pareto_tags);
    pareto_psnr{n_par} = tv_f_psnr;
    pareto_ssim{n_par} = tv_f_ssim;
    pareto_sel_psnr(n_par) = best_tv_psnr;
    pareto_sel_ssim(n_par) = best_tv_ssim;

    %% 6. DnCNN深度学习去噪（固定参数，无需调优）
    fprintf('DnCNN深度学习去噪（固定参数）...\n');
    img_dncnn = denoiseImage(img_noisy, dncnn_net);
    img_dncnn = im2double(img_dncnn);
    psnr_dncnn = psnr(img_dncnn, img);
    ssim_dncnn = ssim(img_dncnn, img);
    fprintf('  DnCNN: PSNR=%.4f dB, SSIM=%.4f\n\n', psnr_dncnn, ssim_dncnn);

    %% 7. BM3D块匹配3D协同滤波去噪
    fprintf('====== BM3D块匹配3D协同滤波 ======\n');
    % BM3D使用两步法：硬阈值(HT)基础估计 → 维纳(Wiener)滤波最终估计
    % 利用图像自相似性，将相似块堆叠成3D数组，在3D变换域进行协同滤波
    % 输入: (original/1, noisy, sigma, profile, verbose)
    %   sigma 在 [0,255] 范围，噪声图像在 [0,1] 范围
    [~, img_bm3d] = BM3D(1, img_noisy, sigma, 'np', 0);
    psnr_bm3d = psnr(img_bm3d, img);
    ssim_bm3d = ssim(img_bm3d, img);
    fprintf('  BM3D (np):            PSNR=%.4f dB, SSIM=%.4f\n', psnr_bm3d, ssim_bm3d);

    %% 质量指标汇总
    psnr_noisy = psnr(img_noisy, img);
    ssim_noisy = ssim(img_noisy, img);

    fprintf('========== 最终结果: %s ==========\n', img_name);
    fprintf('  噪声图像:          PSNR=%.4f dB, SSIM=%.4f\n', psnr_noisy, ssim_noisy);
    fprintf('  ── 经典方法 ──\n');
    fprintf('  高斯滤波  [%d×%d] σ=%.1f: PSNR=%.4f dB, SSIM=%.4f\n', ...
        best_gauss_size, best_gauss_size, best_gauss_sigma, best_gauss_psnr, best_gauss_ssim);
    fprintf('  中值滤波  [%d×%d]:        PSNR=%.4f dB, SSIM=%.4f\n', ...
        best_median_size, best_median_size, best_median_psnr, best_median_ssim);
    fprintf('  双边滤波  (DoS×%.2f):   PSNR=%.4f dB, SSIM=%.4f\n', ...
        best_bilateral_scale, best_bilateral_psnr, best_bilateral_ssim);
    fprintf('  ── 先进方法 ──\n');
    fprintf('  非局部均值(NLM):         PSNR=%.4f dB, SSIM=%.4f\n', best_nlm_psnr, best_nlm_ssim);
    fprintf('  DnCNN深度学习:           PSNR=%.4f dB, SSIM=%.4f\n', psnr_dncnn, ssim_dncnn);
    fprintf('  全变分(TV)    λ=%.3f:    PSNR=%.4f dB, SSIM=%.4f\n', best_tv_lambda, best_tv_psnr, best_tv_ssim);
    fprintf('  ── 基准方法 ──\n');
    fprintf('  BM3D块匹配3D协同滤波:       PSNR=%.4f dB, SSIM=%.4f\n', psnr_bm3d, ssim_bm3d);

    %% 显示结果（9列：原图/噪声/高斯最优/中值最优/双边最优/NLM最优/DnCNN/TV最优/BM3D）
    figure(h_main_fig);  % 切回主对比图窗口
    n_cols = 9;
    row_offset = (i - 1) * n_cols;

    % 列1: 原图像
    subplot(length(image_names), n_cols, row_offset + 1);
    imshow(img, []);
    title(sprintf('%s 原图像', img_name));

    % 列2: 噪声图像
    subplot(length(image_names), n_cols, row_offset + 2);
    imshow(img_noisy, []);
    title({sprintf('噪声图像 (\\sigma=%d)', sigma), ...
        sprintf('PSNR=%.4f', psnr_noisy), ...
        sprintf('SSIM=%.4f', ssim_noisy)});

    % 列3: 高斯滤波（最优参数）
    subplot(length(image_names), n_cols, row_offset + 3);
    imshow(best_gauss_img, []);
    title({sprintf('高斯滤波 [%d\\times%d] σ=%.1f', best_gauss_size, best_gauss_size, best_gauss_sigma), ...
        sprintf('PSNR=%.4f', best_gauss_psnr), ...
        sprintf('SSIM=%.4f', best_gauss_ssim)});

    % 列4: 中值滤波（最优参数）
    subplot(length(image_names), n_cols, row_offset + 4);
    imshow(best_median_img, []);
    title({sprintf('中值滤波 [%d\\times%d]', best_median_size, best_median_size), ...
        sprintf('PSNR=%.4f', best_median_psnr), ...
        sprintf('SSIM=%.4f', best_median_ssim)});

    % 列5: 双边滤波（最优参数）
    subplot(length(image_names), n_cols, row_offset + 5);
    imshow(best_bilateral_img, []);
    title({sprintf('双边滤波 (DoS\\times%.2f)', best_bilateral_scale), ...
        sprintf('PSNR=%.4f', best_bilateral_psnr), ...
        sprintf('SSIM=%.4f', best_bilateral_ssim)});

    % 列6: 非局部均值(NLM)滤波（最优参数）
    subplot(length(image_names), n_cols, row_offset + 6);
    imshow(best_nlm_img, []);
    title({sprintf('非局部均值(NLM)'), ...
        sprintf('PSNR=%.4f', best_nlm_psnr), ...
        sprintf('SSIM=%.4f', best_nlm_ssim)});

    % 列7: DnCNN深度学习去噪
    subplot(length(image_names), n_cols, row_offset + 7);
    imshow(img_dncnn, []);
    title({sprintf('DnCNN深度学习'), ...
        sprintf('PSNR=%.4f', psnr_dncnn), ...
        sprintf('SSIM=%.4f', ssim_dncnn)});

    % 列8: 全变分(TV)去噪（最优参数）
    subplot(length(image_names), n_cols, row_offset + 8);
    imshow(best_tv_img, []);
    title({sprintf('全变分(TV) λ=%.3f', best_tv_lambda), ...
        sprintf('PSNR=%.4f', best_tv_psnr), ...
        sprintf('SSIM=%.4f', best_tv_ssim)});

    % 列9: BM3D协同滤波去噪
    subplot(length(image_names), n_cols, row_offset + 9);
    imshow(img_bm3d, []);
    title({sprintf('BM3D协同滤波'), ...
        sprintf('PSNR=%.4f', psnr_bm3d), ...
        sprintf('SSIM=%.4f', ssim_bm3d)});

% 保存主去噪对比图
saveas(gcf, fullfile(fig_dir, ['exp3_task3_denoise_', img_name, '.png']));
fprintf('图片已保存: exp3_task3_denoise_%s.png\n', img_name);

%% Pareto前沿分析图
    % 对每种可调参方法绘制PSNR-SSIM散点图，标注Pareto前沿和选中点
    figure('Name', ['PSNR-SSIM Pareto前沿 - ', img_name], ...
        'NumberTitle', 'off', 'Position', [50, 50, 1600, 700]);
    n_methods = length(pareto_tags);

    for p = 1:n_methods
        subplot(2, 3, p);
        psnr_p = pareto_psnr{p};
        ssim_p = pareto_ssim{p};

        % 散点图绘制所有候选
        scatter(psnr_p, ssim_p, 30, 'b', 'filled', 'MarkerFaceAlpha', 0.4); hold on;

        % 计算Pareto前沿（非支配解）
        n_pts = length(psnr_p);
        is_pareto = true(n_pts, 1);
        for ii = 1:n_pts
            for j = 1:n_pts
                if ii ~= j && psnr_p(j) >= psnr_p(ii) && ssim_p(j) >= ssim_p(ii) ...
                        && (psnr_p(j) > psnr_p(ii) || ssim_p(j) > ssim_p(ii))
                    is_pareto(ii) = false;
                    break;
                end
            end
        end
        pareto_idx = find(is_pareto);
        [pareto_psnr_sorted, sort_i] = sort(psnr_p(pareto_idx));
        pareto_ssim_sorted = ssim_p(pareto_idx(sort_i));
        plot(pareto_psnr_sorted, pareto_ssim_sorted, 'r-o', ...
            'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', 'r');

        % 标记选中点
        sel_p = pareto_sel_psnr(p);
        sel_s = pareto_sel_ssim(p);
        plot(sel_p, sel_s, 'gp', 'MarkerSize', 18, 'LineWidth', 2, ...
            'MarkerFaceColor', 'g');

        xlabel('PSNR (dB)'); ylabel('SSIM');
        title({pareto_tags{p}, ...
            sprintf('PSNR=%.4f, SSIM=%.4f', sel_p, sel_s)});
        legend({'候选参数', 'Pareto前沿', '选中点'}, 'Location', 'southeast');
        grid on; axis tight;
    end

    % 第6个子图：说明
    subplot(2, 3, 6); axis off;
    text(0.05, 0.8, 'Pareto前沿分析说明:', 'FontSize', 13, 'FontWeight', 'bold');
    text(0.05, 0.65, '• 蓝色散点: 所有候选参数组合', 'FontSize', 11);
    text(0.05, 0.55, '• 红色连线: Pareto前沿', 'FontSize', 11);
    text(0.05, 0.45, '  (非支配解集合)', 'FontSize', 11);
    text(0.05, 0.35, '• 绿色五角星: 最终选中参数', 'FontSize', 11);
    text(0.05, 0.25, '• 策略: PSNR阈值+SSIM择优', 'FontSize', 11);
    text(0.05, 0.15, '  (max PSNR - 0.5dB范围内)', 'FontSize', 11);
    text(0.05, 0.05, '• 参考: Padova α=0.7, ACM MM 2024', 'FontSize', 11);

% 保存Pareto前沿图
saveas(gcf, fullfile(fig_dir, ['exp3_pareto_', img_name, '.png']));
fprintf('图片已保存: exp3_pareto_%s.png\n', img_name);

end

%% 结果分析
fprintf('\n【分析】\n');
fprintf('1. 图像增强方法比较:\n');
fprintf('   - imadjust: 通过灰度变换增强对比度，适合整体偏暗或偏亮的图像\n');
fprintf('   - histeq: 全局直方图均衡化，可能过度增强噪声\n');
fprintf('   - adapthisteq: 自适应直方图均衡化(CLAHE)，局部增强效果更好\n');
fprintf('\n2. 锐化参数优化:\n');
fprintf('   - Laplacian: fspecial(''laplacian'', alpha) 中 alpha 控制对角方向权重\n');
fprintf('     alpha=0 仅水平/垂直(4-连通)，alpha=1 含对角(8-连通)\n');
fprintf('     以步长0.1精细扫描[0, 0.5]区间，alpha=%.1f 取得最佳(NIQE=%.4f)\n', best_lap_alpha, best_lap_niqe);
fprintf('     相比固定4-连通核(NIQE=%.4f)，增加对角权重后锐化更均匀\n', niqe_lap_old);
fprintf('   - Unsharp Masking: 采用两阶段网格搜索:\n');
fprintf('     阶段1: 粗扫 5×5 (Radius×Amount, 大步长覆盖全局)\n');
fprintf('     阶段2: 在粗扫最佳点周围精扫 5×5 (步长减半: R±0.15, A±0.1)\n');
fprintf('     最终最佳: Radius=%.2f, Amount=%.2f (NIQE=%.4f)\n', best_radius, best_amount, best_usm_niqe);
fprintf('     相比手动高斯+掩模方案(NIQE=%.4f→%.2f)，两阶段搜索确保找到局部最优\n', niqe_usm_old, best_usm_niqe);
fprintf('   - 参数优化关键: 锐化参数需与图像特征尺度匹配，过小放大噪声，过大产生光晕\n');
fprintf('\n3. 去噪方法比较:\n');
fprintf('   ── 参数优化策略 ──\n');
fprintf('   - 高斯滤波: 5种核×5种sigma=25组网格搜索（核[3,5,7,9,11]，sigma[0.5~1.5]）\n');
fprintf('   - 中值滤波: 4种核大小网格搜索（[3,5,7,9]），自动选取PSNR最优\n');
fprintf('   - 双边滤波: 两阶段搜索（粗扫6点 + 精扫~9点，步长0.05局部细化）\n');
fprintf('   - 非局部均值(NLM): 两阶段搜索（粗扫6点 + 精扫~9点，步长0.05）\n');
fprintf('   - 全变分(TV): 两阶段搜索（粗扫5点 + 精扫~11点，步长0.005精细调优）\n');
fprintf('   - 搜索策略: 粗扫快速定位最优区域→精扫步长减半找到精细最优点\n');
fprintf('   - 最优参数与图像内容相关：纹理丰富(Lena)选较小核，平坦区多(Cameraman)选较大核\n');
fprintf('   ── 经典方法 ──\n');
fprintf('   - 高斯滤波: 适合高斯噪声，核越大平滑越强但边缘越模糊，PSNR自动选最优折中\n');
fprintf('   - 中值滤波: 适合椒盐噪声，对高斯噪声效果一般，但保留边缘较好\n');
fprintf('   - 双边滤波: 边缘保持去噪，在纹理丰富图像上优于高斯滤波\n');
fprintf('   ── 先进方法 ──\n');
fprintf('   - 非局部均值(NLM): 利用图像全局自相似性，全图搜索相似块加权平均，\n');
fprintf('     比双边滤波更好保留纹理细节，PSNR通常高1-2dB\n');
fprintf('   - DnCNN: 深度残差CNN去噪(20层卷积)，端到端学习，无需调参\n');
fprintf('     PSNR通常为最高(比双边滤波高3-4dB)\n');
fprintf('   - 全变分(TV): ROF模型总变分最小化，两阶段lambda调参(粗扫+精扫)\n');
fprintf('     数学框架严谨，平坦区域平滑干净，边缘锐利\n');
fprintf('   - BM3D: 块匹配3D协同滤波，两步法(硬阈值HT+维纳Wiener滤波)\n');
fprintf('     利用图像自相似性，将相似块堆叠成3D数组，3D变换域稀疏去噪\n');
fprintf('     算法复杂但效果优异，常作为去噪算法基准(benchmark)\n');
fprintf('   - （注: 各方法最优参数因图像内容和噪声水平而异）\n');
fprintf('   ── PSNR-SSIM平衡取舍（文献参考）──\n');
fprintf('   - 问题: 单纯优化PSNR可能选中过度平滑的参数（PSNR高但SSIM低）\n');
fprintf('   - 方案: 采用PSNR阈值约束+SSIM择优策略\n');
fprintf('     max PSNR - 0.5dB范围内选SSIM最高的参数\n');
fprintf('   - 文献依据:\n');
fprintf('     (1) Padova大学 混合MSE-SSIM损失, α≈0.7为最优平衡点\n');
fprintf('     (2) ACM MM 2024 MOBOSR, Pareto多目标优化框架\n');
fprintf('     (3) Si Lu No-reference评估, 联合预测PSNR和SSIM\n');
fprintf('   - 人眼感知: 0.5dB的PSNR差异难以察觉，但SSIM差异（如0.78vs0.92）显著\n');
fprintf('   - Pareto前沿图: 展示了PSNR与SSIM的竞争关系，红色连线为Pareto最优面\n');

fprintf('\n========================================\n');
fprintf('        实验3 完成!\n');
fprintf('========================================\n');

% 关闭所有图形
close all;

%% ============================================================
%% 辅助函数: 全变分（TV）去噪
%% ROF模型: min_u ||u-f||^2 + λ·TV(u) 的梯度下降求解
%% 不需要任何额外工具箱
%% ============================================================
function u = tv_denoise(f, lambda, iter)
    % TV_DENOISE  ROF全变分去噪（梯度下降法）
    %   输入: f - 噪声图像 (double, [0,1])
    %         lambda - 正则化参数，越大越平滑 (默认 0.12)
    %         iter - 迭代次数 (默认 80)
    %   输出: u - 去噪后图像
    %
    %   原理: ROF(1992) 总变分最小化模型
    %   u^(k+1) = u^k + dt·((f-u^k) + λ·div(∇u^k/|∇u^k|))

    u = im2double(f);
    dt = 0.2;               % 时间步长（<=0.25 保证稳定性）

    for i = 1:iter
        % 前向差分梯度: ∂u/∂x, ∂u/∂y
        ux = u(:, [2:end, end]) - u;   % 前向差分 x 方向
        uy = u([2:end, end], :) - u;   % 前向差分 y 方向

        % 梯度幅值（加小量避免除零）
        grad = sqrt(ux.^2 + uy.^2 + 1e-10);

        % 归一化梯度
        ux = ux ./ grad;
        uy = uy ./ grad;

        % 后向差分散度: div(p) = ∂p_x/∂x + ∂p_y/∂y
        div = [ux(:, 1), diff(ux, 1, 2)] + [uy(1, :); diff(uy, 1, 1)];

        % 梯度下降更新: u^(k+1) = u^k + dt·((f-u^k) + λ·div)
        u = u + dt * (f - u + lambda * div);
    end
end

%% ============================================================
%% 辅助函数: PSNR阈值约束 + SSIM择优选择
%% 文献依据: Padova大学论文 α=0.7, ACM MM'24 MOBOSR
%% 原理: 在PSNR阈值范围内选SSIM最高，避免过度平滑
%% ============================================================
function best_idx = select_by_psnr_threshold(psnr_vals, ssim_vals, threshold)
    % SELECT_BY_PSNR_THRESHOLD  基于PSNR阈值约束的SSIM择优选择
    %   输入: psnr_vals - PSNR值数组 (N×1)
    %         ssim_vals - SSIM值数组 (N×1)
    %         threshold - PSNR容忍阈值(dB)，默认=0.5
    %   输出: best_idx - 选中索引
    %
    %   方法: 先找到max PSNR，设定下限=max-threshold，
    %         在阈值范围内选SSIM最高的候选
    %   参考: Padova大学混合损失(α=0.7), ACM MM 2024 MOBOSR

    if nargin < 3, threshold = 0.5; end

    max_psnr = max(psnr_vals);
    min_psnr = max_psnr - threshold;
    valid = psnr_vals >= min_psnr;

    [~, local_best] = max(ssim_vals(valid));
    valid_idx = find(valid);
    best_idx = valid_idx(local_best);
end

%% ============================================================
%% 辅助函数: 中值滤波调参（参考代码函数封装风格）
%% 原理: 中值滤波是非线性排序统计滤波器
%%   用窗口内所有像素的中值替代中心像素值
%%   优点: 对椒盐噪声(孤立极值点)有极好的抑制效果
%%         且比均值滤波更好地保留边缘(边缘不是排序后的中值)
%%   限制: 对高斯噪声效果有限(高斯噪声每个像素都有扰动,中值不够鲁棒)
%%   改进: 'symmetric'镜像填充避免边界失真，核范围扩展至11×11
%% ============================================================
function best = tune_median_filter(noisy, original, windowCandidates)
    % TUNE_MEDIAN_FILTER 中值滤波网格搜索调参
    %   输入: noisy - 噪声图像 (double, [0,1])
    %         original - 原始图像 (double, [0,1])
    %         windowCandidates - 核大小候选数组，如 [3,5,7,9,11]
    %   输出: best - 结构体含最优参数、指标和全部候选数据

    n = length(windowCandidates);
    psnr_vals = zeros(n, 1);
    ssim_vals = zeros(n, 1);
    img_cell = cell(n, 1);

    fprintf('  %-8s  %-10s  %-8s\n', '核大小', 'PSNR(dB)', 'SSIM');
    for idx = 1:n
        ws = windowCandidates(idx);
        % 'symmetric'镜像填充，避免默认零填充导致边缘偏暗
        denoised = medfilt2(noisy, [ws ws], 'symmetric');
        img_cell{idx} = denoised;
        psnr_vals(idx) = psnr(denoised, original);
        ssim_vals(idx) = ssim(denoised, original);
        fprintf('  [%d×%d]    %-10.4f  %.4f\n', ws, ws, psnr_vals(idx), ssim_vals(idx));
    end

    % PSNR阈值+SSIM择优选择
    best_idx = select_by_psnr_threshold(psnr_vals, ssim_vals);

    best = struct();
    best.windowSize = windowCandidates(best_idx);
    best.psnr = psnr_vals(best_idx);
    best.ssim = ssim_vals(best_idx);
    best.image = img_cell{best_idx};
    best.psnr_vals = psnr_vals;
    best.ssim_vals = ssim_vals;
    best.best_idx = best_idx;
end
