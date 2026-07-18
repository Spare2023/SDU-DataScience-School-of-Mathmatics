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

%% 实验7: 彩色图像处理（课堂演示优化版）
% 课程: 视觉与数据计算 | 重点函数: rgb2ycbcr, rgb2lab, rgb2hsv
%
% ★ 课堂演示特性:
%   1. 分步演示: 每步暂停等待按键，方便课堂讲解
%   2. 函数复用: 提取7个辅助函数，消除重复代码
%   3. 模块独立: 每个任务可按 Ctrl+Enter 单独运行
%   4. 调优分离: 参数扫描移至 experiment7_tuning.m
%
% 【操作说明】
%   DEMO_MODE = 1 → 步进演示模式（每步暂停，适合课堂）
%   DEMO_MODE = 0 → 全自动运行（快速出结果）

clear all; close all; clc;

%% ======================================================================
%  0. 初始化配置
%  ======================================================================

DEMO_MODE  = 0;       % 1=步进演示(默认), 0=全自动

% ── 文件路径 ──
p_house  = 'image_House256rgb.png';
p_lena   = 'image_lena512rgb.png';
p_yellow = 'yellowlily.jpg';
p_gt     = 'yellowlily_gt.png';

% ── 参数 ──
sigma_noise = 30;       % 高斯噪声标准差

% ── 去噪方法配置 ──
denoise_names  = {'medfilt2', 'gaussian', 'bilateral', 'wiener', 'nlm', 'diffusion'};
denoise_labels = {'medfilt2[5×5]', 'imgaussfilt σ=1.5', 'imbilatfilt', ...
                  'wiener2[5×5]', 'imnlmfilt', 'imdiffusefilt'};
denoise_count  = length(denoise_names);

color_spaces = {'YCbCr', 'Lab', 'HSV'};
n_cs         = length(color_spaces);

addpath('../BM3D');

% ── 尝试加载调优参数 ──
if exist('exp7_best_params.mat', 'file')
    load('exp7_best_params.mat', 'best_params');
    use_tuned = true;
    fprintf('  [已加载] 调优参数 ← exp7_best_params.mat\n');
else
    use_tuned = false;
    fprintf('  [默认值] 无调优参数(先运行 experiment7_tuning.m 可生成)\n');
end
fprintf('\n');

fprintf('========================================\n');
fprintf('  实验7: 彩色图像处理（课堂演示版）\n');
fprintf('  DEMO_MODE = %d  （1=步进, 0=自动）\n', DEMO_MODE);
fprintf('  参数调优请运行: experiment7_tuning.m\n');
fprintf('========================================\n\n');

%% ======================================================================
%  任务1: RGB → YCbCr 颜色空间转换
%  ======================================================================

fprintf('【任务1】RGB → YCbCr 颜色空间转换\n');
fprintf('  目的: 观察亮度(Y)和色度(Cb/Cr)通道的物理意义\n');
fprintf('----------------------------------------\n');

house = imread(p_house);
house_ycbcr = rgb2ycbcr(house);

Y  = house_ycbcr(:,:,1);    % 亮度 (Luma)
Cb = house_ycbcr(:,:,2);    % 蓝色色度
Cr = house_ycbcr(:,:,3);    % 红色色度

fprintf('  Y  (亮度) : [16, 235]，反映光强变化 — 包含主要结构\n');
fprintf('  Cb (蓝色度): [16, 240] 128=无色度\n');
fprintf('  Cr (红色度): [16, 240] 128=无色度\n');
fprintf('  → 亮度通道保留图像主要结构，色度通道细节较少\n\n');

f1 = demo_figure('任务1: YCbCr颜色空间', 1400, 400);
subplot(1, 4, 1); imshow(house);      title('原RGB图像');
subplot(1, 4, 2); imshow(Y,  []);     title('Y (亮度通道)');
subplot(1, 4, 3); imshow(Cb, []);     title('Cb (蓝色色度)');
subplot(1, 4, 4); imshow(Cr, []);     title('Cr (红色色度)');
sgtitle('任务1: RGB → YCbCr 颜色空间分解', 'FontSize', 14);

demo_pause(DEMO_MODE, '任务1完成，进入任务2 ■');

%% ======================================================================
%  任务2: RGB → Lab + 高斯噪声 → 噪声通道分析
%  ======================================================================

fprintf('\n【任务2】Lab颜色空间转换与噪声分析\n');
fprintf('  目的: 观察高斯噪声在L/a/b通道的分布差异\n');
fprintf('----------------------------------------\n');

lena = im2double(imread(p_lena));
lena_lab_clean = rgb2lab(lena);            % 干净Lab（用于算噪声标准差）

% 添加高斯噪声 (σ=30)
rng(0);
lena_noisy = lena + (sigma_noise / 255) * randn(size(lena));
lena_noisy = max(0, min(1, lena_noisy));

% 噪声图像转Lab
lena_lab  = rgb2lab(lena_noisy);
L_ch = lena_lab(:,:,1);    % 亮度 (0~100)
a_ch = lena_lab(:,:,2);    % 绿-红轴
b_ch = lena_lab(:,:,3);    % 蓝-黄轴

% 计算各通道噪声标准差（噪声Lab - 干净Lab）
L_clean  = lena_lab_clean(:,:,1);
a_clean  = lena_lab_clean(:,:,2);
b_clean  = lena_lab_clean(:,:,3);
noise_std_L = std(L_ch(:) - L_clean(:));
noise_std_a = std(a_ch(:) - a_clean(:));
noise_std_b = std(b_ch(:) - b_clean(:));

fprintf('  各通道噪声标准差:\n');
fprintf('    L(亮度) : %.4f  ← 噪声最集中\n', noise_std_L);
fprintf('    a(绿-红): %.4f\n', noise_std_a);
fprintf('    b(蓝-黄): %.4f\n', noise_std_b);
fprintf('  → 亮度通道集中大部分噪声，色度通道噪声较少\n');
fprintf('  → 这就是"间接去噪"(仅处理亮度)的理论基础\n\n');

f2 = demo_figure('任务2: Lab噪声分析', 1200, 900);
subplot(2, 2, 1); show_result(lena_noisy, sprintf('噪声RGB (σ=%d)', sigma_noise));
% L[0,100], a[-128,127], b[-128,127] → imshow 需要映射到显示范围
subplot(2, 2, 2); imshow(L_ch, [0 100]);          title(sprintf('L(亮度)  std=%.4f', noise_std_L));
subplot(2, 2, 3); imshow(a_ch, [-128 127]);        title(sprintf('a(绿-红) std=%.4f', noise_std_a));
subplot(2, 2, 4); imshow(b_ch, [-128 127]);        title(sprintf('b(蓝-黄) std=%.4f', noise_std_b));
sgtitle('任务2: Lab各通道噪声分布', 'FontSize', 14);

demo_pause(DEMO_MODE, '任务2完成，进入核心任务3（去噪方法比较）■');

%% ======================================================================
%  任务3: 彩色图像去噪 — 直接方法 vs 间接方法
%  ======================================================================

fprintf('\n【任务3】彩色图像去噪: 直接 vs 间接\n');
fprintf('  直接: R/G/B 三通道分别去噪 → 合成\n');
fprintf('  间接: RGB → 颜色空间 → 仅亮度去噪 → 反变换\n');
fprintf('  噪声图像: PSNR=%.4f dB, SSIM=%.4f\n', ...
    psnr(lena_noisy, lena), ssim(lena_noisy, lena));
fprintf('----------------------------------------\n');

%% ─── 3.1 直接方法 ───

fprintf('\n--- 3.1 直接方法（三通道独立去噪）---\n');
fprintf('  思路: 对R/G/B三个通道独立应用去噪算法\n');
fprintf('  优点: 简单直接\n');
fprintf('  缺点: 各通道噪声特性不同，可能引入颜色失真\n\n');

direct_results = cell(denoise_count, 1);
direct_psnr    = zeros(denoise_count, 1);
direct_ssim    = zeros(denoise_count, 1);

for k = 1:denoise_count
    denoised = denoise_rgb_channels(lena_noisy, denoise_names{k});
    direct_psnr(k) = psnr(denoised, lena);
    direct_ssim(k) = ssim(denoised, lena);
    direct_results{k} = denoised;
    fprintf('  [%-2d/%d] %-20s  PSNR=%.4f dB, SSIM=%.4f\n', ...
        k, denoise_count, denoise_labels{k}, direct_psnr(k), direct_ssim(k));
end

[best_d_psnr, best_d_idx] = max(direct_psnr);
fprintf('\n  ★ 直接法最优: %s  (PSNR=%.4f, SSIM=%.4f)\n', ...
    denoise_labels{best_d_idx}, best_d_psnr, direct_ssim(best_d_idx));

%% ─── 3.2 间接方法 ───

fprintf('\n--- 3.2 间接方法（亮度通道去噪）---\n');
fprintf('  思路: RGB → 颜色空间 → 仅对亮度通道去噪 → 反变换\n');
fprintf('  优点: 色度通道保持不变，颜色保真度更高\n\n');

indirect_results = cell(n_cs, denoise_count);
indirect_psnr    = zeros(n_cs, denoise_count);
indirect_ssim    = zeros(n_cs, denoise_count);

for cs = 1:n_cs
    for d = 1:denoise_count
        rgb_den = process_luminance(lena_noisy, color_spaces{cs}, denoise_names{d});
        indirect_psnr(cs, d) = psnr(rgb_den, lena);
        indirect_ssim(cs, d) = ssim(rgb_den, lena);
        indirect_results{cs, d} = rgb_den;
        fprintf('  %-6s + %-17s  PSNR=%.4f dB, SSIM=%.4f\n', ...
            color_spaces{cs}, denoise_labels{d}, indirect_psnr(cs, d), indirect_ssim(cs, d));
    end
end

%% ─── 3.3 CBM3D 协同滤波 ───

fprintf('\n--- 3.3 CBM3D 非局部协同滤波 ---\n');
fprintf('  思路: 利用图像自相似性，3D变换域协同滤波\n');
fprintf('  地位: 当前彩色图像去噪的先进方法\n\n');

[~, lena_cbm3d_opp] = CBM3D(1, lena_noisy, sigma_noise, 'np', 0, 'opp');
[~, lena_cbm3d_ycc] = CBM3D(1, lena_noisy, sigma_noise, 'np', 0, 'yCbCr');

psnr_cbm3d_opp = psnr(lena_cbm3d_opp, lena);
ssim_cbm3d_opp = ssim(lena_cbm3d_opp, lena);
psnr_cbm3d_ycc = psnr(lena_cbm3d_ycc, lena);
ssim_cbm3d_ycc = ssim(lena_cbm3d_ycc, lena);

fprintf('  CBM3D (opponent): PSNR=%.4f dB, SSIM=%.4f\n', psnr_cbm3d_opp, ssim_cbm3d_opp);
fprintf('  CBM3D (yCbCr):    PSNR=%.4f dB, SSIM=%.4f\n', psnr_cbm3d_ycc, ssim_cbm3d_ycc);

if psnr_cbm3d_opp >= psnr_cbm3d_ycc
    lena_cbm3d  = lena_cbm3d_opp;
    cbm3d_label = 'CBM3D (opponent)';
else
    lena_cbm3d  = lena_cbm3d_ycc;
    cbm3d_label = 'CBM3D (yCbCr)';
end

%% ─── 3.4 汇总最优结果 ───

% 间接法最优
[best_i_grid, best_linear] = max(indirect_psnr(:));
[best_cs_i, best_dn_i] = ind2sub([n_cs, denoise_count], best_linear);

if psnr_cbm3d_opp > best_i_grid
    best_i_psnr   = psnr_cbm3d_opp;
    best_i_ssim   = ssim_cbm3d_opp;
    best_i_name   = cbm3d_label;
    best_i_result = lena_cbm3d;
else
    best_i_psnr   = best_i_grid;
    best_i_ssim   = indirect_ssim(best_cs_i, best_dn_i);
    best_i_name   = sprintf('%s + %s', color_spaces{best_cs_i}, denoise_labels{best_dn_i});
    best_i_result = indirect_results{best_cs_i, best_dn_i};
end

% ── 汇总表格 ──
fprintf('\n========== 去噪结果汇总 ==========\n');
fprintf('┌──────────────────────────────────┬──────────┬──────────┬───────────────┐\n');
fprintf('│ %-32s │ %8s │ %8s │ %-13s │\n', '方法', 'PSNR(dB)', 'SSIM', '类型');
fprintf('├──────────────────────────────────┼──────────┼──────────┼───────────────┤\n');
fprintf('│ %-32s │ %8.4f │ %8.4f │ %-13s │\n', sprintf('噪声图像 (σ=%d)', sigma_noise), ...
    psnr(lena_noisy, lena), ssim(lena_noisy, lena), '参考');
for k = 1:denoise_count
    tag = '直接';
    if k == best_d_idx, tag = '★直接'; end
    fprintf('│ %-32s │ %8.4f │ %8.4f │ %-13s │\n', denoise_labels{k}, ...
        direct_psnr(k), direct_ssim(k), tag);
end
fprintf('├──────────────────────────────────┼──────────┼──────────┼───────────────┤\n');
for cs = 1:n_cs
    for d = 1:denoise_count
        n = sprintf('%s + %s', color_spaces{cs}, denoise_labels{d});
        tag = '间接';
        if strcmp(n, best_i_name), tag = '★间接'; end
        fprintf('│ %-32s │ %8.4f │ %8.4f │ %-13s │\n', n, ...
            indirect_psnr(cs,d), indirect_ssim(cs,d), tag);
    end
end
cbm3d_tag = '间接';
if contains(best_i_name, 'CBM3D'), cbm3d_tag = '★间接'; end
fprintf('│ %-32s │ %8.4f │ %8.4f │ %-13s │\n', ...
    cbm3d_label, psnr_cbm3d_opp, ssim_cbm3d_opp, cbm3d_tag);
fprintf('└──────────────────────────────────┴──────────┴──────────┴───────────────┘\n');

%% ─── 3.5 可视化 ───

% Figure A: 噪声 → 最优直接 → 最优间接 → CBM3D
f3a = demo_figure('任务3: 去噪结果对比', 1400, 600);

subplot(2, 3, 1);
show_result(lena_noisy, sprintf('噪声图像 (σ=%d)', sigma_noise), ...
    psnr(lena_noisy, lena), ssim(lena_noisy, lena));

subplot(2, 3, 2);
show_result(direct_results{best_d_idx}, ...
    sprintf('★直接法: %s', denoise_labels{best_d_idx}), ...
    best_d_psnr, direct_ssim(best_d_idx));

subplot(2, 3, 3);
show_result(best_i_result, sprintf('★间接法: %s', best_i_name), ...
    best_i_psnr, best_i_ssim);

subplot(2, 3, 4);
show_result(lena_cbm3d, cbm3d_label, psnr_cbm3d_opp, ssim_cbm3d_opp);

subplot(2, 3, [5, 6]); axis off;
text(0.1, 0.85, '【分析结论】', 'FontSize', 14, 'FontWeight', 'bold');
text(0.1, 0.65, sprintf('直接法: %s  PSNR=%.2f', denoise_labels{best_d_idx}, best_d_psnr), 'FontSize', 12, 'Color', 'b');
text(0.1, 0.45, sprintf('间接法: %s  PSNR=%.2f', best_i_name, best_i_psnr), 'FontSize', 12, 'Color', 'r');
text(0.1, 0.25, '间接法通过亮度/色度分离 → 颜色保真度更高', 'FontSize', 12);
text(0.1, 0.05, 'CBM3D 利用非局部自相似性 → 当前最优', 'FontSize', 12);
sgtitle('任务3: 彩色图像去噪综合对比', 'FontSize', 14);

% Figure B: 误差分布
diff_direct   = abs(double(direct_results{best_d_idx}) - double(lena));
diff_indirect = abs(double(best_i_result) - double(lena));
diff_diff     = double(diff_indirect) - double(diff_direct);

f3b = demo_figure('任务3: 误差分布', 1400, 400);
subplot(1, 3, 1); imshow(diff_direct, []);
title(sprintf('直接法误差  MAE=%.4f', mean(diff_direct(:))));
colorbar; colormap(gca, jet);

subplot(1, 3, 2); imshow(diff_indirect, []);
title(sprintf('间接法误差  MAE=%.4f', mean(diff_indirect(:))));
colorbar; colormap(gca, jet);

subplot(1, 3, 3); imshow(diff_diff, []);
title({'间接−直接误差差', '蓝=间接优  红=直接优'});
colorbar; colormap(gca, jet);
sgtitle('任务3: 误差分布对比', 'FontSize', 14);

fprintf('\n  [误差图分析] 蓝色区域=间接法误差更小，红色=直接法更优\n');
fprintf('  通常间接法在平坦区域和边缘区域都表现更好\n');

demo_pause(DEMO_MODE, '任务3核心内容完成，进入任务4（颜色分割）■');

%% ======================================================================
%  任务4: 基于颜色信息的黄色花朵分割
%  ======================================================================

fprintf('\n【任务4】基于颜色信息的黄色花朵分割\n');
fprintf('  方法: HSV阈值分割 + 形态学优化\n');
fprintf('----------------------------------------\n');

yellow = im2double(imread(p_yellow));
yellow_hsv = rgb2hsv(yellow);
H = yellow_hsv(:,:,1);    % 色调
S = yellow_hsv(:,:,2);    % 饱和度
V = yellow_hsv(:,:,3);    % 明度

%% ─── 4.1 基本分割 ───

% 阈值来源: 有调优参数 → 自动加载; 无 → 默认值
if use_tuned
    H_LOW  = best_params.H_low;
    H_HIGH = best_params.H_high;
    S_TH   = best_params.S_thresh;
    V_TH   = best_params.V_thresh;
    OPEN_R = best_params.open_r;
    CLOSE_R= best_params.close_r;
    fprintf('  [调优] 加载最优分割参数 (tuning.m 生成)\n');
else
    H_LOW  = 0.10; H_HIGH = 0.25;
    S_TH   = 0.3;  V_TH   = 0.3;
    OPEN_R = 3;    CLOSE_R= 5;
    fprintf('  [默认] 使用固定分割参数\n');
end

mask_hsv = (H > H_LOW) & (H < H_HIGH) & (S > S_TH) & (V > V_TH);

% 形态学后处理
mask_hsv = imopen(mask_hsv, strel('disk', OPEN_R));   % 去孤立噪点
mask_hsv = imclose(mask_hsv, strel('disk', CLOSE_R)); % 填缝隙
mask_hsv = imfill(mask_hsv, 'holes');                  % 填孔洞

seg_hsv = yellow .* repmat(mask_hsv, [1, 1, 3]);

fprintf('  分割参数: H∈[%.2f, %.2f], S>%.1f, V>%.1f\n', H_LOW, H_HIGH, S_TH, V_TH);
fprintf('  形态学: 开运算(disk=%d) → 闭运算(disk=%d) → 填孔洞\n', OPEN_R, CLOSE_R);

% IoU
gt_mask = load_ground_truth(p_gt, yellow);
if ~isempty(gt_mask)
    iou_hsv = compute_iou(mask_hsv, gt_mask);
    fprintf('  IoU = %.4f\n', iou_hsv);
else
    fprintf('  未找到GT标注文件，跳过IoU计算\n');
end

% 显示
f4a = demo_figure('任务4: HSV花朵分割', 1400, 400);
subplot(1, 4, 1); imshow(yellow);       title('原图像');
subplot(1, 4, 2); imshow(H, []);        title('H (色调)');
subplot(1, 4, 3); show_result(mask_hsv, '分割掩模');
subplot(1, 4, 4); show_result(seg_hsv,  '分割结果');
sgtitle('任务4: 基于HSV的黄色花朵分割', 'FontSize', 14);

demo_pause(DEMO_MODE, '基本分割完成，观察结果');

%% ─── 4.2 Lab b*辅助分割（拓展） ───

fprintf('\n--- 4.2(拓展) Lab b*辅助分割 ---\n');
fprintf('  原理: Lab b*通道中黄色区域正响应强，Otsu可自动阈值\n');

lab_y = rgb2lab(yellow);
b_star = lab_y(:,:,3);
b_level = graythresh(b_star);
mask_lab = b_star > b_level * max(b_star(:));
mask_lab = imopen(mask_lab, strel('disk', 3));
mask_lab = imclose(mask_lab, strel('disk', 5));
mask_lab = imfill(mask_lab, 'holes');

if ~isempty(gt_mask)
    iou_lab = compute_iou(mask_lab, gt_mask);

    % 融合策略
    fusion_and = mask_hsv & mask_lab;
    fusion_or  = mask_hsv | mask_lab;
    iou_and = compute_iou(fusion_and, gt_mask);
    iou_or  = compute_iou(fusion_or,  gt_mask);

    ious   = [iou_hsv, iou_lab, iou_and, iou_or];
    names  = {'HSV', 'Lab Otsu', 'HSV∩Lab', 'HSV∪Lab'};
    [~, bi] = max(ious);
    fprintf('  HSV:         IoU=%.4f\n', iou_hsv);
    fprintf('  Lab b*+Otsu: IoU=%.4f\n', iou_lab);
    fprintf('  HSV ∩ Lab:   IoU=%.4f\n', iou_and);
    fprintf('  HSV ∪ Lab:   IoU=%.4f\n', iou_or);
    fprintf('  ★ 最优: %s (IoU=%.4f)\n', names{bi}, ious(bi));

    % 显示四种策略
    f4b = demo_figure('任务4: 双空间分割对比', 1400, 500);
    segs = {mask_hsv, mask_lab, fusion_and, fusion_or};
    for i = 1:4
        subplot(2, 4, i);
        imshow(segs{i}); title(sprintf('%s  IoU=%.4f', names{i}, ious(i)));
        subplot(2, 4, i+4);
        imshow(yellow .* repmat(segs{i}, [1,1,3])); title('分割结果');
    end
    sgtitle('HSV + Lab 双空间分割策略对比', 'FontSize', 14);
end

fprintf('\n  HSV: 适合手动调参, Lab b*: Otsu自动阈值, 融合: 提升鲁棒性\n');

%% ======================================================================
%  总结
%  ======================================================================

fprintf('\n========================================\n');
fprintf('  【实验7总结】\n');
fprintf('========================================\n');
fprintf('  1. YCbCr: Y=亮度(主信息), Cb/Cr=色度\n');
fprintf('  2. Lab: L=感知均匀亮度, a/b=颜色对立轴\n');
fprintf('  3. 去噪: 间接法(亮度去噪) > 直接法(通道独立)\n');
fprintf('     ★ 最优直接: %s (PSNR=%.4f)\n', denoise_labels{best_d_idx}, best_d_psnr);
fprintf('     ★ 最优间接: %s (PSNR=%.4f)\n', best_i_name, best_i_psnr);
fprintf('  4. CBM3D利用非局部自相似性达到最优\n');
fprintf('  5. HSV色调度用于颜色分割，Lab b*可自动辅助\n');
if ~isempty(gt_mask)
    fprintf('  6. 黄色分割最优IoU = %.4f\n', max(ious));
end
fprintf('\n  参数调优请运行 → experiment7_tuning.m\n');
fprintf('========================================\n');
fprintf('        实验7 完成!\n');
fprintf('========================================\n\n');

%% ======================================================================
%  局部函数
%  ======================================================================

function h = show_result(img, str, psnr_val, ssim_val)
    % 显示图像，支持3种调用方式
    h = imshow(img);
    if nargin >= 4
        title({str, sprintf('PSNR=%.2f dB  SSIM=%.4f', psnr_val, ssim_val)});
    elseif nargin >= 2
        title(str);
    end
end

function f = demo_figure(name, w, h)
    % 统一风格创建图形窗口
    if nargin < 3, h = 500; end
    if nargin < 2, w = 1200; end
    f = figure('Name', name, 'NumberTitle', 'off', ...
               'Position', [50, 50, w, h], 'Color', 'k');
    % 黑底配白字
    set(f, 'DefaultTextColor', 'w', 'DefaultAxesColor', 'k', ...
           'DefaultAxesXColor', 'w', 'DefaultAxesYColor', 'w');
end

function demo_pause(enabled, msg)
    % 课堂步进暂停（DEMO_MODE=1时等待用户确认）
    % 使用 input 而非 waitforbuttonpress，避免图窗关闭导致崩溃
    if enabled
        if nargin < 2, msg = '继续演示'; end
        input(sprintf('\n  ▶ %s [Enter] ', msg), 's');
        fprintf('\n');
    end
end

function ch = denoise_channel(ch, method, param)
    % 单通道去噪接口（支持可选参数param）
    switch method
        case 'medfilt2'
            if nargin < 3, ws = [5,5]; else, ws = [param, param]; end
            ch = medfilt2(ch, ws);
        case 'gaussian'
            if nargin < 3, s = 1.5; else, s = param; end
            ch = imgaussfilt(ch, s);
        case 'bilateral',   ch = imbilatfilt(ch);
        case 'wiener'
            if nargin < 3, ws = [5,5]; else, ws = [param, param]; end
            ch = wiener2(ch, ws);
        case 'nlm',         ch = imnlmfilt(ch);
        case 'diffusion',   ch = imdiffusefilt(ch);
        otherwise, error('未知方法: %s', method);
    end
end

function rgb_out = denoise_rgb_channels(rgb_in, method)
    % 直接方法: R/G/B三通道分别去噪
    rgb_out = zeros(size(rgb_in));
    for c = 1:3
        rgb_out(:,:,c) = denoise_channel(rgb_in(:,:,c), method);
    end
end

function rgb_out = process_luminance(rgb_in, colorspace, method, param)
    % 间接方法核心: 颜色空间→亮度去噪→反变换
    switch colorspace
        case 'YCbCr'
            cs = rgb2ycbcr(rgb_in);  lum = 1;
        case 'Lab'
            cs = rgb2lab(rgb_in);    lum = 1;
        case 'HSV'
            cs = rgb2hsv(rgb_in);    lum = 3;
    end
    if nargin >= 4
        cs(:,:,lum) = denoise_channel(cs(:,:,lum), method, param);
    else
        cs(:,:,lum) = denoise_channel(cs(:,:,lum), method);
    end
    switch colorspace
        case 'YCbCr', rgb_out = ycbcr2rgb(cs);
        case 'Lab',   rgb_out = max(0, min(1, lab2rgb(cs)));
        case 'HSV',   rgb_out = hsv2rgb(cs);
    end
end

function iou = compute_iou(mask, gt)
    inter = sum(mask(:) & gt(:));
    union = sum(mask(:) | gt(:));
    iou = inter / max(union, 1);
end

function gt = load_ground_truth(path, ref)
    if exist(path, 'file')
        gt = imread(path) > 128;
        if size(gt, 3) > 1, gt = rgb2gray(gt) > 128; end
        if size(gt,1) ~= size(ref,1) || size(gt,2) ~= size(ref,2)
            gt = imresize(gt, [size(ref,1), size(ref,2)]) > 0.5;
        end
        fprintf('  [GT] 已加载: %s\n', path);
    else
        gt = [];
    end
end
