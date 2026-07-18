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

%% 实验7: 参数调优文件（独立运行）
% 用于课堂演示时的参数敏感性分析
% 可单独运行，不依赖主文件变量
%
% 【内容】
%   任务3调优: 中值滤波窗口 / Lab高斯sigma / YCbCr双边强度
%              HSV-V通道sigma / illumpca PCA百分比
%   任务4调优: H/S/V阈值 / 形态学参数 / 双空间融合
%
% 【用法】直接运行即可，所有参数范围与实验要求一致

clear all; close all; clc;

%% ── 初始化 ──

p_lena   = 'image_lena512rgb.png';
p_yellow = 'yellowlily.jpg';
p_gt     = 'yellowlily_gt.png';
sigma_noise = 30;
addpath('../BM3D');

% 读取并添加噪声
lena = im2double(imread(p_lena));
rng(0);
lena_noisy = lena + (sigma_noise/255) * randn(size(lena));
lena_noisy = max(0, min(1, lena_noisy));

fprintf('========================================\n');
fprintf('  实验7 参数调优\n');
fprintf('  参数范围与原始实验要求完全一致\n');
fprintf('========================================\n\n');

%% ======================================================================
%  任务3调优: 去噪参数扫描
%  ======================================================================

%% ─── 3a. 中值滤波窗口大小 ───

fprintf('【3a】中值滤波窗口大小扫描\n');
fprintf('  范围: [3, 5, 7, 9]\n\n');

med_vals = [3, 5, 7, 9];
med_psnr = zeros(size(med_vals));
med_ssim = zeros(size(med_vals));

for k = 1:length(med_vals)
    ws = med_vals(k);
    denoised = zeros(size(lena_noisy));
    for c = 1:3
        denoised(:,:,c) = medfilt2(lena_noisy(:,:,c), [ws ws]);
    end
    med_psnr(k) = psnr(denoised, lena);
    med_ssim(k) = ssim(denoised, lena);
    fprintf('  窗口[%d %d]: PSNR=%.4f dB, SSIM=%.4f\n', ws, ws, med_psnr(k), med_ssim(k));
end
[~, best_med] = max(med_psnr);
fprintf('  ★ 最优窗口: [%d %d] (PSNR=%.4f)\n\n', med_vals(best_med), med_vals(best_med), max(med_psnr));

%% ─── 3b. Lab-L通道高斯sigma ───

fprintf('【3b】Lab-L通道高斯sigma扫描\n');
fprintf('  范围: [0.5, 1, 1.5, 2, 3, 4, 5]\n\n');

lena_lab = rgb2lab(lena_noisy);
sig_vals = [0.5, 1, 1.5, 2, 3, 4, 5];
sig_psnr = zeros(size(sig_vals));
sig_ssim = zeros(size(sig_vals));

for k = 1:length(sig_vals)
    s = sig_vals(k);
    lab_tmp = lena_lab;
    lab_tmp(:,:,1) = imgaussfilt(lena_lab(:,:,1), s);
    rgb_tmp = max(0, min(1, lab2rgb(lab_tmp)));
    sig_psnr(k) = psnr(rgb_tmp, lena);
    sig_ssim(k) = ssim(rgb_tmp, lena);
    fprintf('  sigma=%.1f: PSNR=%.4f dB, SSIM=%.4f\n', s, sig_psnr(k), sig_ssim(k));
end
[~, best_sig] = max(sig_psnr);
fprintf('  ★ 最优sigma=%.1f (PSNR=%.4f)\n\n', sig_vals(best_sig), max(sig_psnr));

%% ─── 3c. YCbCr双边滤波强度 ───

fprintf('【3c】YCbCr双边滤波 DegreeOfSmoothing 扫描\n');
fprintf('  范围: [0.5, 1, 1.5, 2, 3]\n\n');

ycbcr_work = rgb2ycbcr(lena_noisy);
dof_vals   = [0.5, 1, 1.5, 2, 3];
dof_psnr   = zeros(size(dof_vals));
dof_ssim   = zeros(size(dof_vals));

% 固定Cb/Cr滤波参数
cb_gauss = imgaussfilt(ycbcr_work(:,:,2), 1.5);
cr_gauss = imgaussfilt(ycbcr_work(:,:,3), 1.5);

for k = 1:length(dof_vals)
    d = dof_vals(k);
    y_tmp = imbilatfilt(ycbcr_work(:,:,1), 'DegreeOfSmoothing', d);
    rgb_tmp = ycbcr2rgb(cat(3, y_tmp, cb_gauss, cr_gauss));
    dof_psnr(k) = psnr(rgb_tmp, lena);
    dof_ssim(k) = ssim(rgb_tmp, lena);
    fprintf('  DoS=%.1f: PSNR=%.4f dB, SSIM=%.4f\n', d, dof_psnr(k), dof_ssim(k));
end
[~, best_dof] = max(dof_psnr);
fprintf('  ★ 最优DoS=%.1f (PSNR=%.4f)\n\n', dof_vals(best_dof), max(dof_psnr));

%% ─── 3d. HSV V通道高斯sigma ───

fprintf('【3d】HSV V通道高斯sigma扫描\n');
fprintf('  范围: [0.5, 1, 1.5, 2, 3]\n\n');

hsv_work = rgb2hsv(lena_noisy);
hsv_sig  = [0.5, 1, 1.5, 2, 3];
hsv_psnr = zeros(size(hsv_sig));
hsv_ssim = zeros(size(hsv_sig));

for k = 1:length(hsv_sig)
    s = hsv_sig(k);
    v_tmp = imgaussfilt(hsv_work(:,:,3), s);
    rgb_tmp = hsv2rgb(cat(3, hsv_work(:,:,1), hsv_work(:,:,2), v_tmp));
    hsv_psnr(k) = psnr(rgb_tmp, lena);
    hsv_ssim(k) = ssim(rgb_tmp, lena);
    fprintf('  sigma=%.1f: PSNR=%.4f dB, SSIM=%.4f\n', s, hsv_psnr(k), hsv_ssim(k));
end
[~, best_hsv] = max(hsv_psnr);
fprintf('  ★ 最优sigma=%.1f (PSNR=%.4f)\n\n', hsv_sig(best_hsv), max(hsv_psnr));

%% ─── 3e. illumpca PCA百分比 ───

fprintf('【3e】illumpca PCA百分比扫描\n');
fprintf('  范围: [1, 2, 3.5, 5, 10]\n\n');

lin_img  = rgb2lin(lena_noisy);
pct_vals = [1, 2, 3.5, 5, 10];
pct_psnr = zeros(size(pct_vals));
pct_ssim = zeros(size(pct_vals));

for k = 1:length(pct_vals)
    p = pct_vals(k);
    illu = illumpca(lin_img, p);
    wb   = chromadapt(lena_noisy, illu);
    wb_y = rgb2ycbcr(wb);
    wy   = imbilatfilt(wb_y(:,:,1));
    wc   = imgaussfilt(wb_y(:,:,2), 2);
    wr   = imgaussfilt(wb_y(:,:,3), 2);
    rgb_tmp = ycbcr2rgb(cat(3, wy, wc, wr));
    pct_psnr(k) = psnr(rgb_tmp, lena);
    pct_ssim(k) = ssim(rgb_tmp, lena);
    fprintf('  PCA_pct=%.1f: PSNR=%.4f dB, SSIM=%.4f\n', p, pct_psnr(k), pct_ssim(k));
end
[~, best_pct] = max(pct_psnr);
fprintf('  ★ 最优PCA_pct=%.1f (PSNR=%.4f)\n\n', pct_vals(best_pct), max(pct_psnr));

%% ─── 任务3调优汇总图 ───

figure('Name', '任务3调优汇总', 'NumberTitle', 'off', ...
       'Position', [50, 50, 1500, 500]);

subplot(1, 3, 1);
yyaxis left;  plot(med_vals, med_psnr, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 8); ylabel('PSNR (dB)');
yyaxis right; plot(med_vals, med_ssim, 'r-s', 'LineWidth', 1.5, 'MarkerSize', 8); ylabel('SSIM');
xlabel('中值滤波窗口'); title('中值滤波窗口 vs 去噪指标');
legend({'PSNR','SSIM'}, 'Location', 'best'); grid on;

subplot(1, 3, 2);
yyaxis left;  plot(sig_vals, sig_psnr, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 8); ylabel('PSNR (dB)');
yyaxis right; plot(sig_vals, sig_ssim, 'r-s', 'LineWidth', 1.5, 'MarkerSize', 8); ylabel('SSIM');
xlabel('高斯sigma'); title('Lab-L高斯sigma vs 去噪指标');
legend({'PSNR','SSIM'}, 'Location', 'best'); grid on;

subplot(1, 3, 3);
hold on;
plot(dof_vals, dof_psnr, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 8);
plot(hsv_sig,  hsv_psnr, 'g-^', 'LineWidth', 1.5, 'MarkerSize', 8);
plot(pct_vals, pct_psnr, 'm-d', 'LineWidth', 1.5, 'MarkerSize', 8);
hold off;
xlabel('参数值'); ylabel('PSNR (dB)'); title('颜色空间方法参数 vs PSNR');
legend({'YCbCr DoS','HSV sigma','illumpca PCA%'}, 'Location', 'best'); grid on;

sgtitle('任务3: 去噪参数优化汇总', 'FontSize', 14);

%% ======================================================================
%  任务4调优: 分割参数扫描
%  ======================================================================

fprintf('====== 任务4调优: 分割参数扫描 ======\n\n');

yellow   = im2double(imread(p_yellow));
yellow_hsv = rgb2hsv(yellow);
H_ch = yellow_hsv(:,:,1);  S_ch = yellow_hsv(:,:,2);  V_ch = yellow_hsv(:,:,3);

% 加载/生成GT
gt_mask = load_gt(p_gt, yellow);
if isempty(gt_mask)
    fprintf('  未找到GT，用Lab b*+Otsu自动生成参考掩模\n');
    lab_auto = rgb2lab(yellow);
    b_auto   = lab_auto(:,:,3);
    level    = graythresh(b_auto);
    gt_mask  = b_auto > level * max(b_auto(:));
    gt_mask  = imopen(gt_mask, strel('disk', 3));
    gt_mask  = imclose(gt_mask, strel('disk', 5));
    gt_mask  = imfill(gt_mask, 'holes');
end

%% ─── 4a. H阈值扫描 ───

fprintf('【4a】H阈值扫描\n\n');

% H下限
H_low  = 0.05:0.02:0.15;
iou_hl = zeros(size(H_low));
for k = 1:length(H_low)
    m = (H_ch > H_low(k)) & (H_ch < 0.25) & (S_ch > 0.3) & (V_ch > 0.3);
    m = morph_clean(m, 3, 5);
    iou_hl(k) = compute_iou(m, gt_mask);
    fprintf('  H_low=%.2f: IoU=%.4f\n', H_low(k), iou_hl(k));
end
[~, bi_hl] = max(iou_hl);
best_H_low = H_low(bi_hl);
fprintf('  ★ 最优H_low=%.2f (IoU=%.4f)\n\n', best_H_low, max(iou_hl));

% H上限（基于最优下限）
H_high  = 0.18:0.02:0.35;
iou_hh  = zeros(size(H_high));
for k = 1:length(H_high)
    m = (H_ch > best_H_low) & (H_ch < H_high(k)) & (S_ch > 0.3) & (V_ch > 0.3);
    m = morph_clean(m, 3, 5);
    iou_hh(k) = compute_iou(m, gt_mask);
    fprintf('  H_high=%.2f: IoU=%.4f\n', H_high(k), iou_hh(k));
end
[~, bi_hh] = max(iou_hh);
best_H_high = H_high(bi_hh);
fprintf('  ★ 最优H范围: [%.2f, %.2f] (IoU=%.4f)\n\n', best_H_low, best_H_high, max(iou_hh));

%% ─── 4b. S/V阈值扫描 ───

fprintf('【4b】S和V阈值扫描\n\n');

S_vals = 0.1:0.1:0.6;
iou_S  = zeros(size(S_vals));
for k = 1:length(S_vals)
    m = (H_ch > best_H_low) & (H_ch < best_H_high) & (S_ch > S_vals(k)) & (V_ch > 0.3);
    m = morph_clean(m, 3, 5);
    iou_S(k) = compute_iou(m, gt_mask);
    fprintf('  S>%.1f: IoU=%.4f\n', S_vals(k), iou_S(k));
end
[~, bi_S] = max(iou_S);

V_vals = 0.1:0.1:0.6;
iou_V  = zeros(size(V_vals));
for k = 1:length(V_vals)
    m = (H_ch > best_H_low) & (H_ch < best_H_high) & (S_ch > S_vals(bi_S)) & (V_ch > V_vals(k));
    m = morph_clean(m, 3, 5);
    iou_V(k) = compute_iou(m, gt_mask);
    fprintf('  V>%.1f: IoU=%.4f\n', V_vals(k), iou_V(k));
end
[~, bi_V] = max(iou_V);
fprintf('  ★ 最优S>%.1f, V>%.1f (IoU=%.4f)\n\n', S_vals(bi_S), V_vals(bi_V), max(iou_V));

%% ─── 4c. 形态学参数扫描 ───

fprintf('【4c】形态学参数扫描\n\n');

% 开运算
open_r = [1, 2, 3, 5, 7];
iou_op = zeros(size(open_r));
for k = 1:length(open_r)
    m = (H_ch > best_H_low) & (H_ch < best_H_high) & ...
        (S_ch > S_vals(bi_S)) & (V_ch > V_vals(bi_V));
    m = imopen(m, strel('disk', open_r(k)));
    m = imclose(m, strel('disk', 5));
    m = imfill(m, 'holes');
    iou_op(k) = compute_iou(m, gt_mask);
    fprintf('  open_disk=%d: IoU=%.4f\n', open_r(k), iou_op(k));
end
[~, bi_op] = max(iou_op);
fprintf('  ★ 最优open_disk=%d (IoU=%.4f)\n\n', open_r(bi_op), max(iou_op));

% 闭运算
close_r = [1, 3, 5, 7, 10];
iou_cl = zeros(size(close_r));
for k = 1:length(close_r)
    m = (H_ch > best_H_low) & (H_ch < best_H_high) & ...
        (S_ch > S_vals(bi_S)) & (V_ch > V_vals(bi_V));
    m = imopen(m, strel('disk', open_r(bi_op)));
    m = imclose(m, strel('disk', close_r(k)));
    m = imfill(m, 'holes');
    iou_cl(k) = compute_iou(m, gt_mask);
    fprintf('  close_disk=%d: IoU=%.4f\n', close_r(k), iou_cl(k));
end
[~, bi_cl] = max(iou_cl);
fprintf('  ★ 最优close_disk=%d (IoU=%.4f)\n\n', close_r(bi_cl), max(iou_cl));

% 最优参数下的分割
best_mask = (H_ch > best_H_low) & (H_ch < best_H_high) & ...
            (S_ch > S_vals(bi_S)) & (V_ch > V_vals(bi_V));
best_mask = imopen(best_mask, strel('disk', open_r(bi_op)));
best_mask = imclose(best_mask, strel('disk', close_r(bi_cl)));
best_mask = imfill(best_mask, 'holes');
best_iou  = compute_iou(best_mask, gt_mask);
fprintf('  ★ 最优分割 IoU = %.4f\n\n', best_iou);

%% ─── 4d. HSV + Lab 双空间融合 ───

fprintf('【4d】HSV + Lab 双空间融合分割\n\n');

lab_y = rgb2lab(yellow);
b_star = lab_y(:,:,3);
b_level = graythresh(b_star);
lab_mask_raw = b_star > b_level * max(b_star(:));
lab_mask = imopen(lab_mask_raw, strel('disk', 3));
lab_mask = imclose(lab_mask, strel('disk', 5));
lab_mask = imfill(lab_mask, 'holes');
iou_lab = compute_iou(lab_mask, gt_mask);

% 融合策略
and_mask = best_mask & lab_mask;
or_mask  = best_mask | lab_mask;
iou_and  = compute_iou(and_mask, gt_mask);
iou_or   = compute_iou(or_mask,  gt_mask);

fprintf('  HSV最优:   IoU=%.4f\n', best_iou);
fprintf('  Lab Otsu:  IoU=%.4f\n', iou_lab);
fprintf('  HSV ∩ Lab: IoU=%.4f\n', iou_and);
fprintf('  HSV ∪ Lab: IoU=%.4f\n', iou_or);

[best_f_iou, best_f] = max([best_iou, iou_lab, iou_and, iou_or]);
names_f = {'HSV最优','Lab Otsu','HSV∩Lab','HSV∪Lab'};
fprintf('  ★ 最优融合策略: %s (IoU=%.4f)\n\n', names_f{best_f}, best_f_iou);

%% ─── 任务4调优汇总图 ───

figure('Name', '任务4调优汇总', 'NumberTitle', 'off', ...
       'Position', [50, 50, 1400, 700]);

subplot(2, 3, 1);
plot(H_low, iou_hl, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 8); grid on;
xlabel('H下限'); ylabel('IoU'); title('H下限 vs IoU');

subplot(2, 3, 2);
plot(H_high, iou_hh, 'r-s', 'LineWidth', 1.5, 'MarkerSize', 8); grid on;
xlabel('H上限'); ylabel('IoU'); title('H上限 vs IoU');

subplot(2, 3, 3);
plot(S_vals, iou_S, 'g-^', 'LineWidth', 1.5, 'MarkerSize', 8); hold on;
plot(V_vals, iou_V, 'm-d', 'LineWidth', 1.5, 'MarkerSize', 8); grid on;
xlabel('阈值'); ylabel('IoU'); title('S/V阈值 vs IoU');
legend({'S','V'}, 'Location', 'best');

subplot(2, 3, 4);
bar(open_r, iou_op, 0.5, 'FaceColor', [0.5 0.7 1]); grid on;
xlabel('开运算disk半径'); ylabel('IoU'); title('开运算半径 vs IoU');

subplot(2, 3, 5);
bar(close_r, iou_cl, 0.5, 'FaceColor', [0.6 0.9 0.6]); grid on;
xlabel('闭运算disk半径'); ylabel('IoU'); title('闭运算半径 vs IoU');

subplot(2, 3, 6);
bar(categorical(names_f), [best_iou, iou_lab, iou_and, iou_or], 0.6);
ylabel('IoU'); title('分割策略对比'); grid on;
text(1, best_iou, sprintf('%.4f', best_iou), 'HorizontalAlignment','center','VerticalAlignment','bottom');

sgtitle('任务4: 分割参数优化汇总', 'FontSize', 14);

%% ======================================================================
%  最优参数汇总
%  ======================================================================

fprintf('\n========================================\n');
fprintf('  ★ 任务3最优参数汇总\n');
fprintf('========================================\n');
fprintf('  中值滤波窗口:     [%d %d]  PSNR=%.4f\n', med_vals(best_med), med_vals(best_med), max(med_psnr));
fprintf('  Lab-L高斯sigma:   %.1f         PSNR=%.4f\n', sig_vals(best_sig), max(sig_psnr));
fprintf('  YCbCr DoS:        %.1f         PSNR=%.4f\n', dof_vals(best_dof), max(dof_psnr));
fprintf('  HSV-V sigma:      %.1f         PSNR=%.4f\n', hsv_sig(best_hsv), max(hsv_psnr));
fprintf('  illumpca PCA%%:    %.1f         PSNR=%.4f\n', pct_vals(best_pct), max(pct_psnr));

fprintf('\n========================================\n');
fprintf('  ★ 任务4最优参数汇总\n');
fprintf('========================================\n');
fprintf('  H范围:   [%.2f, %.2f]\n', best_H_low, best_H_high);
fprintf('  S阈值:   %.1f\n', S_vals(bi_S));
fprintf('  V阈值:   %.1f\n', V_vals(bi_V));
fprintf('  开运算:  disk=%d\n', open_r(bi_op));
fprintf('  闭运算:  disk=%d\n', close_r(bi_cl));
fprintf('  最优IoU: %.4f  (策略: %s)\n', best_f_iou, names_f{best_f});
fprintf('========================================\n');
fprintf('  参数调优完成!\n');
fprintf('========================================\n');

%% ── 保存最优参数到 .mat 文件 ──
% 供 experiment7_optimized.m 自动加载使用

best_params = struct();

% 任务3 去噪参数
best_params.medfilt2_window = med_vals(best_med);      % 最优中值滤波窗口
best_params.lab_gauss_sigma = sig_vals(best_sig);      % 最优Lab高斯sigma
best_params.ycbcr_dos       = dof_vals(best_dof);      % 最优YCbCr双边强度
best_params.hsv_v_sigma     = hsv_sig(best_hsv);       % 最优HSV-V sigma
best_params.illumpca_pca    = pct_vals(best_pct);      % 最优illumpca PCA%

% 任务4 分割参数
best_params.H_low     = best_H_low;
best_params.H_high    = best_H_high;
best_params.S_thresh  = S_vals(bi_S);
best_params.V_thresh  = V_vals(bi_V);
best_params.open_r    = open_r(bi_op);
best_params.close_r   = close_r(bi_cl);
best_params.fusion_iou = best_f_iou;

save('exp7_best_params.mat', 'best_params');
fprintf('\n  [已保存] 最优参数 → exp7_best_params.mat\n');
fprintf('  主文件 experiment7_optimized.m 将自动加载此文件\n');

%% ======================================================================
%  局部函数
%  ======================================================================

function m = morph_clean(m, r_open, r_close)
    % 形态学清理: 开运算 → 闭运算 → 填孔洞
    m = imopen(m, strel('disk', r_open));
    m = imclose(m, strel('disk', r_close));
    m = imfill(m, 'holes');
end

function iou = compute_iou(mask, gt)
    inter = sum(mask(:) & gt(:));
    union = sum(mask(:) | gt(:));
    iou = inter / max(union, 1);
end

function gt = load_gt(path, ref)
    if exist(path, 'file')
        gt = imread(path) > 128;
        if size(gt,3) > 1, gt = rgb2gray(gt) > 128; end
        if size(gt,1) ~= size(ref,1) || size(gt,2) ~= size(ref,2)
            gt = imresize(gt, [size(ref,1), size(ref,2)]) > 0.5;
        end
    else
        gt = [];
    end
end
