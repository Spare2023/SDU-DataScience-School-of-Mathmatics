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

%% 实验4: 参数调优辅助脚本
%  粗扫（大步长）+ 细扫（小步长）分级调参策略
%  多指标评估：PSNR（像素误差）+ SSIM（结构相似）+ EPR（边缘保留度）
%  Knee-point 拐点检测 + 加权综合评分
%
%  使用方法：
%    1. 确保当前目录包含 lena.png
%    2. 根据需要修改下方 "用户配置" 中的权衡模式
%    3. 运行本脚本（不需要先运行 experiment4.m）
%    4. 查看控制台输出的最优参数和多指标曲线图
%    5. 最优参数自动保存至 optimal_params.mat，供 experiment4.m 加载
%
%  注意：本脚本独立于 experiment4.m，不修改原文件

%% ====================================================================
%  初始化
% ====================================================================
clear; close all; clc;

%% ====================================================================
%  用户配置：去噪与细节保留的权衡模式
% ====================================================================
% 'balance'    - 平衡（默认） PSNR=0.3, SSIM=0.4, EPR=0.3
% 'aggressive' - 强去噪       PSNR=0.4, SSIM=0.4, EPR=0.2
% 'detail'     - 保细节       PSNR=0.2, SSIM=0.3, EPR=0.5
tradeoff_mode = 'balance';

fprintf('========================================================\n');
fprintf('  实验4: 参数调优辅助脚本\n');
fprintf('  策略：粗扫（大步长确定区间）→ 细扫（小步长精确定位）\n');
fprintf('  指标：PSNR + SSIM + 边缘保留度(EPR)\n');

switch tradeoff_mode
    case 'aggressive'
        fprintf('  模式：强去噪（PSNR:SSIM:EPR = 0.4:0.4:0.2）\n');
    case 'detail'
        fprintf('  模式：保细节（PSNR:SSIM:EPR = 0.2:0.3:0.5）\n');
    otherwise
        fprintf('  模式：平衡（PSNR:SSIM:EPR = 0.3:0.4:0.3）\n');
end
fprintf('========================================================\n\n');

%% 读入图像（与 experiment4.m 相同的预处理）
img_path = 'lena.png';

img = imread(img_path);
if size(img, 3) == 3
    img_gray = rgb2gray(img);
else
    img_gray = img;
end
img_gray = im2double(img_gray);

fprintf('使用图像: %s\n', img_path);
fprintf('图像大小: %dx%d\n\n', size(img_gray, 1), size(img_gray, 2));

% ====================================================================
%  调优1：D0（截止频率）— 任务2：高斯卷积与频域低通滤波
%  目的：找到去噪与保留细节的最佳平衡点
%  流程：粗扫(5~200,步长15~20) → 细扫(最佳值±20,步长2~5)
% ====================================================================
fprintf('========================================================\n');
fprintf('  调优1: D0（频域高斯低通截止频率）\n');
fprintf('  对应: 任务2 高斯卷积与频域低通滤波比较\n');
fprintf('========================================================\n');

% 固定噪声参数
sigma_noise = 30;
rng(0);
img_noisy = img_gray + (sigma_noise/255) * randn(size(img_gray));
img_noisy = max(0, min(1, img_noisy));

fprintf('\n噪声水平: sigma_noise = %d\n', sigma_noise);

%% ---- 阶段1：粗扫（大步长，~10次） ----
fprintf('\n--- 阶段1: 粗扫 ---\n');
D0_coarse = [5, 20, 35, 50, 65, 80, 100, 120, 150, 200];
n_coarse = length(D0_coarse);
psnr_coarse = zeros(n_coarse, 1);
ssim_coarse = zeros(n_coarse, 1);
epr_coarse  = zeros(n_coarse, 1);

for k = 1:n_coarse
    img_f = gaussian_lp_filter(img_noisy, img_gray, D0_coarse(k));
    [psnr_coarse(k), ssim_coarse(k), epr_coarse(k)] = calc_metrics(img_f, img_gray);
    fprintf('  D0=%3d  →  PSNR=%.4f  SSIM=%.4f  EPR=%.4f\n', ...
        D0_coarse(k), psnr_coarse(k), ssim_coarse(k), epr_coarse(k));
end

% 综合评分 + knee-point
score_coarse = compute_weighted_score(psnr_coarse, ssim_coarse, epr_coarse, tradeoff_mode);
[~, best_idx] = max(score_coarse);
knee_idx_c = find_knee_point(psnr_coarse);
best_D0 = D0_coarse(best_idx);
knee_D0  = D0_coarse(knee_idx_c);

fprintf('\n  ★ 综合评分最佳: D0=%d  (PSNR=%.4f, SSIM=%.4f, EPR=%.4f)\n', ...
    best_D0, psnr_coarse(best_idx), ssim_coarse(best_idx), epr_coarse(best_idx));
fprintf('  ◆ Knee-point:    D0=%d  (PSNR=%.4f)\n', ...
    knee_D0, psnr_coarse(knee_idx_c));

%% ---- 阶段2：细扫（最佳值附近加密） ----
fprintf('\n--- 阶段2: 细扫 ---\n');
lo = max(5, best_D0 - 20);
hi = min(200, best_D0 + 20);
step = max(2, floor((hi - lo) / 8));
D0_fine = lo:step:hi;
if ~ismember(best_D0, D0_fine)
    D0_fine = sort([D0_fine, best_D0]);
end

n_fine = length(D0_fine);
psnr_fine = zeros(n_fine, 1);
ssim_fine = zeros(n_fine, 1);
epr_fine  = zeros(n_fine, 1);

for k = 1:n_fine
    img_f = gaussian_lp_filter(img_noisy, img_gray, D0_fine(k));
    [psnr_fine(k), ssim_fine(k), epr_fine(k)] = calc_metrics(img_f, img_gray);
    fprintf('  D0=%3d  →  PSNR=%.4f  SSIM=%.4f  EPR=%.4f\n', ...
        D0_fine(k), psnr_fine(k), ssim_fine(k), epr_fine(k));
end

score_fine = compute_weighted_score(psnr_fine, ssim_fine, epr_fine, tradeoff_mode);
[~, best_fine_idx] = max(score_fine);
knee_idx_f = find_knee_point(psnr_fine);
best_D0_final = D0_fine(best_fine_idx);
knee_D0_final = D0_fine(knee_idx_f);

fprintf('\n  ★ 综合评分最佳: D0=%d  (PSNR=%.4f, SSIM=%.4f, EPR=%.4f)\n', ...
    best_D0_final, psnr_fine(best_fine_idx), ssim_fine(best_fine_idx), epr_fine(best_fine_idx));
fprintf('  ◆ Knee-point:    D0=%d  (PSNR=%.4f)\n', ...
    knee_D0_final, psnr_fine(knee_idx_f));

% 对比空域高斯卷积
h_gaussian = fspecial('gaussian', [15 15], 2);
img_spatial = imfilter(img_noisy, h_gaussian, 'replicate');
[psnr_sp, ssim_sp, epr_sp] = calc_metrics(img_spatial, img_gray);
fprintf('\n  对比: 空域高斯卷积 → PSNR=%.4f  SSIM=%.4f  EPR=%.4f\n', psnr_sp, ssim_sp, epr_sp);

%% ---- 绘制 D0 调优曲线（三指标 + knee-point） ----
figure('Name', '调优1: D0 截止频率', 'NumberTitle', 'off', ...
       'Position', [100, 100, 1100, 750]);

% 粗扫图
subplot(2, 2, 1);
yyaxis left;
plot(D0_coarse, psnr_coarse, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 6);
ylabel('PSNR (dB)');
yyaxis right;
plot(D0_coarse, ssim_coarse, 'r-s', 'LineWidth', 1.5, 'MarkerSize', 6);
hold on;
plot(D0_coarse, epr_coarse, 'g-^', 'LineWidth', 1.5, 'MarkerSize', 6);
ylabel('SSIM / EPR');
xlabel('截止频率 D0'); title('D0 粗扫（三指标）');
legend({'PSNR', 'SSIM', 'EPR'}, 'Location', 'best');
grid on;

% 粗扫综合评分
subplot(2, 2, 2);
plot(D0_coarse, score_coarse, 'm-o', 'LineWidth', 1.5, 'MarkerSize', 8); hold on;
plot(best_D0, score_coarse(best_idx), 'r*', 'MarkerSize', 20);
plot(knee_D0, score_coarse(knee_idx_c), 'b^', 'MarkerSize', 12);
text(best_D0, score_coarse(best_idx), sprintf(' 最佳 D0=%d', best_D0), ...
     'VerticalAlignment', 'bottom');
text(knee_D0, score_coarse(knee_idx_c), sprintf(' Knee D0=%d', knee_D0), ...
     'VerticalAlignment', 'top');
xlabel('截止频率 D0'); ylabel('综合评分');
title(sprintf('粗扫综合评分（%s模式）', tradeoff_mode));
grid on; legend('评分', '最佳', 'Knee-point', 'Location', 'best');

% 细扫图
subplot(2, 2, 3);
yyaxis left;
plot(D0_fine, psnr_fine, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 6);
ylabel('PSNR (dB)');
yyaxis right;
plot(D0_fine, ssim_fine, 'r-s', 'LineWidth', 1.5, 'MarkerSize', 6);
hold on;
plot(D0_fine, epr_fine, 'g-^', 'LineWidth', 1.5, 'MarkerSize', 6);
ylabel('SSIM / EPR');
xlabel('截止频率 D0'); title(sprintf('D0 细扫（步长%d）', step));
legend({'PSNR', 'SSIM', 'EPR'}, 'Location', 'best');
grid on;

% 细扫综合评分
subplot(2, 2, 4);
plot(D0_fine, score_fine, 'm-o', 'LineWidth', 1.5, 'MarkerSize', 8); hold on;
plot(best_D0_final, score_fine(best_fine_idx), 'r*', 'MarkerSize', 20);
plot(knee_D0_final, score_fine(knee_idx_f), 'b^', 'MarkerSize', 12);
text(best_D0_final, score_fine(best_fine_idx), sprintf(' 最佳 D0=%d', best_D0_final), ...
     'VerticalAlignment', 'bottom');
text(knee_D0_final, score_fine(knee_idx_f), sprintf(' Knee D0=%d', knee_D0_final), ...
     'VerticalAlignment', 'top');
xlabel('截止频率 D0'); ylabel('综合评分');
title(sprintf('细扫综合评分（%s模式）', tradeoff_mode));
grid on; legend('评分', '最佳', 'Knee-point', 'Location', 'best');

sgtitle(sprintf('调优1: 频域高斯低通截止频率 D0（%s模式）', tradeoff_mode), 'FontSize', 14);

%% ---- 关键位置对比图（粗扫结果中选三个典型 D0） ----
% 选取 low-D0（欠平滑）、mid-D0（knee附近）、high-D0（过平滑）
D0_show = [D0_coarse(1), knee_D0, best_D0];
D0_show = unique(D0_show);
if length(D0_show) < 3
    % 补全：取第一个、中间一个、最后一个
    D0_show = [D0_coarse(1), D0_coarse(round(end/2)), D0_coarse(end)];
end
D0_show_labels = {'细节好·噪声多', 'Knee平衡点', '去噪强·细节少'};

figure('Name', 'D0 效果对比', 'NumberTitle', 'off', ...
       'Position', [100, 100, 1100, 600]);

for k = 1:3
    img_f = gaussian_lp_filter(img_noisy, img_gray, D0_show(k));
    [p, s, e] = calc_metrics(img_f, img_gray);

    subplot(2, 3, k);
    imshow(img_f, []);
    title({sprintf('D0=%d: %s', D0_show(k), D0_show_labels{k}), ...
           sprintf('PSNR=%.4f  SSIM=%.3f  EPR=%.3f', p, s, e)});

    % 局部放大区域（取图像中心附近 80x80 区域）
    subplot(2, 3, k+3);
    [M, N] = size(img_gray);
    crop_rect = round([M/2-20, N/2-20, 79, 79]);
    imshow(img_f(crop_rect(1):crop_rect(1)+crop_rect(3), ...
                 crop_rect(2):crop_rect(2)+crop_rect(4)), []);
    title(sprintf('局部放大 (80×80)'));
end

sgtitle('调优1: 不同 D0 效果对比', 'FontSize', 14);

% ====================================================================
%  调优2：sigma_noise 对滤波效果的影响
%  目的：观察不同噪声强度下固定 D0 的滤波性能
%  流程：固定 D0=最优值，扫描 sigma_noise=5~100，步长15~20
% ====================================================================
fprintf('\n========================================================\n');
fprintf('  调优2: sigma_noise 对滤波效果的影响\n');
fprintf('  固定 D0 = %d\n', best_D0_final);
fprintf('========================================================\n');

sigma_values = [5, 15, 30, 50, 70, 100];
n_sigma = length(sigma_values);
psnr_noisy_sigma = zeros(n_sigma, 1);
psnr_filtered_sigma = zeros(n_sigma, 1);
ssim_filtered_sigma = zeros(n_sigma, 1);

for k = 1:n_sigma
    rng(0);
    noisy = img_gray + (sigma_values(k)/255) * randn(size(img_gray));
    noisy = max(0, min(1, noisy));
    psnr_noisy_sigma(k) = psnr(noisy, img_gray);
    img_f = gaussian_lp_filter(noisy, img_gray, best_D0_final);
    [psnr_filtered_sigma(k), ssim_filtered_sigma(k), ~] = calc_metrics(img_f, img_gray);
    fprintf('  sigma_noise=%3d  →  噪声PSNR=%.4f  →  滤波后PSNR=%.4f  SSIM=%.4f  →  提升%+.4f dB\n', ...
        sigma_values(k), psnr_noisy_sigma(k), psnr_filtered_sigma(k), ssim_filtered_sigma(k), ...
        psnr_filtered_sigma(k) - psnr_noisy_sigma(k));
end

figure('Name', '调优2: sigma_noise 影响', 'NumberTitle', 'off', ...
       'Position', [100, 100, 1200, 400]);

subplot(1, 3, 1);
plot(sigma_values, psnr_filtered_sigma, 'go-', 'LineWidth', 1.5, 'MarkerSize', 8); hold on;
plot(sigma_values, psnr_noisy_sigma, 'r^--', 'LineWidth', 1, 'MarkerSize', 6);
xlabel('sigma_{noise}'); ylabel('PSNR (dB)');
title(sprintf('PSNR vs sigma_{noise} (D0=%d)', best_D0_final));
legend('滤波后', '滤波前（噪声图）', 'Location', 'best'); grid on;

subplot(1, 3, 2);
plot(sigma_values, ssim_filtered_sigma, 'bs-', 'LineWidth', 1.5, 'MarkerSize', 8);
xlabel('sigma_{noise}'); ylabel('SSIM');
title('SSIM vs sigma_{noise}'); grid on;

subplot(1, 3, 3);
improvement = psnr_filtered_sigma - psnr_noisy_sigma;
plot(sigma_values, improvement, 'mo-', 'LineWidth', 1.5, 'MarkerSize', 8);
xlabel('sigma_{noise}'); ylabel('PSNR 提升 (dB)');
title('滤波带来的 PSNR 提升'); grid on;

sgtitle('调优2: 噪声强度影响分析', 'FontSize', 14);

% ====================================================================
%  调优3：3D DCT 硬阈值倍率 — 任务5
%  目的：优化 task5 的硬阈值系数，平衡去噪强度与细节保留
%  流程：粗扫(0.5~5.0,步长0.5) → 细扫(最佳值±0.8,步长0.1~0.2)
% ====================================================================
fprintf('\n========================================================\n');
fprintf('  调优3: 3D DCT 硬阈值倍率\n');
fprintf('  对应: 任务5 3D DCT硬阈值去噪\n');
fprintf('========================================================\n');

% 读入彩色图像
img_color = imread(img_path);
if size(img_color, 3) ~= 3
    img_color = repmat(im2double(img_color), [1, 1, 3]);
else
    img_color = im2double(img_color);
end

sigma_noise = 30;
rng(0);
noisy_color = img_color + (sigma_noise/255) * randn(size(img_color));
noisy_color = max(0, min(1, noisy_color));

noise_std = sigma_noise / 255;
fprintf('\n噪声水平: sigma_noise = %d (归一化后 noise_std = %.4f)\n', ...
    sigma_noise, noise_std);

%% ---- 阶段1：粗扫 ----
fprintf('\n--- 阶段1: 粗扫 ---\n');
thresh_coarse = 0.5:0.5:5.0;
n_tc = length(thresh_coarse);
psnr_tc = zeros(n_tc, 1);
ssim_tc = zeros(n_tc, 1);
epr_tc  = zeros(n_tc, 1);

for k = 1:n_tc
    denoised = dct3d_denoise(noisy_color, noise_std, thresh_coarse(k), 8);
    [psnr_tc(k), ssim_tc(k), epr_tc(k)] = calc_metrics(denoised, img_color);
    fprintf('  thresh=%.1f  →  PSNR=%.4f  SSIM=%.4f  EPR=%.4f\n', ...
        thresh_coarse(k), psnr_tc(k), ssim_tc(k), epr_tc(k));
end

score_tc = compute_weighted_score(psnr_tc, ssim_tc, epr_tc, tradeoff_mode);
[~, best_tc_idx] = max(score_tc);
knee_idx_tc = find_knee_point(psnr_tc);
best_t = thresh_coarse(best_tc_idx);
knee_t  = thresh_coarse(knee_idx_tc);

fprintf('\n  ★ 综合评分最佳: thresh=%.1f  (PSNR=%.4f, SSIM=%.4f, EPR=%.4f)\n', ...
    best_t, psnr_tc(best_tc_idx), ssim_tc(best_tc_idx), epr_tc(best_tc_idx));
fprintf('  ◆ Knee-point:    thresh=%.1f  (PSNR=%.4f)\n', ...
    knee_t, psnr_tc(knee_idx_tc));

%% ---- 阶段2：细扫 ----
fprintf('\n--- 阶段2: 细扫 ---\n');
lo_t = max(0.5, best_t - 0.8);
hi_t = min(5.0, best_t + 0.8);
step_t = 0.2;
thresh_fine = round((lo_t:step_t:hi_t) * 10) / 10;
if ~ismember(best_t, thresh_fine)
    thresh_fine = sort([thresh_fine, best_t]);
end

n_tf = length(thresh_fine);
psnr_tf = zeros(n_tf, 1);
ssim_tf = zeros(n_tf, 1);
epr_tf  = zeros(n_tf, 1);

for k = 1:n_tf
    denoised = dct3d_denoise(noisy_color, noise_std, thresh_fine(k), 8);
    [psnr_tf(k), ssim_tf(k), epr_tf(k)] = calc_metrics(denoised, img_color);
    fprintf('  thresh=%.1f  →  PSNR=%.4f  SSIM=%.4f  EPR=%.4f\n', ...
        thresh_fine(k), psnr_tf(k), ssim_tf(k), epr_tf(k));
end

score_tf = compute_weighted_score(psnr_tf, ssim_tf, epr_tf, tradeoff_mode);
[~, best_tf_idx] = max(score_tf);
knee_idx_tf = find_knee_point(psnr_tf);
best_t_final = thresh_fine(best_tf_idx);
knee_t_final = thresh_fine(knee_idx_tf);

fprintf('\n  ★ 综合评分最佳: thresh=%.1f  (PSNR=%.4f, SSIM=%.4f, EPR=%.4f)\n', ...
    best_t_final, psnr_tf(best_tf_idx), ssim_tf(best_tf_idx), epr_tf(best_tf_idx));
fprintf('  ◆ Knee-point:    thresh=%.1f  (PSNR=%.4f)\n', ...
    knee_t_final, psnr_tf(knee_idx_tf));

%% ---- 绘制阈值倍率调优曲线（三指标 + knee-point） ----
figure('Name', '调优3: 阈值倍率', 'NumberTitle', 'off', ...
       'Position', [100, 100, 1100, 750]);

% 粗扫三指标
subplot(2, 2, 1);
yyaxis left;
plot(thresh_coarse, psnr_tc, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 6);
ylabel('PSNR (dB)');
yyaxis right;
plot(thresh_coarse, ssim_tc, 'r-s', 'LineWidth', 1.5, 'MarkerSize', 6); hold on;
plot(thresh_coarse, epr_tc, 'g-^', 'LineWidth', 1.5, 'MarkerSize', 6);
ylabel('SSIM / EPR');
xlabel('阈值倍率'); title('粗扫（三指标）');
legend({'PSNR', 'SSIM', 'EPR'}, 'Location', 'best');
grid on;

% 粗扫综合评分
subplot(2, 2, 2);
plot(thresh_coarse, score_tc, 'm-o', 'LineWidth', 1.5, 'MarkerSize', 8); hold on;
plot(best_t, score_tc(best_tc_idx), 'r*', 'MarkerSize', 20);
plot(knee_t, score_tc(knee_idx_tc), 'b^', 'MarkerSize', 12);
text(best_t, score_tc(best_tc_idx), sprintf(' 最佳 thresh=%.1f', best_t), ...
     'VerticalAlignment', 'bottom');
text(knee_t, score_tc(knee_idx_tc), sprintf(' Knee thresh=%.1f', knee_t), ...
     'VerticalAlignment', 'top');
xlabel('阈值倍率'); ylabel('综合评分');
title(sprintf('粗扫综合评分（%s模式）', tradeoff_mode));
grid on; legend('评分', '最佳', 'Knee-point', 'Location', 'best');

% 细扫三指标
subplot(2, 2, 3);
yyaxis left;
plot(thresh_fine, psnr_tf, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 6);
ylabel('PSNR (dB)');
yyaxis right;
plot(thresh_fine, ssim_tf, 'r-s', 'LineWidth', 1.5, 'MarkerSize', 6); hold on;
plot(thresh_fine, epr_tf, 'g-^', 'LineWidth', 1.5, 'MarkerSize', 6);
ylabel('SSIM / EPR');
xlabel('阈值倍率'); title(sprintf('细扫（步长%.1f）', step_t));
legend({'PSNR', 'SSIM', 'EPR'}, 'Location', 'best');
grid on;

% 细扫综合评分
subplot(2, 2, 4);
plot(thresh_fine, score_tf, 'm-o', 'LineWidth', 1.5, 'MarkerSize', 8); hold on;
plot(best_t_final, score_tf(best_tf_idx), 'r*', 'MarkerSize', 20);
plot(knee_t_final, score_tf(knee_idx_tf), 'b^', 'MarkerSize', 12);
text(best_t_final, score_tf(best_tf_idx), sprintf(' 最佳 thresh=%.1f', best_t_final), ...
     'VerticalAlignment', 'bottom');
text(knee_t_final, score_tf(knee_idx_tf), sprintf(' Knee thresh=%.1f', knee_t_final), ...
     'VerticalAlignment', 'top');
xlabel('阈值倍率'); ylabel('综合评分');
title(sprintf('细扫综合评分（%s模式）', tradeoff_mode));
grid on; legend('评分', '最佳', 'Knee-point', 'Location', 'best');

sgtitle(sprintf('调优3: 3D DCT 硬阈值倍率（%s模式）', tradeoff_mode), 'FontSize', 14);

%% ---- 关键位置对比图（粗扫中选三个典型阈值） ----
thresh_show = [thresh_coarse(1), knee_t, best_t];
thresh_show = unique(thresh_show);
if length(thresh_show) < 3
    thresh_show = [thresh_coarse(1), max(thresh_coarse(round(end/2)), 0.5), thresh_coarse(end)];
end
thresh_labels = {'阈值低·保留多', 'Knee平衡点', '阈值高·去噪强'};

figure('Name', '阈值倍率效果对比', 'NumberTitle', 'off', ...
       'Position', [100, 100, 1100, 800]);

for k = 1:3
    denoised = dct3d_denoise(noisy_color, noise_std, thresh_show(k), 8);
    [p, s, e] = calc_metrics(denoised, img_color);

    subplot(3, 3, k);
    imshow(denoised, []);
    title({sprintf('thresh=%.1f: %s', thresh_show(k), thresh_labels{k}), ...
           sprintf('PSNR=%.4f  SSIM=%.3f  EPR=%.3f', p, s, e)});

    % 局部放大
    subplot(3, 3, k+3);
    [H, W, ~] = size(img_color);
    cr = round([H/2-30, W/2-30, 79, 79]);
    imshow(denoised(cr(1):cr(1)+cr(3), cr(2):cr(2)+cr(4), :), []);
    title('局部放大 (80×80)');

    % 边缘检测对比（显示 sobel 边缘图，直观展示边缘保留）
    subplot(3, 3, k+6);
    edges = edge(rgb2gray(denoised), 'sobel');
    imshow(edges, []);
    title(sprintf('Sobel边缘 EPR=%.3f', e));
end

sgtitle('调优3: 不同阈值倍率效果对比', 'FontSize', 14);

% ====================================================================
%  调优4：block_size 对比
%  目的：观察块大小对去噪质量和计算效率的影响
%  流程：保持最优 threshold 倍率，扫描 block_size = 4, 8, 12, 16
% ====================================================================
fprintf('\n========================================================\n');
fprintf('  调优4: block_size 对比\n');
fprintf('  固定 threshold倍率 = %.1f\n', best_t_final);
fprintf('========================================================\n');

block_sizes = [4, 8, 12, 16];
n_bs = length(block_sizes);
psnr_block = zeros(n_bs, 1);
ssim_block = zeros(n_bs, 1);
time_block = zeros(n_bs, 1);

for k = 1:n_bs
    t_start = tic;
    denoised = dct3d_denoise(noisy_color, noise_std, best_t_final, block_sizes(k));
    time_block(k) = toc(t_start);
    [psnr_block(k), ssim_block(k), ~] = calc_metrics(denoised, img_color);
    fprintf('  block_size=%2d  →  PSNR=%.4f  SSIM=%.4f  耗时%.2fs\n', ...
        block_sizes(k), psnr_block(k), ssim_block(k), time_block(k));
end

% 选择最优 block_size
psnr_threshold = max(psnr_block) - 0.5;
viable_idx = find(psnr_block >= psnr_threshold);
[~, fastest_idx] = min(time_block(viable_idx));
best_block = block_sizes(viable_idx(fastest_idx));

fprintf('\n  ★ 推荐 block_size=%d  (PSNR=%.4f, 耗时%.2fs)\n', ...
    best_block, psnr_block(block_sizes == best_block), time_block(block_sizes == best_block));

figure('Name', '调优4: block_size 对比', 'NumberTitle', 'off', ...
       'Position', [100, 100, 1200, 400]);

subplot(1, 3, 1);
bar(block_sizes, psnr_block, 0.5, 'FaceColor', [0.5 0.7 1]); hold on;
plot(block_sizes, psnr_block, 'ro-', 'LineWidth', 1.5, 'MarkerSize', 8);
xlabel('块大小'); ylabel('PSNR (dB)');
title('block_size vs PSNR'); ylim([min(psnr_block)-0.5, max(psnr_block)+0.5]); grid on;

subplot(1, 3, 2);
bar(block_sizes, ssim_block, 0.5, 'FaceColor', [0.6 0.9 0.6]); hold on;
plot(block_sizes, ssim_block, 'ro-', 'LineWidth', 1.5, 'MarkerSize', 8);
xlabel('块大小'); ylabel('SSIM');
title('block_size vs SSIM'); grid on;

subplot(1, 3, 3);
bar(block_sizes, time_block, 0.5, 'FaceColor', [1 0.6 0.6]); hold on;
plot(block_sizes, time_block, 'ro-', 'LineWidth', 1.5, 'MarkerSize', 8);
xlabel('块大小'); ylabel('耗时 (s)');
title('block_size vs 计算时间'); grid on;

sgtitle('调优4: 3D DCT 块大小对比', 'FontSize', 14);

% ====================================================================
%  汇总报告
% ====================================================================
fprintf('\n');
fprintf('================================================================\n');
fprintf('  参数调优汇总报告\n');
fprintf('================================================================\n\n');
fprintf('  权衡模式: %s\n', tradeoff_mode);
fprintf('\n');
fprintf('  参数             推荐值         对应任务     最优指标\n');
fprintf('  ─────────────────────────────────────────────────────────────\n');
fprintf('  D0（截止频率）    %3d           任务2       PSNR=%.4f  SSIM=%.4f  EPR=%.4f\n', ...
    best_D0_final, psnr_fine(best_fine_idx), ssim_fine(best_fine_idx), epr_fine(best_fine_idx));
fprintf('  threshold 倍率    %3.1f           任务5       PSNR=%.4f  SSIM=%.4f  EPR=%.4f\n', ...
    best_t_final, psnr_tf(best_tf_idx), ssim_tf(best_tf_idx), epr_tf(best_tf_idx));
fprintf('  block_size        %2d            任务5       PSNR=%.4f  SSIM=%.4f  耗时=%.2fs\n', ...
    best_block, psnr_block(block_sizes == best_block), ssim_block(block_sizes == best_block), ...
    time_block(block_sizes == best_block));
fprintf('\n');
fprintf('  σ_noise = %d（固定，可根据需要调整）\n', sigma_noise);
fprintf('\n');
fprintf('  Knee-point 参考:\n');
fprintf('    D0: %d  (PSNR曲线上拐点，之后收益递减)\n', knee_D0_final);
fprintf('    threshold倍率: %.1f\n', knee_t_final);
fprintf('\n');
fprintf('  与空域高斯卷积对比:  PSNR=%.4f  SSIM=%.4f  EPR=%.4f\n', psnr_sp, ssim_sp, epr_sp);
fprintf('\n');
fprintf('  建议：将以上最优参数代回 experiment4.m 中的对应变量：\n');
fprintf('    第96行:  D0 = %d\n', best_D0_final);
fprintf('    第244行: threshold倍率 = %.1f\n', best_t_final);
fprintf('    第245行: block_size = %d\n', best_block);
fprintf('\n================================================================\n');
fprintf('  调优完成！\n');
fprintf('================================================================\n');

%% 保存最优参数到 .mat 文件（供 experiment4.m 自动加载使用）
optimal_D0 = best_D0_final;
optimal_threshold_mult = best_t_final;
optimal_block_size = best_block;

save('optimal_params.mat', 'optimal_D0', 'optimal_threshold_mult', 'optimal_block_size');
fprintf('\n最优参数已保存至 optimal_params.mat\n');
fprintf('运行 experiment4.m 时将自动加载并使用这些参数\n');

%% ====================================================================
%  辅助函数 1: 频域高斯低通滤波
%  对应: 任务2
% ====================================================================
function img_filtered = gaussian_lp_filter(img_noisy, ~, D0)
    % 输入:
    %   img_noisy  - 噪声图像 (double, [0,1])
    %   ~          - 保留接口兼容（原 img_ref，已不在此函数内使用）
    %   D0         - 截止频率
    % 输出:
    %   img_filtered - 滤波结果

    % 【性能优化】持久化缓存距离矩阵 D，避免每次重新计算 meshgrid
    % 图像大小在调优过程中固定，D 只与图像尺寸有关
    [M, N] = size(img_noisy);
    persistent D_cached M_cached N_cached
    if isempty(D_cached) || M_cached ~= M || N_cached ~= N
        [u, v] = meshgrid(1:N, 1:M);
        D_cached = sqrt((u - N/2).^2 + (v - M/2).^2);
        M_cached = M;
        N_cached = N;
    end
    H = exp(-(D_cached.^2) / (2 * D0^2));

    F = fft2(img_noisy);
    F_shifted = fftshift(F);
    F_filtered = F_shifted .* H;
    img_filtered = real(ifft2(ifftshift(F_filtered)));
end


%% ====================================================================
%  辅助函数 2: 3D DCT 硬阈值去噪
%  对应: 任务5（完全复用 experiment4.m 中的算法）
% ====================================================================
function denoised = dct3d_denoise(noisy, noise_std, threshold_mult, block_size)
    % 输入:
    %   noisy           - 噪声彩色图像 (double, [0,1], HxWx3)
    %   noise_std       - 像素域噪声标准差 (如 sigma_noise/255)
    %   threshold_mult  - 硬阈值倍率 (如 2.5)
    %   block_size      - DCT 块大小 (如 8)
    % 输出:
    %   denoised        - 去噪后的彩色图像

    [H, W, C] = size(noisy);
    denoised = zeros(size(noisy));
    threshold = threshold_mult * noise_std;

    % 【性能优化】预计算 DCT 变换矩阵，BLAS 矩阵乘法替代 dct() 函数调用
    DCT_mat = dctmtx(block_size);

    for i = 1:block_size:H
        for j = 1:block_size:W
            block_i_end = min(i + block_size - 1, H);
            block_j_end = min(j + block_size - 1, W);
            block = noisy(i:block_i_end, j:block_j_end, :);

            if size(block, 1) == block_size && size(block, 2) == block_size
                % 【优化】2D DCT: DCT_mat * X * DCT_mat' (BLAS 矩阵乘法)
                for c = 1:C
                    dct_block(:, :, c) = DCT_mat * block(:, :, c) * DCT_mat';
                end
                dct_block = dct(dct_block, [], 3);

                dct_block(abs(dct_block) < threshold) = 0;

                % 【优化】3D逆变换
                block_denoised = idct(dct_block, [], 3);
                for c = 1:C
                    block_denoised(:, :, c) = DCT_mat' * block_denoised(:, :, c) * DCT_mat;
                end
                denoised(i:i + block_size - 1, j:j + block_size - 1, :) = block_denoised;
            else
                denoised(i:block_i_end, j:block_j_end, :) = block;
            end
        end
    end

    denoised = max(0, min(1, denoised));
end


%% ====================================================================
%  辅助函数 3: 多指标评估 PSNR + SSIM + EPR
% ====================================================================
function [psnr_val, ssim_val, epr_val] = calc_metrics(denoised, original)
    % 同时计算三个评估指标
    %   PSNR - 峰值信噪比（像素级误差）
    %   SSIM - 结构相似性（亮度/对比度/结构）
    %   EPR  - 边缘保留度（Sobel边缘图的相关系数）

    psnr_val = psnr(denoised, original);
    ssim_val = ssim(denoised, original);

    % EPR: 提取第一通道的 Sobel 边缘，计算相关系数
    if ndims(original) == 3
        ref_ch = original(:, :, 1);
        den_ch = denoised(:, :, 1);
    else
        ref_ch = original;
        den_ch = denoised;
    end
    edges_orig = edge(ref_ch, 'sobel');
    edges_deno = edge(den_ch, 'sobel');
    epr_val = corr2(double(edges_orig), double(edges_deno));
end


%% ====================================================================
%  辅助函数 4: Knee-point 检测
%  找曲线"肘点"——到对角线的最大距离点
% ====================================================================
function knee_idx = find_knee_point(values)
    % 输入:
    %   values    - 一维数列（如 PSNR 随参数变化的序列）
    % 输出:
    %   knee_idx  - 肘点索引
    %
    % 原理: 将序列归一化到 [0,1]^2 空间后，找距离对角线最远的点
    %       该点之后曲线趋于平缓，即"收益递减"的转折点

    n = length(values);
    if n < 3
        knee_idx = round(n / 2);
        return;
    end

    % 归一化到 [0,1]
    v_min = min(values);
    v_max = max(values);
    if v_max - v_min < eps
        knee_idx = round(n / 2);
        return;
    end
    x = (0:n-1) / (n-1);
    y = (values - v_min) / (v_max - v_min);

    % 找距离对角线 (y=x) 最远的点
    d = abs(y - x) / sqrt(2);
    [~, knee_idx] = max(d);
end


%% ====================================================================
%  辅助函数 5: 加权综合评分
% ====================================================================
function [scores, best_idx] = compute_weighted_score(psnr_vals, ssim_vals, epr_vals, mode)
    % 输入:
    %   psnr_vals, ssim_vals, epr_vals - 三个指标的数列（列向量）
    %   mode    - 权衡模式: 'balance' / 'aggressive' / 'detail'
    % 输出:
    %   scores   - 综合评分
    %   best_idx - 评分最高的索引

    % 归一化到 [0,1]，防止某一指标主导
    p_norm = (psnr_vals - min(psnr_vals)) / (max(psnr_vals) - min(psnr_vals) + eps);
    s_norm = (ssim_vals - min(ssim_vals)) / (max(ssim_vals) - min(ssim_vals) + eps);
    e_norm = (epr_vals  - min(epr_vals))  / (max(epr_vals)  - min(epr_vals)  + eps);

    switch mode
        case 'aggressive'  % 强去噪：重视 PSNR 和 SSIM
            w = [0.40, 0.40, 0.20];
        case 'detail'      % 保细节：重视 EPR
            w = [0.20, 0.30, 0.50];
        otherwise          % 'balance' 默认平衡模式
            w = [0.30, 0.40, 0.30];
    end

    scores = w(1) * p_norm + w(2) * s_norm + w(3) * e_norm;
    [~, best_idx] = max(scores);
end
