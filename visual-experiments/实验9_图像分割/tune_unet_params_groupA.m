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

%% ========================================================================
%  tune_unet_params_groupA.m — U-Net 后处理参数调优 (组A)
%  ========================================================================
%  功能: 网格搜索 sigma(高斯平滑) × threshold(二值化) 的最佳组合
%  流程: 训练/加载模型 → 推理得 score_map → 网格搜索 → 热力图 → 最佳参数
%  特点: 只需训练/推理一次，扫描 sigma+threshold 无需重训，极快
%  ========================================================================

clear; close all; clc;
fprintf('====================================================\n');
fprintf('  组A调优: sigma(高斯平滑) × threshold(二值化)\n');
fprintf('====================================================\n\n');

%% ========================================================================
%  === 0. 依赖: 确保已训练好模型 (调用 task2_partB_unet 的训练部分)
%  ========================================================================

% 检查模型文件是否存在
ROOT_DIR = fileparts(mfilename('fullpath'));
model_path = fullfile(ROOT_DIR, 'unet_cnv_model.mat');

if ~exist(model_path, 'file')
    fprintf('⚠️  未找到模型文件，先执行训练...\n');
    run(fullfile(ROOT_DIR, 'task2_partB_unet.m'));
    fprintf('\n训练完成，开始调优...\n\n');
else
    fprintf('✓ 模型已存在: %s\n', model_path);
end

%% ========================================================================
%  === 1. 加载数据 + 模型 + 推理得 score_map ==============================
%  ========================================================================

fprintf('【1/4】加载数据...\n');
cnv_path = fullfile(ROOT_DIR, 'cnv.png');
cnv = imread(cnv_path);
if size(cnv, 3) == 3
    gray = im2double(rgb2gray(cnv));
else
    gray = im2double(cnv);
end
img_u8 = im2uint8(gray);

gt_path = fullfile(ROOT_DIR, 'mask_cnv.png');
gtMask = imread(gt_path);
if size(gtMask, 3) == 3
    gtMask = rgb2gray(gtMask);
end
gtMask = gtMask > 0;
fprintf('  OCT: %dx%d | GT: %d CNV px (全图)\n', ...
    size(gray,2), size(gray,1), sum(gtMask(:)));

fprintf('【2/4】加载模型 + 推理 score_map...\n');
load(model_path, 'net');

% 滑窗推理
STRIDE = 32;
PATCH_SIZE = 64;
[h, w] = size(img_u8);
score_map = zeros(h, w);
weight_map = zeros(h, w);

for r = 1:STRIDE:h-PATCH_SIZE+1
    for c = 1:STRIDE:w-PATCH_SIZE+1
        patch = single(img_u8(r:r+PATCH_SIZE-1, c:c+PATCH_SIZE-1));
        p = predict(net, patch);
        s = p(:,:,2);
        score_map(r:r+PATCH_SIZE-1, c:c+PATCH_SIZE-1) = ...
            score_map(r:r+PATCH_SIZE-1, c:c+PATCH_SIZE-1) + s;
        weight_map(r:r+PATCH_SIZE-1, c:c+PATCH_SIZE-1) = ...
            weight_map(r:r+PATCH_SIZE-1, c:c+PATCH_SIZE-1) + 1;
    end
end
weight_map(weight_map == 0) = 1;
score_map = score_map ./ weight_map;
fprintf('  推理完成 ✓\n');

%% ========================================================================
%  === 2. 网格搜索: sigma × threshold =====================================
%  ========================================================================

fprintf('\n【3/4】网格搜索 sigma × threshold...\n');

% ---- 粗扫范围 ----
sigma_coarse  = [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 7];
thresh_coarse = [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9];

% ---- 粗扫 ----
fprintf('  粗扫: %d sigma × %d threshold = %d 组合\n', ...
    length(sigma_coarse), length(thresh_coarse), ...
    length(sigma_coarse) * length(thresh_coarse));

iou_grid = zeros(length(sigma_coarse), length(thresh_coarse));
tic;
for i = 1:length(sigma_coarse)
    sig = sigma_coarse(i);
    if sig > 0
        s_smooth = imgaussfilt(score_map, sig);
    else
        s_smooth = score_map;  % sigma=0 = 不平滑
    end
    for j = 1:length(thresh_coarse)
        bw = s_smooth > thresh_coarse(j);
        inter = sum(bw(:) & gtMask(:));
        union = sum(bw(:) | gtMask(:));
        iou_grid(i, j) = inter / union;
    end
end
toc;
fprintf('  粗扫完成 ✓\n');

% ---- 找到粗扫最佳 ----
[max_iou_coarse, idx] = max(iou_grid(:));
[best_sigma_idx, best_thresh_idx] = ind2sub(size(iou_grid), idx);
best_sigma_c = sigma_coarse(best_sigma_idx);
best_thresh_c = thresh_coarse(best_thresh_idx);
fprintf('\n  粗扫最佳: sigma=%.2f, threshold=%.2f → IoU=%.4f\n', ...
    best_sigma_c, best_thresh_c, max_iou_coarse);

% ---- 细扫范围 (在粗扫最佳点附近加密) ----
sigma_fine  = linspace(max(0, best_sigma_c-1), best_sigma_c+1, 9);
thresh_fine = linspace(max(0.05, best_thresh_c-0.1), ...
    min(0.95, best_thresh_c+0.1), 9);

fprintf('\n  细扫: %d sigma × %d threshold = %d 组合\n', ...
    length(sigma_fine), length(thresh_fine), ...
    length(sigma_fine) * length(thresh_fine));

iou_fine = zeros(length(sigma_fine), length(thresh_fine));
tic;
for i = 1:length(sigma_fine)
    sig = sigma_fine(i);
    if sig > 0.01
        s_smooth = imgaussfilt(score_map, sig);
    else
        s_smooth = score_map;
    end
    for j = 1:length(thresh_fine)
        bw = s_smooth > thresh_fine(j);
        inter = sum(bw(:) & gtMask(:));
        union = sum(bw(:) | gtMask(:));
        iou_fine(i, j) = inter / union;
    end
end
toc;
fprintf('  细扫完成 ✓\n');

% ---- 细扫最佳 ----
[max_iou_fine, idx_f] = max(iou_fine(:));
[best_sigma_f_idx, best_thresh_f_idx] = ind2sub(size(iou_fine), idx_f);
best_sigma   = sigma_fine(best_sigma_f_idx);
best_thresh  = thresh_fine(best_thresh_f_idx);

%% ========================================================================
%  === 3. 输出结果 ========================================================
%  ========================================================================

fprintf('\n【4/4】调优结果\n');
fprintf('====================================\n');
fprintf('  最佳 sigma     = %.2f\n', best_sigma);
fprintf('  最佳 threshold = %.2f\n', best_thresh);
fprintf('  最佳 IoU       = %.4f\n', max_iou_fine);
fprintf('  (粗扫最佳 IoU  = %.4f)\n', max_iou_coarse);
fprintf('====================================\n');

%% ========================================================================
%  === 4. 热力图可视化 ====================================================
%  ========================================================================

% 粗扫热力图
figure('Name', '组A调优: 粗扫', 'NumberTitle', 'off', ...
    'Position', [100 100 600 500]);
imagesc(thresh_coarse, sigma_coarse, iou_grid);
colorbar; colormap(jet);
xlabel('Threshold'); ylabel('Sigma');
title(sprintf('粗扫: 最佳 IoU=%.4f (σ=%.1f, th=%.2f)', ...
    max_iou_coarse, best_sigma_c, best_thresh_c));
set(gca, 'YDir', 'normal');
hold on;
plot(best_thresh_c, best_sigma_c, 'k*', 'MarkerSize', 15, 'LineWidth', 2);
hold off;

% 细扫热力图
figure('Name', '组A调优: 细扫', 'NumberTitle', 'off', ...
    'Position', [100 100 600 500]);
imagesc(thresh_fine, sigma_fine, iou_fine);
colorbar; colormap(jet);
xlabel('Threshold'); ylabel('Sigma');
title(sprintf('细扫: 最佳 IoU=%.4f (σ=%.2f, th=%.2f)', ...
    max_iou_fine, best_sigma, best_thresh));
set(gca, 'YDir', 'normal');
hold on;
plot(best_thresh, best_sigma, 'k*', 'MarkerSize', 15, 'LineWidth', 2);
hold off;

% 最佳结果可视化
bw_best = imgaussfilt(score_map, best_sigma) > best_thresh;
figure('Name', '最佳参数分割结果', 'NumberTitle', 'off', ...
    'Position', [100 100 1400 400]);

subplot(1,4,1); imshow(gray, []); title('OCT原图像');
subplot(1,4,2); imshow(gtMask); title('金标准 GT');
subplot(1,4,3);
imshow(labeloverlay(gray, bw_best, 'Transparency', 0.6));
title(sprintf('最佳分割 (IoU=%.4f)', max_iou_fine));

% 对比：原始阈值 0.5 vs 最佳
subplot(1,4,4);
bw_orig = score_map > 0.5;
overlay = imfuse(bw_orig, bw_best, 'falsecolor', ...
    'Scaling', 'joint', 'ColorChannels', [1 2 0]);
imshow(overlay);
title('青色=原始0.5  |  红色=最佳参数');

fprintf('\n✅ 组A调优完成!\n');
fprintf('   最佳参数已保存到 best_params_groupA.mat\n');
fprintf('   主程序 experiment9.m 将自动加载使用\n');

% 保存最佳参数供主程序调用
save(fullfile(ROOT_DIR, 'best_params_groupA.mat'), 'best_sigma', 'best_thresh', 'max_iou_fine');
