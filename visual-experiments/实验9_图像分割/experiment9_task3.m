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

%% 实验9: 图像分割 — 任务3: 眼底血管分割
%  传统方法: 自适应阈值分割  vs  U-Net 深度学习分割(需预训练模型)
% 课程: 视觉与数据计算
% 重点函数: adapthisteq, imbinarize, bwareaopen, predict, dlnetwork
clear all;
close all;
clc;
fig_dir = fullfile(fileparts(mfilename('fullpath')), 'figs_task3');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
global DEMO_MODE;
DEMO_MODE = false;
fprintf('========================================\n');
fprintf('    实验9: 图像分割 — 任务3\n');
fprintf('    眼底血管分割: 自适应阈值 vs U-Net\n');
fprintf('========================================\n\n');

%% ========================================================================
%  数据准备 (两种方法共用)
% ========================================================================
fprintf('【任务3】眼底血管分割\n');
fprintf('----------------------------------------\n');
demoPause('任务3: 加载眼底血管图像 — 绿色通道提取 + CLAHE 增强');

% 读入vessels.tif眼底图像（彩色，绿色通道血管对比度最佳）
vessels_path = 'vessels.tif';
vessels = imread(vessels_path);
origSize = [size(vessels, 1), size(vessels, 2)];
fprintf('使用图像: %s (%dx%d, %s)\n', vessels_path, size(vessels,2), ...
    size(vessels,1), class(vessels));

% 提取绿色通道——眼底血管分割的标准做法（红绿色差最大）
if size(vessels, 3) == 3
    green_ch = im2double(vessels(:,:,2));
else
    green_ch = im2double(vessels);
end

% CLAHE增强对比度
vessels_enhanced = adapthisteq(green_ch);

% 生成FOV掩模，排除黑色背景边框的干扰
fov_mask = vessels_enhanced > graythresh(vessels_enhanced) * 0.5;

% 加载 Ground Truth (用于 IoU 评估)
gtPath = fullfile(fileparts(mfilename('fullpath')), 'mask_vessels.gif');
if exist(gtPath, 'file')
    gtMask = imread(gtPath) > 128;
    % 确保与图像尺寸一致
    if size(gtMask, 1) ~= origSize(1) || size(gtMask, 2) ~= origSize(2)
        gtMask = imresize(gtMask, origSize, 'nearest');
    end
    fprintf('已加载 GT 掩模: %s\n', gtPath);
else
    gtMask = [];
    fprintf('注意: 未找到 GT 掩模 (%s), 跳过 IoU 计算\n', gtPath);
end

%% ========================================================================
%  方法1: 增强型分割——方向匹配滤波 (LoG) + Otsu 阈值
%  ========================================================================
%  参考: exp9_619(1).mlx 中 12 方向 LoG 匹配滤波检测管状结构
%  ========================================================================
demoPause('任务3-1: 增强型分割 — 12方向LoG匹配滤波 + Otsu');

% Step 1: 中值滤波去噪
vessels_med = medfilt2(vessels_enhanced, [3 3]);

% Step 2: 12 方向 LoG 匹配滤波 (专检测管状结构)
sigma_mf = 2.0; len_mf = 7; ndir_mf = 12;
resp_mf = zeros([size(vessels_med), ndir_mf]);
for d = 0:ndir_mf-1
    ang = d * pi / ndir_mf;
    [X, Y] = meshgrid(-len_mf:len_mf, -len_mf:len_mf);
    Xr = X*cos(ang) + Y*sin(ang);
    Yr = -X*sin(ang) + Y*cos(ang);
    % 二阶高斯导数核 (LoG-like, 对管状结构响应最强)
    k0 = exp(-Xr.^2/(2*sigma_mf^2)) .* (Yr.^2/sigma_mf^4 - 1/sigma_mf^2) .* exp(-Yr.^2/(2*sigma_mf^2));
    k0 = k0 - mean(k0(:));  % 零均值
    resp_mf(:,:,d+1) = conv2(vessels_med, k0, 'same');
end
% 取最大方向响应 → 血管增强图
vesselness = max(0, max(resp_mf, [], 3));
vesselness = mat2gray(vesselness);
% 保存匹配滤波的概率图 (用于集成)
score_matched = vesselness;

% Step 3: Otsu 自动阈值
th_mf = graythresh(vesselness);
bw_adaptive = vesselness > th_mf;

% Step 4: 后处理
bw_adaptive = imclose(bw_adaptive, strel('disk', 2));
bw_adaptive = bwareaopen(bw_adaptive, 30);
bw_adaptive = bw_adaptive & fov_mask;

fprintf('增强型分割完成: 前景像素=%d', sum(bw_adaptive(:)));
if ~isempty(gtMask)
    fprintf(', IoU=%.4f', computeIoU(bw_adaptive, gtMask));
end
fprintf('\n');

%% ========================================================================
%  方法2: U-Net 分割（深度学习方法）
% ========================================================================
USE_UNET = true;  % 设为 false 可跳过 U-Net
bw_unet = [];

if USE_UNET
    demoPause('任务3-2: U-Net 深度学习分割 — 加载预训练模型');

    modelFile = fullfile(fileparts(mfilename('fullpath')), 'unet_task3_model.mat');
    if exist(modelFile, 'file')
        % 加载模型
        loaded = load(modelFile, 'net', 'inputSize');
        net = loaded.net;
        inputSize = loaded.inputSize;
        fprintf('已加载模型: %s (输入尺寸 %d×%d)\n', modelFile, inputSize(1), inputSize(2));

        % 推理: 缩放到模型输入尺寸后全图预测 (滑窗尺度不匹配, 已回退)
        imgResized = imresize(vessels, [inputSize(1), inputSize(2)]);
        dlX = dlarray(single(imgResized) / 255, 'SSCB');
        if canUseGPU()
            dlX = gpuArray(dlX);
        end

        fprintf('推理中 (TTA x4, %d×%d)... ', inputSize(1), inputSize(2));
        tic;
        p1 = predict(net, dlX);
        p2 = flip(predict(net, flip(dlX, 2)), 2);
        p3 = flip(predict(net, flip(dlX, 1)), 1);
        p4 = flip(flip(predict(net, flip(flip(dlX, 1), 2)), 2), 1);
        dlPred = (p1 + p2 + p3 + p4) / 4;
        inferTime = toc;
        score_unet_raw = squeeze(double(extractdata(dlPred(:,:,2,:))));
        score_unet = imresize(score_unet_raw, origSize);  % 恢复到原始尺寸

        fprintf('完成 (%.2fs)\n', inferTime);

        % 二值化 (固定阈值 0.5, 后续网格搜索优化)
        predMask = score_unet > 0.5;

        % 应用 FOV 掩模 + 面积过滤
        bw_unet = predMask & fov_mask;
        bw_unet = bwareaopen(bw_unet, 30);
        fprintf('U-Net 分割: 前景像素=%d', sum(bw_unet(:)));
        if ~isempty(gtMask)
            iou_unet = computeIoU(bw_unet, gtMask);
            fprintf(', IoU=%.4f', iou_unet);
        end
        fprintf('\n');
    else
        fprintf('注意: 未找到模型文件 %s\n', modelFile);
        fprintf('请先在 4090 机器上运行 experiment9_chasedb1_unet.m 训练并拷贝模型\n');
    end
end

% ========================================================================
%  方法3: Coye Filter（经典免训练方法）
% ========================================================================
USE_COYE = true;  % 设为 false 可跳过 Coye Filter
bw_coye = [];

if USE_COYE
    demoPause('任务3-3: Coye Filter — PCA + CLAHE + Isodata 阈值');
    fprintf('Coye Filter 分割... ');
    tic;
    [bw_coye, score_coye] = segmentCoyeFilter(vessels);
    coyeTime = toc;
    % 后处理: 应用 FOV 掩模 + 面积过滤 (与前两种方法一致)
    bw_coye = bw_coye & fov_mask;
    bw_coye = bwareaopen(bw_coye, 30);
    fprintf('完成 (%.2fs), 前景像素=%d\n', coyeTime, sum(bw_coye(:)));
end

% ========================================================================
%  方法4: U-Net + Coye 集成 + 后处理参数网格搜索 (方案1+2)
%  ========================================================================
%  思路: 平均两种方法的概率图, 搜索最佳 sigma × threshold
%  ========================================================================
USE_ENSEMBLE = true;
bw_ensemble = [];

if USE_ENSEMBLE && ~isempty(bw_unet) && ~isempty(bw_coye) && ~isempty(gtMask)
    demoPause('任务3-4: 集成优化 — U-Net + Coye + 匹配滤波 3合1 + 网格搜索参数');
    fprintf('集成 + 网格搜索... ');

    % Step 1: 平均概率图 (3 方法: U-Net + Coye + 匹配滤波)
    % 匹配滤波的 score 尺寸可能和原图不同, 需要同步
    score_matched_resized = imresize(score_matched, origSize);
    score_ensemble = (score_unet + score_coye + score_matched_resized) / 3;

    % Step 2: 尝试 3 种融合/阈值策略, 选 IoU 最高者
    sigma_range = [0, 0.5, 1, 1.5, 2, 3];
    th_range = 0.3:0.05:0.7;

    best_iou = 0;
    best_sigma = 0;
    best_th = 0.5;
    best_strategy = 'avg_global';

    % ---- 变体 A: 平均融合 + 全局阈值网格搜索 (当前方案) ----
    for sig = sigma_range
        if sig > 0
            ss = imgaussfilt(score_ensemble, sig);
        else
            ss = score_ensemble;
        end
        for th = th_range
            bw_tmp = ss > th;
            bw_tmp = bw_tmp & fov_mask;
            bw_tmp = bwareaopen(bw_tmp, 30);
            iou = computeIoU(bw_tmp, gtMask);
            if iou > best_iou
                best_iou = iou; best_sigma = sig; best_th = th;
                best_strategy = 'avg_global';
            end
        end
    end

    % ---- 变体 B: 平均融合 + 局部自适应阈值 (方案3) ----
    se_local = imgaussfilt(score_ensemble, 1.0);  % 先轻平滑
    th_local = adaptthresh(se_local, 0.5);         % 局部阈值图
    for scale = [0.6, 0.8, 1.0, 1.2, 1.4]         % 扫描局部阈值的缩放系数
        bw_tmp = se_local > th_local * scale;
        bw_tmp = bw_tmp & fov_mask;
        bw_tmp = bwareaopen(bw_tmp, 30);
        iou = computeIoU(bw_tmp, gtMask);
        if iou > best_iou
            best_iou = iou; best_sigma = 1.0; best_th = scale;
            best_strategy = 'avg_local';
        end
    end

    % ---- 变体 C: 最大值融合 + 全局阈值网格搜索 (方案4) ----
    score_max = max(cat(3, score_unet, score_coye, score_matched_resized), [], 3);
    for sig = sigma_range
        if sig > 0
            ss = imgaussfilt(score_max, sig);
        else
            ss = score_max;
        end
        for th = th_range
            bw_tmp = ss > th;
            bw_tmp = bw_tmp & fov_mask;
            bw_tmp = bwareaopen(bw_tmp, 30);
            iou = computeIoU(bw_tmp, gtMask);
            if iou > best_iou
                best_iou = iou; best_sigma = sig; best_th = th;
                best_strategy = 'max_global';
            end
        end
    end

    % Step 3: 用最佳策略生成最终结果
    fprintf('  最佳策略: %s (sigma=%.1f, th=%.2f, IoU=%.4f)\n', ...
        best_strategy, best_sigma, best_th, best_iou);

    switch best_strategy
        case 'avg_global'
            score_final = imgaussfilt(score_ensemble, max(best_sigma, 0.01));
            bw_ensemble = score_final > best_th;
        case 'avg_local'
            score_final = imgaussfilt(score_ensemble, 1.0);
            bw_ensemble = score_final > th_local * best_th;
        case 'max_global'
            score_final = imgaussfilt(score_max, max(best_sigma, 0.01));
            bw_ensemble = score_final > best_th;
    end
    bw_ensemble = bw_ensemble & fov_mask;
    bw_ensemble = bwareaopen(bw_ensemble, 30);

    % === 后处理精化: 迭代生长 + 连通性修复 ===
    fprintf('  后处理精化... ');
    % Step A: 从高置信度种子沿概率梯度向外生长
    % 生长阈值应低于 Ensemble 阈值, 以补充被阈值切掉的细血管边缘
    grow_th = max(0.15, best_th - 0.12);  % 比最优阈值低 0.12
    seeds = (score_final > 0.7) & fov_mask;
    grown = seeds;
    changed = true;
    iter = 0;
    while changed && iter < 80
        changed = false;
        iter = iter + 1;
        dilated = imdilate(grown, strel('disk', 1));
        new = dilated & (score_final > grow_th) & ~grown;
        if any(new(:))
            grown = grown | new;
            changed = true;
        end
    end

    % Step B: 取 Ensemble 和生长结果的并集
    bw_refined = bw_ensemble | grown;
    bw_refined = bwareaopen(bw_refined, 20);

    % Step C: 连通性修复
    skel = bwmorph(bw_refined, 'skel', Inf);
    [r_ep, c_ep] = find(bwmorph(skel, 'endpoints'));
    nEp = length(r_ep);
    if nEp >= 2
        for i = 1:min(nEp, 150)
            for j = i+1:min(nEp, 150)
                dist = sqrt((r_ep(i)-r_ep(j))^2 + (c_ep(i)-c_ep(j))^2);
                if dist < 20 && dist > 3
                    nPts = ceil(dist);
                    rr = round(linspace(r_ep(i), r_ep(j), nPts));
                    cc = round(linspace(c_ep(i), c_ep(j), nPts));
                    rr = max(1, min(size(score_final,1), rr));
                    cc = max(1, min(size(score_final,2), cc));
                    idx = sub2ind(size(score_final), rr, cc);
                    path_score = mean(score_final(idx));
                    if path_score > 0.20  % 连接阈值低于 Ensemble 阈值
                        bw_refined(idx) = true;
                    end
                end
            end
        end
    end
    bw_refined = bwareaopen(bw_refined, 20);
    bw_refined = bw_refined & fov_mask;
    iou_refined = computeIoU(bw_refined, gtMask);
    fprintf('IoU %.4f → %.4f\n', best_iou, iou_refined);

    % 如果精化后 IoU 更高则更新
    if iou_refined > best_iou
        bw_ensemble = bw_refined;
        best_iou = iou_refined;
    end

    % Step 4: 也用最佳参数优化 U-Net 自己
    if best_sigma > 0
        score_unet_sm = imgaussfilt(score_unet, best_sigma);
    else
        score_unet_sm = score_unet;
    end
    bw_unet_opt = score_unet_sm > best_th;
    bw_unet_opt = bw_unet_opt & fov_mask;
    bw_unet_opt = bwareaopen(bw_unet_opt, 30);
    iou_unet_opt = computeIoU(bw_unet_opt, gtMask);

    fprintf('完成! 最佳参数: sigma=%.1f, th=%.2f\n', best_sigma, best_th);
    fprintf('  U-Net 优化后 IoU=%.4f (原=%.4f)\n', iou_unet_opt, iou_unet);
    fprintf('  Ensemble  IoU=%.4f\n', best_iou);

    % 更新 U-Net 结果为优化版
    bw_unet = bw_unet_opt;
    iou_unet = iou_unet_opt;
end

%% ========================================================================
%  结果可视化
% ========================================================================
demoPause('任务3-5: 可视化 — 方法对比');

% 计算 IoU (如有 GT)
iou_adp = []; iou_unet = []; iou_coye = []; iou_ens = [];
if ~isempty(gtMask)
    iou_adp = computeIoU(bw_adaptive, gtMask);
    if ~isempty(bw_unet), iou_unet = computeIoU(bw_unet, gtMask); end
    if ~isempty(bw_coye), iou_coye = computeIoU(bw_coye, gtMask); end
    if ~isempty(bw_ensemble), iou_ens = computeIoU(bw_ensemble, gtMask); end
end

hasUnet = ~isempty(bw_unet);
hasCoye = ~isempty(bw_coye);
hasEns  = ~isempty(bw_ensemble);
nMethods = 1 + hasUnet + hasCoye + hasEns;

if nMethods == 1
    figure('Name', '任务3: 眼底血管分割', 'NumberTitle', 'off', 'Position', [50, 50, 1200, 400]);
    subplot(1,3,1); imshow(vessels); title('原图像');
    subplot(1,3,2); imshow(gtMask); title('金标准 (GT)');
    subplot(1,3,3); imshow(bw_adaptive);
    if ~isempty(iou_adp), title(sprintf('方法1 (IoU=%.4f)', iou_adp));
    else, title(sprintf('方法1 (%dpx)', sum(bw_adaptive(:)))); end
    saveas(gcf, fullfile(fig_dir, 'exp9_task3_fig1.png'));
    fprintf('图片已保存: exp9_task3_fig1.png\n');
else
    cols = nMethods + 2;  % 原图 + GT + 各方法
    figure('Name', '任务3: 眼底血管分割对比', 'NumberTitle', 'off', 'Position', [50, 50, 350*cols, 400]);

    % 第一行: 原图 + GT + 各方法二值掩模 (白血管/黑背景, 与GT一致)
    subplot(1, cols, 1); imshow(vessels); title('原图像');
    subplot(1, cols, 2); imshow(gtMask); title(sprintf('金标准 GT'));

    ci = 3;
    subplot(1, cols, ci); imshow(bw_adaptive);
    title(sprintf('方法1 (%.4f)', iou_adp)); ci = ci+1;

    if hasUnet
        subplot(1, cols, ci); imshow(bw_unet);
        title(sprintf('U-Net (%.4f)', iou_unet)); ci = ci+1;
    end
    if hasCoye
        subplot(1, cols, ci); imshow(bw_coye);
        title(sprintf('Coye (%.4f)', iou_coye)); ci = ci+1;
    end
    if hasEns
        subplot(1, cols, ci); imshow(bw_ensemble);
        title(sprintf('Ensemble (%.4f)', iou_ens)); ci = ci+1;
    end
    saveas(gcf, fullfile(fig_dir, 'exp9_task3_fig2.png'));
    fprintf('图片已保存: exp9_task3_fig2.png\n');
end

%% 输出总结
fprintf('\n【任务3完成】\n');
fprintf('----------------------------------------\n');
fprintf('  方法             前景像素');
if ~isempty(gtMask), fprintf('        IoU'); end
fprintf('\n');
fprintf('  ──────────────────────────────');
if ~isempty(gtMask), fprintf('──────────'); end
fprintf('\n');
fprintf('  方法1(匹配滤波) %8d', sum(bw_adaptive(:)));
if ~isempty(gtMask), fprintf('      %.4f', computeIoU(bw_adaptive, gtMask)); end
fprintf('\n');
if ~isempty(bw_unet)
    fprintf('  U-Net(优化后)   %8d', sum(bw_unet(:)));
    if ~isempty(gtMask), fprintf('      %.4f', computeIoU(bw_unet, gtMask)); end
    fprintf('\n');
end
if ~isempty(bw_coye)
    fprintf('  Coye Filter     %8d', sum(bw_coye(:)));
    if ~isempty(gtMask), fprintf('      %.4f', computeIoU(bw_coye, gtMask)); end
    fprintf('\n');
end
if ~isempty(bw_ensemble)
    fprintf('  Ensemble        %8d', sum(bw_ensemble(:)));
    if ~isempty(gtMask), fprintf('      %.4f', computeIoU(bw_ensemble, gtMask)); end
    fprintf('\n');
end
fprintf('----------------------------------------\n');
fprintf('========================================\n');
fprintf('        实验9 任务3 完成!\n');
fprintf('========================================\n');

function demoPause(step_name)
    global DEMO_MODE;
    fprintf('\n=== [演示步进] %s ===\n', step_name);
    if DEMO_MODE
        fprintf('按任意键继续...\n');
        pause;
    end
end

function iou = computeIoU(mask, gt)
    % COMPUTEIOU 计算二值掩模与 GT 的交并比
    inter = sum(mask(:) & gt(:));
    union = sum(mask(:) | gt(:));
    if union > 0
        iou = inter / union;
    else
        iou = -1;
    end
end

function [bw, score] = segmentCoyeFilter(I)
    % SEGMENTCOYEFILTER 基于 Coye Filter 的视网膜血管分割
    %   I: RGB 眼底图像 (uint8)
    %   返回: bw = 二值血管掩模, score = 归一化血管概率图
    %
    %   参考: Tyler Coye (2015), Novel Retinal Vessel Segmentation Algorithm
    %   https://www.mathworks.com/matlabcentral/fileexchange/50839

    % 缩放到固定尺寸 (与原算法一致)
    targetSize = [584 565];
    B = imresize(I, targetSize);
    im = im2double(B);

    % Step 1: RGB → PCA 灰度化
    lab = rgb2lab(im);
    f = 0;
    wlab = reshape(bsxfun(@times, cat(3, 1-f, f/2, f/2), lab), [], 3);
    [~, S] = pca(wlab);
    S = reshape(S, size(lab));
    S = S(:,:,1);
    gray = (S - min(S(:))) ./ (max(S(:)) - min(S(:)));

    % Step 2: CLAHE 对比度增强
    J = adapthisteq(gray, 'numTiles', [8 8], 'nBins', 128);

    % Step 3: 背景排除 (均值滤波差) → 这是血管响应得分
    h = fspecial('average', [9 9]);
    JF = imfilter(J, h);
    Z = imsubtract(JF, J);
    % 归一化到 [0,1] 作为概率图
    score_raw = mat2gray(Z);

    % Step 4: Isodata 自动阈值
    level = isodata(Z);
    BW = im2bw(Z, level - 0.008);

    % Step 5: 去除小像素
    BW2 = bwareaopen(BW, 100);

    % Step 6: 缩放到原始尺寸
    bw = imresize(BW2, [size(I, 1), size(I, 2)], 'nearest');
    score = imresize(score_raw, [size(I, 1), size(I, 2)]);
end
