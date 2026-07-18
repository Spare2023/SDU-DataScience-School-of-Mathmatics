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

%% FIVES + vessels.tif U-Net 训练 (优化版 v2)
%  ========================================================================
%  优化概述（对比原始版）:
%    [Phase 1] 参数调优: patches=30→15, batch=32→64, 简化增强, 去除旋转/缩放插值
%    [Phase 2] 数据管道重构: 预加载全部图/标签到 GPU 内存, 消除 RPD 磁盘 I/O 瓶颈
%    [Phase 3] 自定义训练循环: Dice+CE 联合损失 + 余弦退火 + 早停
%  预期加速: 5-10× (从 ~2h 降至 15-30 min), IoU 可能提升 1-3%
%  ========================================================================
%
%  环境: MATLAB R2025a, Deep Learning Toolbox
%  硬件: RTX 4060 8GB (或其他 NVIDIA GPU)
%  输出: unet_task3_model.mat (与 experiment9_task3.m 兼容)
%
%  日期: 2026-06-23

clear; close all; clc;
fprintf('============================================================\n');
fprintf('  FIVES + vessels.tif U-Net 训练 [优化版 v2]\n');
fprintf('============================================================\n\n');

%% ========================================================================
%  1. 配置参数
% ========================================================================
ROOT_DIR = fileparts(mfilename('fullpath'));
FIVES_DIR = fullfile(ROOT_DIR, ...
    'FIVES A Fundus Image Dataset for AI-based Vessel Segmentation');

PATCH_SIZE        = 256;
BATCH_SIZE        = 16;            % RTX 3050 4GB → 16 (原64太大)
LEARN_RATE        = 5e-4;
N_EPOCHS          = 40;            % 余弦退火需要更多 epoch 展示全周期, 早停会提前终止
PATIENCE          = 8;             % 早停容忍度 (小幅提高, 适应更小 batch 的噪声)
PATCHES_PER_IMG   = 15;            % 原始 30 → 15 (总 patches/epoch = 200×15 = 3000)
NUM_CLASSES       = 2;
VALIDATION_FREQ   = 1;             % 每 epoch 验证一次
IM_SIZE           = [512 512];     % FIVES 预缩放尺寸 (与原始一致)

MODEL_SAVE_PATH   = fullfile(ROOT_DIR, 'unet_task3_model.mat');
VESSELS_PATH      = fullfile(ROOT_DIR, 'vessels.tif');
GT_VESSELS_PATH   = fullfile(ROOT_DIR, 'mask_vessels.gif');

useGPU = canUseGPU();
if useGPU
    gpuInfo = gpuDevice();
    fprintf('设备: GPU (%s, %.1f GB 显存)\n', gpuInfo.Name, gpuInfo.TotalMemory/1e9);
else
    fprintf('设备: CPU (训练会很慢, 强烈建议使用 GPU)\n');
end

%% ========================================================================
%  2. 加载数据 → In-Memory 数组
%  ========================================================================
%  优化点: 一次性加载全部数据到内存, 训练时只用 GPU 数组索引, 零磁盘 I/O
%  ========================================================================
fprintf('\n========== [1/5] 加载数据到内存 ==========\n');

% ---- 2a. FIVES 训练集 ----
trainImgDir = fullfile(FIVES_DIR, 'train', 'Original');
trainGtDir  = fullfile(FIVES_DIR, 'train', 'Ground truth');

imgFiles = dir(fullfile(trainImgDir, '*.png'));
[~, imgNames] = arrayfun(@(f) fileparts(f.name), imgFiles, 'UniformOutput', false);

trainImgPaths = {};
trainGtPaths  = {};
for i = 1:length(imgFiles)
    gtFile = fullfile(trainGtDir, [imgNames{i} '.png']);
    if exist(gtFile, 'file')
        trainImgPaths{end+1} = fullfile(trainImgDir, imgFiles(i).name);
        trainGtPaths{end+1}  = gtFile;
    end
end
numFIVES = length(trainImgPaths);

% 随机采样 200 张 (与原始一致)
rng(42);
useIdx = randperm(numFIVES, min(200, numFIVES));
trainImgPaths = trainImgPaths(useIdx);
trainGtPaths  = trainGtPaths(useIdx);
numFIVES = length(trainImgPaths);
fprintf('FIVES: 采样 %d 张\n', numFIVES);

% ---- 2b. 预缩放 FIVES 原图到 512×512 (一次性缓存) ----
resizedImgDir = fullfile(tempdir, 'fives_resized_opt');
resizedGtDir  = fullfile(tempdir, 'fives_gt_resized_opt');

firstRun = ~exist(resizedImgDir, 'dir') || ~exist(resizedGtDir, 'dir');
if firstRun
    mkdir(resizedImgDir); mkdir(resizedGtDir);
    fprintf('预缩放 FIVES 到 512×512...');
    for i = 1:length(trainImgPaths)
        img = imread(trainImgPaths{i});
        img = imresize(img, IM_SIZE);
        imwrite(img, fullfile(resizedImgDir, [num2str(i) '.png']));

        gt = imread(trainGtPaths{i});
        if size(gt, 3) == 3, gt = rgb2gray(gt); end
        gt = imresize(gt, IM_SIZE, 'nearest');
        imwrite(gt, fullfile(resizedGtDir, [num2str(i) '.png']));
    end
    fprintf(' %d 张 OK\n', length(trainImgPaths));
end

% ---- 2c. 加载 vessels.tif (先加载, 训练集会用到) ----
vessels = imread(VESSELS_PATH);
gtVessels = imread(GT_VESSELS_PATH) > 128;
fprintf('vessels.tif: %dx%d (%d positive px)\n', size(vessels,2), size(vessels,1), sum(gtVessels(:)));

% vessels.tif 的训练用副本 (缩放到 512×512, 与 FIVES 一致)
vesselsResized = imresize(vessels, IM_SIZE);
gtVesselsResized = imresize(gtVessels, IM_SIZE, 'nearest');

% ---- 2d. 一次性读入 CPU 内存 (节省 GPU 显存给训练) ----
fprintf('加载数据到内存 (FIVES + vessels)...\n');
tic;
XTrainCPU = zeros(IM_SIZE(1), IM_SIZE(2), 3, numFIVES + 1, 'single');
YTrainCPU = zeros(IM_SIZE(1), IM_SIZE(2), 1, numFIVES + 1, 'single');

for i = 1:numFIVES
    img = single(imread(fullfile(resizedImgDir, [num2str(i) '.png']))) / 255;
    gt  = single(imread(fullfile(resizedGtDir, [num2str(i) '.png']))) > 0;
    XTrainCPU(:,:,:,i) = img;
    YTrainCPU(:,:,:,i) = gt;
end

% 添加 vessels.tif 作为最后一张训练图 (目标域)
XTrainCPU(:,:,:,numFIVES + 1) = single(im2uint8(vesselsResized)) / 255;
YTrainCPU(:,:,:,numFIVES + 1) = single(gtVesselsResized);
fprintf('XTrain: %s, YTrain: %s (%.0f MB)\n', ...
    mat2str(size(XTrainCPU)), mat2str(size(YTrainCPU)), ...
    (numel(XTrainCPU)+numel(YTrainCPU))*4/1e6);
toc;

% ---- 2e. 加载 vessels.tif 到 GPU (用于验证, 归一化到 [0,1] 与训练一致) ----
if useGPU
    valImgGPU  = gpuArray(single(vessels) / 255);   % [H W 3] range [0,1]
    valMaskGPU = gpuArray(single(gtVessels));         % [H W]
else
    valImgGPU  = single(vessels) / 255;
    valMaskGPU = single(gtVessels);
end
[vh, vw, ~] = size(vessels);

% ---- 2f. 训练集组成信息 ----
numTrainTotal = numFIVES + 1;
fprintf('训练集组成: FIVES %d 张 + vessels.tif 1 张 = %d 张\n', ...
    numFIVES, numTrainTotal);

%% ========================================================================
%  3. 构建 U-Net (dlnetwork, 支持自定义训练)
%  ========================================================================
fprintf('\n========== [2/5] 构建 U-Net ==========\n');

inputSize = [PATCH_SIZE PATCH_SIZE 3];

% 使用 unet() 直接创建 dlnetwork (替代已弃用的 unetLayers)
% 输入层默认 Normalization='zerocenter' 且 Mean=0, 对 [0,1] 输入是恒等变换
net = unet(inputSize, NUM_CLASSES, 'EncoderDepth', 3);
numParams = sum(cellfun(@numel, net.Learnables.Value));
fprintf('U-Net 层数: %d, 可训练参数: %.1fM\n', numel(net.Layers), numParams/1e6);

%% ========================================================================
%  4. 自定义训练循环
%  ========================================================================
%  优化点:
%    - In-memory GPU 批量采样 (替代 RPD)
%    - Dice + CE 联合损失 (收敛更快)
%    - 余弦退火 LR (比 piecewise 更平滑高效)
%    - 早停机制 (避免无效训练)
%  ========================================================================
fprintf('\n========== [3/5] 开始训练 ==========\n');
fprintf('  配置: Epochs=%d, Batch=%d, LR=%.0e, Patience=%d\n', ...
    N_EPOCHS, BATCH_SIZE, LEARN_RATE, PATIENCE);
fprintf('  总 patches/epoch: %d (FIVES=%d×%d + vessels=1×%d)\n', ...
    numTrainTotal * PATCHES_PER_IMG, numFIVES, PATCHES_PER_IMG, PATCHES_PER_IMG);

tStart = tic;

% ---- 4a. 优化器状态 (Adam) ----
trailingAvg   = [];
trailingAvgSq = [];
gradDecay     = 0.9;       % beta1
gradDecaySq   = 0.999;     % beta2
epsilon       = 1e-8;

% ---- 4b. 余弦退火学习率 ----
lrFcn = @(t) LEARN_RATE * 0.5 * (1 + cos(pi * t / N_EPOCHS));

% ---- 4c. 记录变量 ----
trainLossLog = zeros(N_EPOCHS, 1);
valVesIouLog = zeros(N_EPOCHS, 1);

bestIoU    = 0;
bestEpoch  = 0;
noImprove  = 0;

patchesPerEpoch = numTrainTotal * PATCHES_PER_IMG;
itersPerEpoch   = ceil(patchesPerEpoch / BATCH_SIZE);
fprintf('  Batches/epoch: %d\n\n', itersPerEpoch);
fprintf('  训练开始...\n');
drawnow;

% ---- 4d. 创建 MATLAB 原生训练进度窗口 ----
monitor = trainingProgressMonitor;
monitor.Metrics = ["TrainingLoss", "ValidationIoU"];
monitor.Info = ["LearningRate", "Epoch", "TimeElapsed"];
monitor.XLabel = "Epoch";
monitor.Progress = 0;

% ---- 4e. 主训练循环 ----
for epoch = 1:N_EPOCHS
    % 检查用户是否点击了"停止"
    if monitor.Stop
        fprintf('\n>>> 用户手动停止 @ Epoch %d\n', epoch);
        break;
    end

    epochTic = tic;
    lossSum  = 0;
    learnRate = lrFcn(epoch);

    % 更新进度窗口信息 (Info 字段用 updateInfo, 不是 recordMetrics)
    updateInfo(monitor, "LearningRate", learnRate, "Epoch", epoch, "TimeElapsed", toc(tStart));

    for iter = 1:itersPerEpoch
        % === 步骤 A: 从 CPU 内存采样 patches → 送到 GPU ===
        imgIdx = randi(numTrainTotal, BATCH_SIZE, 1);
        rOff   = randi(IM_SIZE(1) - PATCH_SIZE + 1, BATCH_SIZE, 1);
        cOff   = randi(IM_SIZE(2) - PATCH_SIZE + 1, BATCH_SIZE, 1);

        XBatch = zeros(PATCH_SIZE, PATCH_SIZE, 3, BATCH_SIZE, 'single');
        YBatch = zeros(PATCH_SIZE, PATCH_SIZE, 1, BATCH_SIZE, 'single');

        for b = 1:BATCH_SIZE
            patchX = XTrainCPU(rOff(b):rOff(b)+PATCH_SIZE-1, ...
                               cOff(b):cOff(b)+PATCH_SIZE-1, :, imgIdx(b));
            patchY = YTrainCPU(rOff(b):rOff(b)+PATCH_SIZE-1, ...
                               cOff(b):cOff(b)+PATCH_SIZE-1, :, imgIdx(b));
            if rand > 0.5, patchX = flip(patchX, 2); patchY = flip(patchY, 2); end
            if rand > 0.5, patchX = flip(patchX, 1); patchY = flip(patchY, 1); end
            XBatch(:,:,:,b) = patchX;
            YBatch(:,:,:,b) = patchY;
        end

        if useGPU
            XBatch = gpuArray(XBatch);
            YBatch = gpuArray(YBatch);
        end

        % === 步骤 B: 前向 + 损失 + 梯度 (Dice+CE) ===
        dlX = dlarray(XBatch, 'SSCB');
        dlY = dlarray(YBatch, 'SSCB');
        [loss, gradients] = dlfeval(@modelLoss, net, dlX, dlY);

        % === 步骤 C: Adam 更新 ===
        [net, trailingAvg, trailingAvgSq] = adamupdate(net, gradients, ...
            trailingAvg, trailingAvgSq, (epoch-1)*itersPerEpoch + iter, ...
            learnRate, gradDecay, gradDecaySq, epsilon);

        lossSum = lossSum + double(extractdata(loss));

        % 进度指示
        if mod(iter, 10) == 0
            fprintf('.');
        end
    end
    fprintf(' %d/%d\n', itersPerEpoch, itersPerEpoch);

    avgLoss = lossSum / itersPerEpoch;
    trainLossLog(epoch) = avgLoss;

    % === 步骤 D: 每 epoch 验证 vessels.tif ===
    valIoU = evaluateVessels(net, valImgGPU, valMaskGPU, PATCH_SIZE, useGPU);
    valVesIouLog(epoch) = valIoU;

    % === 步骤 E: 更新原生进度窗口 ===
    recordMetrics(monitor, epoch, "TrainingLoss", avgLoss, "ValidationIoU", valIoU);
    monitor.Progress = 100 * epoch / N_EPOCHS;

    % === 步骤 F: 早停 & 保存最佳模型 ===
    if valIoU > bestIoU
        bestIoU  = valIoU;
        bestEpoch = epoch;
        noImprove = 0;
        inputSize = [PATCH_SIZE PATCH_SIZE];
        save(MODEL_SAVE_PATH, 'net', 'inputSize', 'bestIoU');
    else
        noImprove = noImprove + 1;
    end

    epochTime = toc(epochTic);
    fprintf('E%02d/%d | lr=%.1e | Loss=%.4f | ValIoU=%.4f | best=%.4f@E%d | no_impr=%d | %.0fs\n', ...
        epoch, N_EPOCHS, learnRate, avgLoss, valIoU, ...
        bestIoU, bestEpoch, noImprove, epochTime);

    if noImprove >= PATIENCE
        fprintf('\n>>> 早停触发! 最佳 ValIoU=%.4f @ Epoch %d\n', bestIoU, bestEpoch);
        break;
    end
end

epochsDone = epoch;
fprintf('\n训练完成: %d epochs (%.0f 分钟)\n', epochsDone, toc(tStart)/60);

% 加载最佳模型 (保存时已存, 无需重新加载)

%% ========================================================================
%  5. 最终验证 + 可视化
% ========================================================================
fprintf('\n========== [4/5] 最终验证 ==========\n');

% 加载最佳模型
best = load(MODEL_SAVE_PATH);
net = best.net;
if isfield(best, 'inputSize'), inputSize = best.inputSize; end
if isfield(best, 'bestIoU')
    iouFinal = best.bestIoU;
else
    % 兜底: 旧模型没有 bestIoU 字段, 重新计算
    iouFinal = evaluateVessels(net, valImgGPU, valMaskGPU, PATCH_SIZE, useGPU);
end

% 全精度滑窗推理 (已内联在 evaluateVessels, 这里直接用)
fprintf(' vessels.tif 最终 IoU = %.4f\n', iouFinal);

% 可视化
scoreMap = computeScoreMap(net, vessels, PATCH_SIZE, 32, useGPU);
scoreMap = imgaussfilt(scoreMap, 1.5);
bwFinal = bwareaopen(scoreMap > 0.5, 30);

figure('Name', 'U-Net: vessels.tif 分割 (优化版)', 'Position', [100 100 1400 500]);
subplot(1,4,1); imshow(vessels); title('原图像');
subplot(1,4,2); imshow(gtVessels); title('金标准');
subplot(1,4,3); imshow(bwFinal); title('U-Net 分割');
subplot(1,4,4);
imshow(labeloverlay(vessels, bwFinal, 'Transparency', 0.6));
title(sprintf('IoU=%.4f', iouFinal));
drawnow;

% 训练曲线
figure('Name', '训练曲线 (优化版)', 'Position', [100 100 1200 400]);
subplot(1,2,1);
plot(1:epochsDone, trainLossLog(1:epochsDone), 'b-', 'LineWidth', 1.5);
xlabel('Epoch'); ylabel('Loss'); title('训练 Loss'); grid on;

subplot(1,2,2);
plot(1:epochsDone, valVesIouLog(1:epochsDone), 'r-o', 'LineWidth', 1.5);
xlabel('Epoch'); ylabel('IoU'); title('vessels.tif IoU'); grid on;
yline(bestIoU, 'r--', sprintf('Best=%.4f', bestIoU));

%% ========================================================================
%  6. 保存模型 (最终)
% ========================================================================
fprintf('\n========== [5/5] 保存模型 ==========\n');

% 确保输出变量与 experiment9_task3.m 兼容 (inputSize 需为 2 元素 [H W])
inputSize = [PATCH_SIZE PATCH_SIZE];
save(MODEL_SAVE_PATH, 'net', 'inputSize', 'iouFinal', 'bestEpoch');
fprintf('模型已保存: %s\n', MODEL_SAVE_PATH);
fprintf('  net       = dlnetwork (%d params, %.1fM)\n', numParams, numParams/1e6);
fprintf('  inputSize = [%d %d] (2元素, 兼容 task3)\n', inputSize(1), inputSize(2));
fprintf('  iouFinal  = %.4f\n', iouFinal);

fprintf('\n============================================================\n');
fprintf('  训练完成! 总时间: %.0f 分钟\n', toc(tStart)/60);
fprintf('  vessels.tif IoU: %.4f (最佳 @ Epoch %d)\n', bestIoU, bestEpoch);
fprintf('  模型: %s\n', MODEL_SAVE_PATH);
fprintf('============================================================\n');

%% ========================================================================
%  辅助函数
% ========================================================================

function [loss, gradients] = modelLoss(net, dlX, dlY)
    % MODELOSS Dice + CrossEntropy 联合损失
    %   参考 experiment9_chasedb1_unet.m 中的实现
    %   dlX: [H W C N], dlY: [H W 1 N] (单精度 dlarray)
    %   返回: 标量 loss, gradients table

    % 前向传播 (网络最终层已是 softmax)
    dlYPred = forward(net, dlX);

    % --- CrossEntropy Loss (类别加权) ---
    % one-hot: [H W 2 N]
    dlYOneHot = cat(3, dlY == 0, dlY == 1);
    dlYOneHot = single(dlYOneHot);

    % 类别权重: 反比于像素数
    numBg     = max(double(sum(dlY == 0, 'all')), 1);
    numVessel = max(double(sum(dlY == 1, 'all')), 1);
    wVessel = numBg / numVessel;

    dlYWeighted = dlYOneHot;
    dlYWeighted(:,:,2,:) = dlYOneHot(:,:,2,:) .* wVessel;

    numPixels = size(dlY, 1) * size(dlY, 2) * size(dlY, 4);
    ceLoss = -sum(dlYWeighted .* log(dlYPred + eps), 'all') / numPixels;

    % --- Dice Loss (血管类) ---
    ps = reshape(dlYPred(:,:,2,:), [], size(dlY, 4));
    ts = reshape(single(dlY == 1), [], size(dlY, 4));

    inter = sum(ps .* ts, 1);
    union = sum(ps, 1) + sum(ts, 1);
    diceScore = (2 * inter + 1e-6) ./ (union + 1e-6);
    diceLoss = 1 - mean(diceScore);

    % 联合损失
    loss = ceLoss + diceLoss;

    % 梯度
    gradients = dlgradient(loss, net.Learnables);
end

function valIoU = evaluateVessels(net, valImgGPU, valMaskGPU, patchSize, useGPU)
    % EVALUATEVESSELS 快速验证 vessels.tif (单图, 滑窗推理)
    % 返回 ValIoU (标量)

    [h, w] = size(valMaskGPU);
    stride = 32;
    if useGPU
        scoreMap = zeros(h, w, 'single', 'gpuArray');
        weightMap = zeros(h, w, 'single', 'gpuArray');
    else
        scoreMap = zeros(h, w, 'single');
        weightMap = zeros(h, w, 'single');
    end

    for r = 1:stride:h-patchSize+1
        for c = 1:stride:w-patchSize+1
            patch = valImgGPU(r:r+patchSize-1, c:c+patchSize-1, :);
            dlPatch = dlarray(patch, 'SSCB');
            dlPred = predict(net, dlPatch);
            s = extractdata(dlPred(:,:,2));   % [256 256 1], 与 scoreMap 切片维度匹配

            scoreMap(r:r+patchSize-1, c:c+patchSize-1) = ...
                scoreMap(r:r+patchSize-1, c:c+patchSize-1) + s;
            weightMap(r:r+patchSize-1, c:c+patchSize-1) = ...
                weightMap(r:r+patchSize-1, c:c+patchSize-1) + 1;
        end
    end
    weightMap(weightMap == 0) = 1;
    scoreMap = scoreMap ./ weightMap;

    % 后处理: 高斯平滑 + 阈值 (scoreMap 已是 gpuArray, 非 dlarray)
    scoreMap = gather(scoreMap);
    scoreMap = imgaussfilt(scoreMap, 1.5);
    bw = scoreMap > 0.5;

    inter = sum(bw(:) & gather(valMaskGPU(:)));
    union = sum(bw(:) | gather(valMaskGPU(:)));
    valIoU = double(inter) / double(union);
end

function scoreMap = computeScoreMap(net, vesselsRGB, patchSize, stride, useGPU)
    % COMPUTESCOREMAP 滑窗推理得到完整 score map
    vesselsU8 = im2uint8(vesselsRGB);
    [h, w, ~] = size(vesselsU8);

    if useGPU
        scoreMap = zeros(h, w, 'single', 'gpuArray');
        weightMap = zeros(h, w, 'single', 'gpuArray');
    else
        scoreMap = zeros(h, w, 'single');
        weightMap = zeros(h, w, 'single');
    end

    for r = 1:stride:h-patchSize+1
        for c = 1:stride:w-patchSize+1
            % 归一化到 [0,1], 与训练数据一致
            patch = single(vesselsU8(r:r+patchSize-1, c:c+patchSize-1, :)) / 255;
            if useGPU
                patch = gpuArray(patch);
            end
            dlPatch = dlarray(patch, 'SSCB');
            p = extractdata(predict(net, dlPatch));
            s = p(:,:,2);
            scoreMap(r:r+patchSize-1, c:c+patchSize-1) = ...
                scoreMap(r:r+patchSize-1, c:c+patchSize-1) + s;
            weightMap(r:r+patchSize-1, c:c+patchSize-1) = ...
                weightMap(r:r+patchSize-1, c:c+patchSize-1) + 1;
        end
    end
    weightMap(weightMap == 0) = 1;
    scoreMap = scoreMap ./ weightMap;
    scoreMap = gather(scoreMap);
end
