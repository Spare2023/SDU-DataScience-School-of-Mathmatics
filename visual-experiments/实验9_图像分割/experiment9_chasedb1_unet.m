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

%% CHASEDB1 视网膜血管分割 — U-Net 完整训练
% 参考: dinov3-course/chapter10/chasedb1_train.py
%       使用 Python 相同策略: Dice+CE 损失, CosineAnnealing, 早停
%
% 环境: MATLAB R2025a, Deep Learning Toolbox
% 硬件: 建议 NVIDIA RTX 4060 (8GB) 或同等 GPU (512×512 depth=3)
%
% 数据: CHASEDB1 ~28 张眼底图像, 20 训练 / 8 验证
%       结构: chasedb1/train/{input,label}/  chasedb1/val/{input,label}/
%

clear; close all; clc;

%% ========================================================================
%  1. 配置参数 (与 Python 参考一致)
% ========================================================================
DATA_ROOT  = fullfile(fileparts(mfilename('fullpath')), 'chasedb1');
OUTPUT_DIR = fullfile(fileparts(mfilename('fullpath')), 'unet_output');
mkdir(OUTPUT_DIR);

IMG_SIZE    = [512 512];     % RTX 4060 优化: 512×512 (原1024)
NUM_CLASSES = 2;              % 背景 / 血管
BATCH_SIZE  = 4;              % RTX 4060 优化: bs=4 (原2)
MAX_EPOCHS  = 200;            % Python: max_epochs=200
LR_INIT     = 3e-4;           % Python: AdamW lr=3e-4
PATIENCE    = 20;             % Python: patience=20
SEED        = 42;
rng(SEED);

useGPU = canUseGPU();
if useGPU
    fprintf('设备: GPU (%s)\n', gpuDevice().Name);
else
    fprintf('设备: CPU (训练会很慢，建议使用 GPU)\n');
end
fprintf('输出目录: %s\n', OUTPUT_DIR);
fprintf('数据集: %s\n', DATA_ROOT);

%% ========================================================================
%  2. 数据加载与预处理
% ========================================================================
fprintf('\n========== [1/5] 加载数据 ==========\n');

% ---- 训练集 ----
fprintf('加载训练集... ');
trainImgDir   = fullfile(DATA_ROOT, 'train', 'input');
trainLabelDir = fullfile(DATA_ROOT, 'train', 'label');
trainFiles    = dir(fullfile(trainImgDir, '*.png'));
numTrain      = length(trainFiles);

% 预分配 (小数据集直接全部读入内存)
trainImages = zeros(IMG_SIZE(1), IMG_SIZE(2), 3, numTrain, 'uint8');
trainLabels = zeros(IMG_SIZE(1), IMG_SIZE(2), 1, numTrain, 'uint8');

for i = 1:numTrain
    % 图像: 读取 → 缩放 → uint8
    img = imread(fullfile(trainImgDir, trainFiles(i).name));
    trainImages(:,:,:,i) = imresize(img, IMG_SIZE);

    % 标签: 读取 → 缩放到相同尺寸 (最近邻, 保持离散值) → 二值化
    [~, baseName] = fileparts(trainFiles(i).name);
    labelFile = fullfile(trainLabelDir, [baseName '_1stHO.png']);
    if ~exist(labelFile, 'file')
        labelFile = fullfile(trainLabelDir, [baseName '.png']);
    end
    lbl = imread(labelFile);
    lbl = imresize(lbl, IMG_SIZE, 'nearest');
    trainLabels(:,:,1,i) = uint8(lbl > 0);
end
fprintf('%d 张 OK\n', numTrain);

% ---- 验证集 ----
fprintf('加载验证集... ');
valImgDir   = fullfile(DATA_ROOT, 'val', 'input');
valLabelDir = fullfile(DATA_ROOT, 'val', 'label');
valFiles    = dir(fullfile(valImgDir, '*.png'));
numVal      = length(valFiles);

valImages = zeros(IMG_SIZE(1), IMG_SIZE(2), 3, numVal, 'uint8');
valLabels = zeros(IMG_SIZE(1), IMG_SIZE(2), 1, numVal, 'uint8');

for i = 1:numVal
    img = imread(fullfile(valImgDir, valFiles(i).name));
    valImages(:,:,:,i) = imresize(img, IMG_SIZE);
    [~, baseName] = fileparts(valFiles(i).name);
    labelFile = fullfile(valLabelDir, [baseName '_1stHO.png']);
    if ~exist(labelFile, 'file')
        labelFile = fullfile(valLabelDir, [baseName '.png']);
    end
    lbl = imread(labelFile);
    lbl = imresize(lbl, IMG_SIZE, 'nearest');
    valLabels(:,:,1,i) = uint8(lbl > 0);
end
fprintf('%d 张 OK\n', numVal);

% 归一化到 [0,1] single 精度 (dlarray 推荐)
XTrain = single(trainImages) / 255;
YTrain = single(trainLabels);
XVal   = single(valImages) / 255;
YVal   = single(valLabels);

fprintf('训练: X %s, Y %s\n', mat2str(size(XTrain)), mat2str(size(YTrain)));
fprintf('验证: X %s, Y %s\n', mat2str(size(XVal)), mat2str(size(YVal)));

%% ========================================================================
%  3. 构建 U-Net (dlnetwork)
% ========================================================================
fprintf('\n========== [2/5] 构建 U-Net ==========\n');

USE_PRETRAINED = false;   % ← 设为 true 并使用预训练编码器可获得更高 IoU
if USE_PRETRAINED
    % 需要先安装支持包:
    %   matlab.addons.supportpackage.internal.explorer.showSupportPackages('RESNET18')
    % 或: matlab.addons.supportpackage.internal.explorer.showSupportPackages('RESNET50')
    encoderNet = imagePretrainedNetwork('resnet18');
    net = unet([IMG_SIZE 3], NUM_CLASSES, 'EncoderDepth', 3, 'EncoderNetwork', encoderNet);
    fprintf('使用预训练编码器: resnet18\n');
else
    net = unet([IMG_SIZE 3], NUM_CLASSES, 'EncoderDepth', 3);
    fprintf('使用随机初始化编码器\n');
end
numParams = sum(cellfun(@numel, net.Learnables.Value));
fprintf('U-Net 层数: %d\n', numel(net.Layers));
fprintf('可训练参数: %d (%.1fM)\n', numParams, numParams/1e6);

%% ========================================================================
%  4. 训练循环 (自定义: Dice + CrossEntropy + CosineAnnealing + 早停)
% ========================================================================
fprintf('\n========== [3/5] 开始训练 ==========\n');

% 优化器状态 (Adam)
trailingAvg   = [];
trailingAvgSq = [];
gradDecay     = 0.9;       % beta1 (Python: betas=(0.9, 0.999))
gradDecaySq   = 0.999;     % beta2
epsilon       = 1e-8;

% 余弦退火学习率 (CosineAnnealingLR)
lrFcn = @(epoch) LR_INIT * 0.5 * (1 + cos(pi * epoch / MAX_EPOCHS));

% 记录
trainLossLog = zeros(MAX_EPOCHS, 1);
valMioULog   = zeros(MAX_EPOCHS, 1);
valVesIouLog = zeros(MAX_EPOCHS, 1);

bestMIoU    = 0;
bestEpoch   = 0;
noImprove   = 0;
iterPerEpoch = ceil(numTrain / BATCH_SIZE);
globalIter   = 0;

for epoch = 1:MAX_EPOCHS
    epochTic = tic;
    lossSum  = 0;

    % ---- Shuffle ----
    idxOrder = randperm(numTrain);

    % ---- Mini-batch 训练 ----
    for iter = 1:iterPerEpoch
        globalIter = globalIter + 1;
        batchIdx = idxOrder((iter-1)*BATCH_SIZE + 1 : ...
                            min(iter*BATCH_SIZE, numTrain));

        XBatch = XTrain(:,:,:,batchIdx);
        YBatch = YTrain(:,:,:,batchIdx);

        % [数据增强] 随机水平 + 垂直翻转 (Python: torchvision.RandomHorizontalFlip)
        if rand > 0.5, XBatch = fliplr(XBatch); YBatch = fliplr(YBatch); end
        if rand > 0.5, XBatch = flipud(XBatch); YBatch = flipud(YBatch); end

        % 转为 dlarray (SSCB)
        dlX = dlarray(XBatch, 'SSCB');
        dlY = dlarray(YBatch, 'SSCB');
        if useGPU
            dlX = gpuArray(dlX);
            dlY = gpuArray(dlY);
        end

        % 前向 + 梯度计算 (dlfeval 内部调用 dlgradient)
        [loss, gradients] = dlfeval(@modelLoss, net, dlX, dlY);

        % Adam 更新 (含 weight decay)
        learnRate = lrFcn(epoch);
        [net, trailingAvg, trailingAvgSq] = adamupdate(net, gradients, ...
            trailingAvg, trailingAvgSq, globalIter, learnRate, ...
            gradDecay, gradDecaySq, epsilon);

        lossSum = lossSum + double(extractdata(loss));
    end

    avgLoss = lossSum / iterPerEpoch;
    trainLossLog(epoch) = avgLoss;

    % ---- 验证集评估 ----
    metrics = evaluateNet(net, XVal, YVal, useGPU);
    valMioULog(epoch)   = metrics.mIoU;
    valVesIouLog(epoch) = metrics.VesselIoU;

    % ---- 早停 & 保存最佳 ----
    if metrics.mIoU > bestMIoU
        bestMIoU  = metrics.mIoU;
        bestEpoch = epoch;
        noImprove = 0;
        save(fullfile(OUTPUT_DIR, 'unet_best.mat'), 'net', 'metrics', 'epoch');
    else
        noImprove = noImprove + 1;
    end

    epochTime = toc(epochTic);
    fprintf('E%03d/%d | Loss=%.4f | mIoU=%.4f | VesIoU=%.4f | best=%.4f@E%d | no_impr=%d | %.0fs\n', ...
        epoch, MAX_EPOCHS, avgLoss, metrics.mIoU, metrics.VesselIoU, ...
        bestMIoU, bestEpoch, noImprove, epochTime);

    if noImprove >= PATIENCE
        fprintf('>>> 早停触发! (best mIoU=%.4f @ Epoch %d)\n', bestMIoU, bestEpoch);
        break;
    end
end

% 截断未使用的记录
epochsDone = epoch;
trainLossLog = trainLossLog(1:epochsDone);
valMioULog   = valMioULog(1:epochsDone);
valVesIouLog = valVesIouLog(1:epochsDone);

% 加载最佳模型
best = load(fullfile(OUTPUT_DIR, 'unet_best.mat'));
finalMetrics = best.metrics;
net = best.net;

% 保存部署模型到实验目录 (供 experiment9_task3.m 复用)
% 包含: dlnetwork, 输入尺寸, 最佳指标
inputSize = IMG_SIZE;
save(fullfile(fileparts(mfilename('fullpath')), 'unet_task3_model.mat'), ...
    'net', 'inputSize', 'finalMetrics');
fprintf('部署模型已保存: unet_task3_model.mat\n');

fprintf('\n========== 训练完成 ==========\n');
fprintf('最佳: mIoU=%.4f, Vessel IoU=%.4f @ Epoch %d (共 %d 轮)\n', ...
    finalMetrics.mIoU, finalMetrics.VesselIoU, bestEpoch, epochsDone);

%% ========================================================================
%  5. 可视化 (训练曲线 + 样本预测)
% ========================================================================
fprintf('\n========== [4/5] 可视化 ==========\n');

% -- 5a. 训练曲线 --
figure('Name', 'U-Net 训练曲线', 'Position', [100 100 1400 500]);

subplot(1, 3, 1);
plot(1:epochsDone, trainLossLog, 'b-', 'LineWidth', 1.5);
xlabel('Epoch'); ylabel('Loss'); title('训练 Loss'); grid on;
xlim([1 epochsDone]);

subplot(1, 3, 2);
plot(1:epochsDone, valMioULog, 'g-', 'LineWidth', 1.5); hold on;
plot(1:epochsDone, valVesIouLog, 'r-', 'LineWidth', 1.5);
yline(finalMetrics.mIoU, 'g--', 'LineWidth', 1);
yline(finalMetrics.VesselIoU, 'r--', 'LineWidth', 1);
xlabel('Epoch'); ylabel('IoU'); xlim([1 epochsDone]);
title('验证集 mIoU / Vessel IoU');
legend('mIoU', 'Vessel IoU', 'Location', 'best'); grid on;

subplot(1, 3, 3);
barVals = [finalMetrics.mIoU, finalMetrics.VesselIoU];
b = bar(categorical({'mIoU', 'Vessel IoU'}), barVals);
ylabel('IoU'); ylim([0 1]); grid on; title('最佳模型指标');
text(1, barVals(1)+0.02, sprintf('%.4f', barVals(1)), ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold');
text(2, barVals(2)+0.02, sprintf('%.4f', barVals(2)), ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold');

saveas(gcf, fullfile(OUTPUT_DIR, 'curves.png'));
fprintf('  [OK] curves.png\n');

% -- 5b. 样本预测 (6 张验证图) --
figure('Name', 'U-Net 预测样本', 'Position', [100 100 1600 900]);
numShow = min(6, numVal);

for i = 1:numShow
    % 预测
    dlX = dlarray(XVal(:,:,:,i), 'SSCB');
    if useGPU, dlX = gpuArray(dlX); end
    dlPred = predict(net, dlX);
    predMask = squeeze(double(extractdata(dlPred(:,:,2,:)))) > 0.5;

    % 原图
    imgDisp = im2uint8(XVal(:,:,:,i));

    % 叠加 (蓝)
    overlayR = imgDisp;
    overlayR(:,:,1) = overlayR(:,:,1) * 0.6 + uint8(predMask) * 0.4 * 255;
    overlayR(:,:,3) = overlayR(:,:,3) * 0.6;

    % 真值
    gtMask = logical(YVal(:,:,:,i));

    subplot(3, numShow, i);
    imshow(imgDisp); title(sprintf('原图 %d', i));

    subplot(3, numShow, i + numShow);
    imshow(overlayR); title(sprintf('预测 (蓝色)'));

    subplot(3, numShow, i + 2*numShow);
    imshowpair(gtMask, predMask, 'montage');
    title('真值 | 预测');
end
sgtitle('CHASEDB1 — U-Net 分割结果');
saveas(gcf, fullfile(OUTPUT_DIR, 'samples.png'));
fprintf('  [OK] samples.png\n');

%% ========================================================================
%  6. 输出摘要
% ========================================================================
fprintf('\n========== [5/5] 摘要 ==========\n');
fprintf('  模型      mIoU     Vessel IoU    训练轮数\n');
fprintf('  ────────────────────────────────────────\n');
fprintf('  U-Net     %.4f     %.4f       %d/%d\n', ...
    finalMetrics.mIoU, finalMetrics.VesselIoU, epochsDone, MAX_EPOCHS);
fprintf('  ────────────────────────────────────────\n');
fprintf('  输出目录: %s\n', OUTPUT_DIR);
fprintf('  最佳模型: unet_best.mat\n');
fprintf('\n========================================\n');
fprintf('  训练完成!\n');
fprintf('========================================\n');

%% ========================================================================
%  辅助函数
% ========================================================================

function [loss, gradients] = modelLoss(net, dlX, dlY)
    % MODELOSS Dice + CrossEntropy 联合损失
    %   dlX: [H W C N] 输入图像
    %   dlY: [H W 1 N] 标签 (0=背景, 1=血管)
    %   返回: loss (标量), gradients (table)

    % 前向传播 (输出已是 softmax 概率, 网络末层为 FinalNetworkSoftmax-Layer)
    dlYPred = forward(net, dlX);

    % --- CrossEntropy Loss (类别加权) ---
    % 类别不均衡: 血管像素仅占 ~5%, CE 梯度被背景淹没, 模型卡在全背景预测
    % 加权后血管类梯度放大, 打破平衡
    % One-hot 编码标签: [H W 2 N]
    dlYOneHot = cat(3, dlY == 0, dlY == 1);
    dlYOneHot = single(dlYOneHot);

    % 计算类别权重: 反比于像素数 (转为普通数值, 避免 dlarray 标签冲突)
    numBg     = max(double(sum(dlY == 0, 'all')), 1);
    numVessel = max(double(sum(dlY == 1, 'all')), 1);
    wVessel = numBg / numVessel;  % ~19 (血管加权 ~19×)

    % 加权 one-hot (用 .* 避免 dlarray 标签冲突)
    dlYWeighted = dlYOneHot;
    dlYWeighted(:,:,2,:) = dlYOneHot(:,:,2,:) .* wVessel;

    % 归一化: 除以总像素数 (H×W×N)
    numPixels = size(dlY, 1) * size(dlY, 2) * size(dlY, 4);
    ceLoss = -sum(dlYWeighted .* log(dlYPred + eps), 'all') / numPixels;

    % --- Dice Loss (血管类) ---
    % 注意: SSCB 格式中 C(通道)在第3维 → (:,:,2,:)
    ps = reshape(dlYPred(:,:,2,:), [], size(dlY, 4));
    ts = reshape(single(dlY == 1), [], size(dlY, 4));

    inter = sum(ps .* ts, 1);
    union = sum(ps, 1) + sum(ts, 1);
    diceScore = (2 * inter + 1e-6) ./ (union + 1e-6);
    diceLoss = 1 - mean(diceScore);

    % 联合损失 (Python: F.cross_entropy + DiceLoss())
    loss = ceLoss + diceLoss;

    % 梯度
    gradients = dlgradient(loss, net.Learnables);
end

function metrics = evaluateNet(net, XVal, YVal, useGPU)
    % EVALUATENET 在验证集上计算 mIoU / Vessel IoU / Acc
    numVal = size(XVal, 4);
    numPixels = size(XVal, 1) * size(XVal, 2) * numVal;
    allPred = false(numPixels, 1);
    allTrue = false(numPixels, 1);
    offset = 0;

    for i = 1:numVal
        dlX = dlarray(XVal(:,:,:,i), 'SSCB');
        if useGPU
            dlX = gpuArray(dlX);
        end
        dlPred = predict(net, dlX);
        % Softmax 输出 [H W 2 N], 取通道2 (血管) → (:,:,2,:)
        predLabel = squeeze(double(extractdata(dlPred(:,:,2,:)))) > 0.5;
        trueLabel = squeeze(double(YVal(:,:,1,i))) > 0;

        n = numel(predLabel);
        allPred(offset+1:offset+n) = predLabel(:);
        allTrue(offset+1:offset+n) = trueLabel(:);
        offset = offset + n;
    end

    % 各类 IoU
    ious = zeros(2, 1);
    for cls = 0:1
        inter = sum((allPred == cls) & (allTrue == cls));
        union = sum((allPred == cls) | (allTrue == cls));
        if union > 0
            ious(cls+1) = inter / union;
        end
    end

    metrics.mIoU      = mean(ious);
    metrics.VesselIoU = ious(2);
    metrics.Acc       = sum(allPred == allTrue) / numel(allPred);
end
