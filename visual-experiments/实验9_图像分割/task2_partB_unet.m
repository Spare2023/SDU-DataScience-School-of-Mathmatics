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
%  task2_partB_unet.m — 任务2 Part B: U-Net CNV 分割 (独立脚本)
%  ========================================================================
%  功能: 从 experiment9.m 中独立出来的 Part B 代码
%  训练方式:
%    方案A (推荐CPU): 仅 cnv.png 单图 patch 训练 (≈20分钟)
%    方案B (需AMD-SD): AMD-SD 预训练 + cnv.png 微调 (≈2-15小时)
%  输出: IoU, Dice, 分割结果图
%
%  ╔══════════════════════════════════════════════════════════════╗
%  ║  数据集路径说明（按需修改）                                    ║
%  ║                                                            ║
%  ║  1. cnv.png     — 输入 OCT 图像 (放在本脚本同目录)            ║
%  ║  2. mask_cnv.png — 金标准掩膜 (放在本脚本同目录)              ║
%  ║  3. AMD-SD 预训练集:                                         ║
%  ║     E:\视觉与数据计算\数据集\AMD-SD\preprocessed\train\      ║
%  ║     E:\视觉与数据计算\数据集\AMD-SD\preprocessed\val\        ║
%  ║     → 见下方 "=== 路径配置 ===" 区块                          ║
%  ╚══════════════════════════════════════════════════════════════╝
%  ========================================================================

clear; close all; clc;
fprintf('========================================\n');
fprintf('  任务2 Part B: U-Net CNV 分割\n');
fprintf('========================================\n\n');

%% ========================================================================
%  === 路径配置 ============================================================
%  ========================================================================

% 当前脚本所在目录 (cnv.png 和 mask_cnv.png 应放在此目录)
ROOT_DIR = fileparts(mfilename('fullpath'));

% ---- AMD-SD 数据集路径 (方案B使用) ----
% 如果已运行 prepare_amd_sd.m 预处理，这里就是输出目录
% 如果路径不对，改成你自己的路径
AMD_SD_TRAIN_DIR = 'E:\视觉与数据计算\数据集\AMD-SD\preprocessed\train';
AMD_SD_VAL_DIR   = 'E:\视觉与数据计算\数据集\AMD-SD\preprocessed\val';

% ---- 模型保存路径 ----
MODEL_SAVE_PATH  = fullfile(ROOT_DIR, 'unet_cnv_model.mat');

%% ========================================================================
%  === 训练配置 ===========================================================
%  ========================================================================

% 训练模式:
%   'single'   — 仅 cnv.png 单图训练 (CPU推荐，≈20分钟)
%   'transfer' — AMD-SD预训练 + cnv.png微调 (需AMD-SD数据集)
TRAIN_MODE = 'transfer';     % ← 改成 'transfer' 使用AMD-SD数据集

% 网络参数
PATCH_SIZE       = 64;     % patch大小
STRIDE           = 16;     % 推理步长 (参考 exp9_619: 16, 精度更高)
PATCHES_PER_IMG  = 2000;   % 每张图提取的patch数 (参考: 2000)
BATCH_SIZE       = 16;     % mini-batch大小
LEARN_RATE       = 5e-4;   % 学习率
N_EPOCHS_SINGLE  = 50;     % 单图训练epochs (参考: 50, 分段LR)
N_EPOCHS_PRETR   = 5;      % AMD-SD预训练epochs
N_EPOCHS_FINETUNE = 50;    % 微调epochs (参考: 50)
AMD_SD_PATCHES   = 200;    % AMD-SD每图patch数
AMD_SD_MAX_IMGS  = 200;    % AMD-SD使用图像数上限
USE_DICE_LOSS    = false;  % true=Dice Loss, false=交叉熵

% ---- 组B调参开关 ----
TUNE_PARAMS      = true;   % true=开启调参, false=用固定值
% 支持 single 和 transfer 两种模式

%% ========================================================================
%  === 加载数据 ===========================================================
%  ========================================================================

fprintf('【加载数据】\n');

% 读入 cnv.png
cnv_path = fullfile(ROOT_DIR, 'cnv.png');
if ~exist(cnv_path, 'file')
    error('找不到 %s\n请将 cnv.png 放在 %s', cnv_path, ROOT_DIR);
end
cnv = imread(cnv_path);
if size(cnv, 3) == 3
    gray = im2double(rgb2gray(cnv));
else
    gray = im2double(cnv);
end
fprintf('  OCT图像: %s (%dx%d)\n', cnv_path, size(gray,2), size(gray,1));

% 读入金标准
gt_path = fullfile(ROOT_DIR, 'mask_cnv.png');
if ~exist(gt_path, 'file')
    error('找不到 %s', gt_path);
end
gtMask = imread(gt_path);
if size(gtMask, 3) == 3
    gtMask = rgb2gray(gtMask);
end
gtMask = gtMask > 0;
cnv_px = sum(gtMask(:));
fprintf('  金标准: CNV区域 %d px (%.1f%%)\n', cnv_px, 100*cnv_px/numel(gtMask));

img_u8 = im2uint8(gray);
class_names = ["background" "CNV"];

%% ========================================================================
%  === 数据准备 ===========================================================
%  ========================================================================

fprintf('\n【数据准备】\n');

switch TRAIN_MODE
    case 'transfer'
        % === 方案B: AMD-SD 预训练 + cnv.png 微调 ===
        if ~exist(AMD_SD_TRAIN_DIR, 'dir')
            error(['AMD-SD 训练集不存在: %s\n', ...
                   '请先运行 prepare_amd_sd.m 预处理，', ...
                   '或修改 AMD_SD_TRAIN_DIR 路径'], AMD_SD_TRAIN_DIR);
        end

        % 扫描 AMD-SD 训练图像
        img_files = dir(fullfile(AMD_SD_TRAIN_DIR, '*_img.png'));
        n_total = length(img_files);
        n_use = min(n_total, AMD_SD_MAX_IMGS);
        % 均匀采样
        use_idx = 1:round(n_total/n_use):n_total;
        use_idx = unique(min(use_idx, n_total));
        use_idx = use_idx(1:min(length(use_idx), n_use));

        train_img_paths = cell(length(use_idx), 1);
        train_msk_paths = cell(length(use_idx), 1);
        for i = 1:length(use_idx)
            base_name = img_files(use_idx(i)).name;
            base_name = strrep(base_name, '_img.png', '');
            train_img_paths{i} = fullfile(AMD_SD_TRAIN_DIR, [base_name '_img.png']);
            train_msk_paths{i} = fullfile(AMD_SD_TRAIN_DIR, [base_name '_mask.png']);
        end
        fprintf('  AMD-SD 训练集: %d 张图像\n', length(train_img_paths));

        % 创建 AMD-SD datastore
        imds_amd = imageDatastore(train_img_paths);
        pxds_amd = pixelLabelDatastore(train_msk_paths, class_names, [0; 255]);
        aug_amd = imageDataAugmenter(...
            'RandRotation', [-10 10], ...
            'RandXTranslation', [-5 5], ...
            'RandYTranslation', [-5 5], ...
            'RandXReflection', true, ...
            'RandXScale', [0.9 1.1], ...
            'RandYScale', [0.9 1.1]);

        ds_amd = randomPatchExtractionDatastore(imds_amd, pxds_amd, ...
            [PATCH_SIZE PATCH_SIZE], ...
            'PatchesPerImage', AMD_SD_PATCHES, ...
            'DataAugmentation', aug_amd);

        % 创建 cnv.png 微调 datastore
        temp_label_path = fullfile(tempdir, 'temp_cnv_label.png');
        imwrite(uint8(gtMask) * 255, temp_label_path);
        temp_img_path = fullfile(tempdir, 'temp_cnv_gray.png');
        imwrite(gray, temp_img_path);

        imds_fine = imageDatastore(temp_img_path);
        pxds_fine = pixelLabelDatastore(temp_label_path, class_names, [0; 255]);
        aug_fine = imageDataAugmenter(...
            'RandRotation', [-15 15], ...
            'RandXTranslation', [-5 5], ...
            'RandYTranslation', [-5 5], ...
            'RandXReflection', true);

        ds_fine = randomPatchExtractionDatastore(imds_fine, pxds_fine, ...
            [PATCH_SIZE PATCH_SIZE], ...
            'PatchesPerImage', 1000, ...
            'DataAugmentation', aug_fine);

    case 'single'
        % === 方案A: 仅 cnv.png 单图训练 ===
        fprintf('  单图训练模式\n');
        temp_label_path = fullfile(tempdir, 'temp_cnv_label.png');
        imwrite(uint8(gtMask) * 255, temp_label_path);
        temp_img_path = fullfile(tempdir, 'temp_cnv_gray.png');
        imwrite(gray, temp_img_path);

        imds_single = imageDatastore(temp_img_path);
        pxds_single = pixelLabelDatastore(temp_label_path, class_names, [0; 255]);
        aug_single = imageDataAugmenter(...
            'RandRotation', [-15 15], ...
            'RandXTranslation', [-5 5], ...
            'RandYTranslation', [-5 5], ...
            'RandXReflection', true);

        ds_train = randomPatchExtractionDatastore(imds_single, pxds_single, ...
            [PATCH_SIZE PATCH_SIZE], ...
            'PatchesPerImage', PATCHES_PER_IMG, ...
            'DataAugmentation', aug_single);
        fprintf('  Patches/epoch: %d\n', PATCHES_PER_IMG);

    otherwise
        error('未知 TRAIN_MODE: %s (可选: single / transfer)', TRAIN_MODE);
end

%% ========================================================================
%  === 训练 U-Net =========================================================
%  ========================================================================

fprintf('\n【训练 U-Net】\n');
inputSize = [PATCH_SIZE PATCH_SIZE 1];
warning('off', 'all');

switch TRAIN_MODE
    case 'transfer'
        if TUNE_PARAMS
            % === 组B调参: transfer 模式 ===
            fprintf('【组B调参】Transfer 单变量扫描\n');
            fprintf('  注意: 每轮需完整运行预训练+微调, 耗时较长\n\n');

            base = struct(...
                'AMD_SD_PATCHES',   AMD_SD_PATCHES, ...
                'AMD_SD_MAX_IMGS',  AMD_SD_MAX_IMGS, ...
                'N_EPOCHS_PRETR',   N_EPOCHS_PRETR, ...
                'N_EPOCHS_FINETUNE', N_EPOCHS_FINETUNE, ...
                'BATCH_SIZE',       BATCH_SIZE, ...
                'LEARN_RATE',       LEARN_RATE, ...
                'PATCH_SIZE',       PATCH_SIZE, ...
                'PATCHES_PER_IMG',  1000);

            sweep_ranges = {...
                'AMD_SD_PATCHES',   [100, 200, 500]; ...
                'AMD_SD_MAX_IMGS',  [100, 200, 500]; ...
                'N_EPOCHS_PRETR',   [3, 5, 10]; ...
                'N_EPOCHS_FINETUNE', [10, 20, 30]; ...
                'BATCH_SIZE',       [8, 16, 32]; ...
                'LEARN_RATE',       [1e-4, 5e-4, 1e-3]; ...
                'PATCH_SIZE',       [48, 64, 96] ...
            };

            best_overall = struct('IoU', 0);
            all_results = {};

            for p = 1:size(sweep_ranges, 1)
                param_name = sweep_ranges{p, 1};
                param_vals = sweep_ranges{p, 2};
                n_vals = length(param_vals);

                fprintf('\n── 调参 %d/%d: %s (%d 个值) ──\n', ...
                    p, size(sweep_ranges,1), param_name, n_vals);

                ious = zeros(n_vals, 1);
                models = cell(n_vals, 1);
                times = zeros(n_vals, 1);

                for v = 1:n_vals
                    val = param_vals(v);
                    eval(sprintf('%s = %s;', param_name, mat2str(val)));
                    fprintf('  %s = %s ', param_name, mat2str(val));

                    % 重建 datastore (参数变了需重建)
                    [ds_amd, ds_fine] = rebuildTransferDS(ROOT_DIR, ...
                        AMD_SD_TRAIN_DIR, gray, gtMask, class_names, ...
                        AMD_SD_PATCHES, AMD_SD_MAX_IMGS, PATCH_SIZE, ...
                        1000);

                    % 两阶段训练
                    inputSize = [PATCH_SIZE PATCH_SIZE 1];
                    lgraph = buildUnetWithDiceLoss(inputSize, USE_DICE_LOSS);

                    opts1 = trainingOptions('adam', ...
                        'MaxEpochs', N_EPOCHS_PRETR, ...
                        'MiniBatchSize', BATCH_SIZE, ...
                        'InitialLearnRate', LEARN_RATE, ...
                        'Verbose', true, 'VerboseFrequency', 20, ...
                        'Shuffle', 'every-epoch', 'Plots', 'none', ...
                        'ExecutionEnvironment', 'auto');
                    t_v = tic;
                    net_v = trainNetwork(ds_amd, lgraph, opts1);

                    opts2 = trainingOptions('adam', ...
                        'MaxEpochs', N_EPOCHS_FINETUNE, ...
                        'MiniBatchSize', BATCH_SIZE, ...
                        'InitialLearnRate', 1e-4, ...
                        'Verbose', true, 'VerboseFrequency', 20, ...
                        'Shuffle', 'every-epoch', 'Plots', 'none', ...
                        'ExecutionEnvironment', 'auto');
                    net_v = trainNetwork(ds_fine, lgraph, opts2);
                    times(v) = toc(t_v);

                    iou_v = evalIoU(net_v, img_u8, gtMask, STRIDE, PATCH_SIZE);
                    ious(v) = iou_v;
                    models{v} = net_v;
                    fprintf('→ IoU=%.4f (%.0fs)\n', iou_v, times(v));
                end

                [best_iou, best_idx] = max(ious);
                best_val = param_vals(best_idx);
                fprintf('  ★ 最佳 %s = %s → IoU=%.4f\n', ...
                    param_name, mat2str(best_val), best_iou);

                result = struct('param', param_name, ...
                    'values', param_vals, 'ious', ious, ...
                    'best_val', best_val, 'best_iou', best_iou);
                all_results{end+1} = result;

                if best_iou > best_overall.IoU
                    best_overall.IoU = best_iou;
                    best_overall.net = models{best_idx};
                    best_overall.params = struct();
                    for k = 1:size(sweep_ranges,1)
                        pn = sweep_ranges{k,1};
                        best_overall.params.(pn) = eval(pn);
                    end
                end

                for fn = fieldnames(base)'
                    eval(sprintf('%s = %f;', fn{1}, base.(fn{1})));
                end
            end

            % 总结
            fprintf('\n═══════════════════════════════════════════\n');
            fprintf('  组B调参总结 (transfer)\n');
            fprintf('═══════════════════════════════════════════\n');
            for p = 1:length(all_results)
                r = all_results{p};
                fprintf('  %-18s: 最佳=%-8s  IoU=%.4f\n', ...
                    r.param, mat2str(r.best_val), r.best_iou);
            end
            fprintf('───────────────────────────────────────────\n');
            fprintf('  ★ 全局最佳 IoU = %.4f\n', best_overall.IoU);
            fprintf('═══════════════════════════════════════════\n');

            best_params_B = best_overall.params;
            save(fullfile(ROOT_DIR, 'best_params_groupB.mat'), 'best_params_B', ...
                'all_results', '-v7.3');
            fprintf('  组B最佳参数已保存 → best_params_groupB.mat\n');

            net = best_overall.net;
            save(MODEL_SAVE_PATH, 'net');
            fprintf('  最佳模型已保存 → %s\n', MODEL_SAVE_PATH);
            for fn = fieldnames(best_overall.params)'
                eval(sprintf('%s = %f;', fn{1}, best_overall.params.(fn{1})));
            end

            figure('Name', '组B调参结果(transfer)', 'NumberTitle', 'off', ...
                'Position', [100 100 1200 600]);
            n_plots = length(all_results);
            for p = 1:n_plots
                subplot(2, ceil(n_plots/2), p);
                r = all_results{p};
                plot(1:length(r.values), r.ious, 'b-o', 'LineWidth', 1.5);
                hold on;
                [~, bp] = max(r.ious);
                plot(bp, r.ious(bp), 'r*', 'MarkerSize', 12);
                hold off;
                set(gca, 'XTick', 1:length(r.values), ...
                    'XTickLabel', cellstr(num2str(r.values(:), '%.0e')));
                xlabel(r.param); ylabel('IoU');
                title(sprintf('%s: 最佳=%s (IoU=%.4f)', ...
                    r.param, mat2str(r.best_val), max(r.ious)));
                grid on;
            end
            sgtitle('组B调参 — Transfer 单变量扫描 IoU 曲线');

        else
            % ---- 阶段1: AMD-SD 预训练 ----
            fprintf('  阶段1: AMD-SD 预训练 (%d epochs)...\n', N_EPOCHS_PRETR);
            lgraph = buildUnetWithDiceLoss(inputSize, USE_DICE_LOSS);
            opts1 = trainingOptions('adam', ...
                'MaxEpochs', N_EPOCHS_PRETR, ...
                'MiniBatchSize', BATCH_SIZE, ...
                'InitialLearnRate', LEARN_RATE, ...
                'Verbose', true, 'VerboseFrequency', 20, ...
                'Shuffle', 'every-epoch', 'Plots', 'none', ...
                'ExecutionEnvironment', 'auto');
            t_pretrain = tic;
            net = trainNetwork(ds_amd, lgraph, opts1);
            fprintf('  预训练完成: %.1f秒\n', toc(t_pretrain));

            % ---- 阶段2: cnv.png 微调 ----
            fprintf('  阶段2: cnv.png 微调 (%d epochs)...\n', N_EPOCHS_FINETUNE);
            opts2 = trainingOptions('adam', ...
                'MaxEpochs', N_EPOCHS_FINETUNE, ...
                'MiniBatchSize', BATCH_SIZE, ...
                'InitialLearnRate', 1e-4, ...
                'LearnRateSchedule', 'piecewise', ...
                'LearnRateDropFactor', 0.3, ...
                'LearnRateDropPeriod', 20, ...
                'Verbose', true, 'VerboseFrequency', 10, ...
                'Shuffle', 'every-epoch', 'Plots', 'none', ...
                'ExecutionEnvironment', 'auto');
            t_finetune = tic;
            net = trainNetwork(ds_fine, lgraph, opts2);
            fprintf('  微调完成: %.1f秒\n', toc(t_finetune));

            % 保存模型
            fprintf('  保存模型 → %s\n', MODEL_SAVE_PATH);
            save(MODEL_SAVE_PATH, 'net');
        end

    case 'single'
        if TUNE_PARAMS
            % === 组B调参模式: 扫参 + 训练 ===
            fprintf('【组B调参】单变量扫描 (每次改一个参数, 其余固定)\n');
            fprintf('  总耗时预估: 约 %.0f 分钟 (GPU)\n\n', ...
                2 * (4+4+3+5+3));  % 各参数取值个数 × 每轮约2分钟

            % 当前固定基线值
            base = struct(...
                'PATCHES_PER_IMG', PATCHES_PER_IMG, ...
                'N_EPOCHS_SINGLE', N_EPOCHS_SINGLE, ...
                'BATCH_SIZE',      BATCH_SIZE, ...
                'LEARN_RATE',      LEARN_RATE, ...
                'PATCH_SIZE',      PATCH_SIZE);

            % 各参数扫描范围
            sweep_ranges = {...
                'PATCHES_PER_IMG', [500, 1000, 1500, 2000]; ...
                'N_EPOCHS_SINGLE', [10, 20, 30, 40]; ...
                'BATCH_SIZE',      [8, 16, 32]; ...
                'LEARN_RATE',      [1e-4, 3e-4, 5e-4, 8e-4, 1e-3]; ...
                'PATCH_SIZE',      [48, 64, 96] ...
            };

            best_overall = struct('IoU', 0);
            all_results = {};

            for p = 1:size(sweep_ranges, 1)
                param_name  = sweep_ranges{p, 1};
                param_vals  = sweep_ranges{p, 2};
                n_vals      = length(param_vals);

                fprintf('\n── 调参 %d/%d: %s (%d 个值) ──\n', ...
                    p, size(sweep_ranges,1), param_name, n_vals);

                ious = zeros(n_vals, 1);
                models = cell(n_vals, 1);
                times = zeros(n_vals, 1);

                for v = 1:n_vals
                    % 设置当前参数值
                    val = param_vals(v);
                    eval(sprintf('%s = %s;', param_name, mat2str(val)));
                    fprintf('  %s = %s ', param_name, mat2str(val));

                    % 构建数据 (patch_size 变了需重建 datastore)
                    if strcmp(param_name, 'PATCH_SIZE')
                        ds_train = buildSingleDS(gray, gtMask, ...
                            PATCH_SIZE, PATCHES_PER_IMG, class_names);
                    end

                    % 训练
                    inputSize = [PATCH_SIZE PATCH_SIZE 1];
                    lgraph = buildUnetWithDiceLoss(inputSize, USE_DICE_LOSS);
                    opts = trainingOptions('adam', ...
                        'MaxEpochs', N_EPOCHS_SINGLE, ...
                        'MiniBatchSize', BATCH_SIZE, ...
                        'InitialLearnRate', LEARN_RATE, ...
                        'LearnRateSchedule', 'piecewise', ...
                        'LearnRateDropFactor', 0.3, ...
                        'LearnRateDropPeriod', 20, ...
                        'Verbose', true, 'VerboseFrequency', 20, ...
                        'Shuffle', 'every-epoch', 'Plots', 'none', ...
                        'ExecutionEnvironment', 'auto');
                    t_v = tic;
                    net_v = trainNetwork(ds_train, lgraph, opts);
                    times(v) = toc(t_v);

                    % 推理 + 评估 IoU
                    iou_v = evalIoU(net_v, img_u8, gtMask, STRIDE, PATCH_SIZE);
                    ious(v) = iou_v;
                    models{v} = net_v;
                    fprintf('→ IoU=%.4f (%.0fs)\n', iou_v, times(v));
                end

                % 找当前参数最佳
                [best_iou, best_idx] = max(ious);
                best_val = param_vals(best_idx);
                fprintf('  ★ 最佳 %s = %s → IoU=%.4f\n', ...
                    param_name, mat2str(best_val), best_iou);

                % 记录
                result = struct('param', param_name, ...
                    'values', param_vals, 'ious', ious, ...
                    'best_val', best_val, 'best_iou', best_iou);
                all_results{end+1} = result;

                % 更新全局最佳
                if best_iou > best_overall.IoU
                    best_overall.IoU = best_iou;
                    best_overall.net = models{best_idx};
                    best_overall.params = struct();
                    for k = 1:size(sweep_ranges,1)
                        pn = sweep_ranges{k,1};
                        best_overall.params.(pn) = eval(pn);
                    end
                end

                % 恢复基线（除当前参数外保持默认）
                for fn = fieldnames(base)'
                    eval(sprintf('%s = %f;', fn{1}, base.(fn{1})));
                end
            end

            % ---- 扫参总结 ----
            fprintf('\n═══════════════════════════════════════════\n');
            fprintf('  组B调参总结\n');
            fprintf('═══════════════════════════════════════════\n');
            for p = 1:length(all_results)
                r = all_results{p};
                fprintf('  %-16s: 最佳=%-8s  IoU=%.4f\n', ...
                    r.param, mat2str(r.best_val), r.best_iou);
            end
            fprintf('───────────────────────────────────────────\n');
            fprintf('  ★ 全局最佳 IoU = %.4f\n', best_overall.IoU);
            fprintf('═══════════════════════════════════════════\n');

            % 保存组B调参结果供主程序参考
            best_params_B = best_overall.params;
            ROOT_DIR = fileparts(mfilename('fullpath'));
            save(fullfile(ROOT_DIR, 'best_params_groupB.mat'), 'best_params_B', ...
                'all_results', '-v7.3');
            fprintf('  组B最佳参数已保存 → best_params_groupB.mat\n');

            % 使用最佳模型
            net = best_overall.net;
            save(MODEL_SAVE_PATH, 'net');
            fprintf('  最佳模型已保存 → %s\n', MODEL_SAVE_PATH);
            % 同步参数到工作区
            for fn = fieldnames(best_overall.params)'
                eval(sprintf('%s = %f;', fn{1}, best_overall.params.(fn{1})));
            end
            fprintf('\n  使用最佳参数进行推理\n');

            % ---- 画扫参曲线 ----
            figure('Name', '组B调参结果', 'NumberTitle', 'off', ...
                'Position', [100 100 1200 600]);
            n_plots = length(all_results);
            for p = 1:n_plots
                subplot(2, ceil(n_plots/2), p);
                r = all_results{p};
                vals = r.values;
                ious = r.ious;
                plot(1:length(vals), ious, 'b-o', 'LineWidth', 1.5);
                hold on;
                [~, best_p] = max(ious);
                plot(best_p, ious(best_p), 'r*', 'MarkerSize', 12);
                hold off;
                set(gca, 'XTick', 1:length(vals), ...
                    'XTickLabel', cellstr(num2str(vals(:), '%.0e')));
                xlabel(r.param); ylabel('IoU');
                title(sprintf('%s: 最佳=%s (IoU=%.4f)', ...
                    r.param, mat2str(r.best_val), max(ious)));
                grid on;
            end
            sgtitle('组B调参 — 单变量扫描 IoU 曲线');

        else
            % === 常规单图训练 (不调参) ===
            fprintf('  训练 (%d epochs, %d patches/epoch)...\n', ...
                N_EPOCHS_SINGLE, PATCHES_PER_IMG);
            lgraph = buildUnetWithDiceLoss(inputSize, USE_DICE_LOSS);
            opts = trainingOptions('adam', ...
                'MaxEpochs', N_EPOCHS_SINGLE, ...
                'MiniBatchSize', BATCH_SIZE, ...
                'InitialLearnRate', LEARN_RATE, ...
                'LearnRateSchedule', 'piecewise', ...
                'LearnRateDropFactor', 0.3, ...
                'LearnRateDropPeriod', 20, ...
                'Verbose', true, 'VerboseFrequency', 10, ...
                'Shuffle', 'every-epoch', 'Plots', 'none', ...
                'ExecutionEnvironment', 'auto');
            t_train = tic;
            net = trainNetwork(ds_train, lgraph, opts);
            fprintf('  训练完成: %.1f秒\n', toc(t_train));
            fprintf('  保存模型 → %s\n', MODEL_SAVE_PATH);
            save(MODEL_SAVE_PATH, 'net');
        end
end

%% ========================================================================
%  === 推理 (滑窗预测) =====================================================
%  ========================================================================

fprintf('\n【推理】滑窗预测 (stride=%d)...\n', STRIDE);
[h, w] = size(img_u8);
score_map = zeros(h, w);
weight_map = zeros(h, w);

for r = 1:STRIDE:h-PATCH_SIZE+1
    for c = 1:STRIDE:w-PATCH_SIZE+1
        patch = single(img_u8(r:r+PATCH_SIZE-1, c:c+PATCH_SIZE-1));
        p = predict(net, patch);
        s = p(:,:,2);  % CNV 类得分
        score_map(r:r+PATCH_SIZE-1, c:c+PATCH_SIZE-1) = ...
            score_map(r:r+PATCH_SIZE-1, c:c+PATCH_SIZE-1) + s;
        weight_map(r:r+PATCH_SIZE-1, c:c+PATCH_SIZE-1) = ...
            weight_map(r:r+PATCH_SIZE-1, c:c+PATCH_SIZE-1) + 1;
    end
end
weight_map(weight_map == 0) = 1;
score_map = score_map ./ weight_map;

% ★ 高斯平滑 + 二值化 + 后处理（参考 exp9_619）
score_map = imgaussfilt(score_map, 1.5);
resultMask_B = score_map > 0.5;
resultMask_B = bwareaopen(resultMask_B, 50);           % 去小噪点
resultMask_B = imclose(resultMask_B, strel('disk', 3)); % 闭运算弥合小洞
resultMask_B = imfill(resultMask_B, 'holes');           % 填充孔洞
L_B = bwlabel(resultMask_B);                            % 取最大连通域
s_B = regionprops(L_B, 'Area');
if ~isempty(s_B)
    [~, mi] = max([s_B.Area]);
    resultMask_B = (L_B == mi);
end

%% ========================================================================
%  === 评估 ===============================================================
%  ========================================================================

fprintf('\n【评估】\n');
inter_B = sum(resultMask_B(:) & gtMask(:));
union_B = sum(resultMask_B(:) | gtMask(:));
IoU_B   = inter_B / union_B;
dice_B  = 2 * inter_B / (sum(resultMask_B(:)) + sum(gtMask(:)));

pred_px = sum(resultMask_B(:));
fprintf('  U-Net预测: %d px (CNV)\n', pred_px);
fprintf('  GT金标准:  %d px (CNV)\n', sum(gtMask(:)));
fprintf('  IoU = %.4f\n', IoU_B);
fprintf('  Dice = %.4f\n', dice_B);

%% ========================================================================
%  === 可视化 =============================================================
%  ========================================================================

figure('Name', 'Part B: U-Net CNV分割', 'NumberTitle', 'off', ...
    'Position', [100 100 1200 400]);

subplot(1,3,1);
imshow(gray, []); title('OCT原图像');

subplot(1,3,2);
imshow(gtMask); title('金标准 GT');

subplot(1,3,3);
imshow(labeloverlay(gray, resultMask_B, 'Transparency', 0.6));
title(sprintf('U-Net分割 (IoU=%.4f, Dice=%.4f)', IoU_B, dice_B));

fprintf('\n✅ 任务2 Part B 完成!\n');

%% ========================================================================
%  === 路径修改备忘 =======================================================
%  ========================================================================
%  如需修改数据集路径，改上面 "=== 路径配置 ===" 区块里的:
%
%     AMD_SD_TRAIN_DIR — AMD-SD 训练集目录（含 *_img.png 和 *_mask.png）
%     AMD_SD_VAL_DIR   — AMD-SD 验证集目录（同上结构）
%     MODEL_SAVE_PATH  — 训练好的模型保存位置
%
%  如果 AMD-SD 数据集不在默认位置:
%    1. 找到你的 AMD-SD/preprocessed/train 文件夹
%    2. 把 AMD_SD_TRAIN_DIR 改成该文件夹的完整路径
%    3. 同样改 AMD_SD_VAL_DIR
%
%  如果只想用 cnv.png 单图训练:
%    设置 TRAIN_MODE = 'single' (默认)
%  ========================================================================

%% ========================================================================
%  辅助函数: 构建 U-Net (可选 Dice Loss)
%  ========================================================================
function lgraph = buildUnetWithDiceLoss(inputSize, useDice)
    % 构建 U-Net，可选择 Dice Loss 替换默认交叉熵
    %
    % 输入:
    %   inputSize — [h w c] 输入尺寸
    %   useDice   — true → Dice Loss, false → 交叉熵 (默认)
    % 输出:
    %   lgraph    — LayerGraph 对象

    numClasses = 2;   % 背景 + CNV
    lgraph = unetLayers(inputSize, numClasses, 'EncoderDepth', 3);

    if useDice
        % 找到最后一层 (pixelClassificationLayer) 和它的前一层 (softmax)
        layers = lgraph.Layers;
        lastLayer = layers(end);

        % 确认最后一层确实是分类层
        if isa(lastLayer, 'nnet.cnn.layer.ClassificationLayer')
            softmaxName = layers(end-1).Name;
            lastLayerName = lastLayer.Name;

            % 删除原分类层
            lgraph = removeLayers(lgraph, lastLayerName);

            % 添加 Dice 损失层
            diceLayer = dicePixelClassificationLayer('diceLoss');
            lgraph = addLayers(lgraph, diceLayer);

            % softmax → diceLoss
            lgraph = connectLayers(lgraph, softmaxName, 'diceLoss');
            fprintf('    ✓ 使用 Dice Loss (直接优化 IoU)\n');
        else
            fprintf('    → 使用默认交叉熵损失\n');
        end
    else
        fprintf('    → 使用默认交叉熵损失\n');
    end
end

%% ========================================================================
%  辅助函数: 构建单图 datastore (patch_size 可变)
%  ========================================================================
function ds = buildSingleDS(gray, gtMask, patch_size, patches_per_img, class_names)
    temp_label = fullfile(tempdir, 'temp_cnv_label.png');
    imwrite(uint8(gtMask) * 255, temp_label);
    temp_img = fullfile(tempdir, 'temp_cnv_gray.png');
    imwrite(gray, temp_img);

    imds = imageDatastore(temp_img);
    pxds = pixelLabelDatastore(temp_label, class_names, [0; 255]);
    aug = imageDataAugmenter(...
        'RandRotation', [-15 15], ...
        'RandXTranslation', [-5 5], ...
        'RandYTranslation', [-5 5], ...
        'RandXReflection', true);
    ds = randomPatchExtractionDatastore(imds, pxds, ...
        [patch_size patch_size], ...
        'PatchesPerImage', patches_per_img, ...
        'DataAugmentation', aug);
end

%% ========================================================================
%  辅助函数: 推理 + 计算 IoU (用于调参)
%  ========================================================================
function iou = evalIoU(net, img_u8, gtMask, stride, patch_size)
    % 滑窗推理 + IoU 计算 (无声)
    [h, w] = size(img_u8);
    score_map = zeros(h, w);
    weight_map = zeros(h, w);
    for r = 1:stride:h-patch_size+1
        for c = 1:stride:w-patch_size+1
            patch = single(img_u8(r:r+patch_size-1, c:c+patch_size-1));
            p = predict(net, patch);
            s = p(:,:,2);
            score_map(r:r+patch_size-1, c:c+patch_size-1) = ...
                score_map(r:r+patch_size-1, c:c+patch_size-1) + s;
            weight_map(r:r+patch_size-1, c:c+patch_size-1) = ...
                weight_map(r:r+patch_size-1, c:c+patch_size-1) + 1;
        end
    end
    weight_map(weight_map == 0) = 1;
    score_map = score_map ./ weight_map;
    score_map = imgaussfilt(score_map, 1.5);
    bw = score_map > 0.5;
    inter = sum(bw(:) & gtMask(:));
    union = sum(bw(:) | gtMask(:));
    iou = inter / union;
end

%% ========================================================================
%  辅助函数: 重建 transfer 模式的 datastore (调参时使用)
%  ========================================================================
function [ds_amd, ds_fine] = rebuildTransferDS(~, amd_dir, ...
    gray, gtMask, class_names, amd_patches, amd_max_imgs, patch_size, fine_patches)

    % 扫描 AMD-SD 训练图像
    img_files = dir(fullfile(amd_dir, '*_img.png'));
    n_total = length(img_files);
    n_use = min(n_total, amd_max_imgs);
    use_idx = 1:round(n_total/n_use):n_total;
    use_idx = unique(min(use_idx, n_total));
    use_idx = use_idx(1:min(length(use_idx), n_use));

    train_img_paths = cell(length(use_idx), 1);
    train_msk_paths = cell(length(use_idx), 1);
    for i = 1:length(use_idx)
        base_name = img_files(use_idx(i)).name;
        base_name = strrep(base_name, '_img.png', '');
        train_img_paths{i} = fullfile(amd_dir, [base_name '_img.png']);
        train_msk_paths{i} = fullfile(amd_dir, [base_name '_mask.png']);
    end

    % AMD-SD datastore
    imds = imageDatastore(train_img_paths);
    pxds = pixelLabelDatastore(train_msk_paths, class_names, [0; 255]);
    aug = imageDataAugmenter(...
        'RandRotation', [-10 10], ...
        'RandXTranslation', [-5 5], ...
        'RandYTranslation', [-5 5], ...
        'RandXReflection', true, ...
        'RandXScale', [0.9 1.1], ...
        'RandYScale', [0.9 1.1]);
    ds_amd = randomPatchExtractionDatastore(imds, pxds, ...
        [patch_size patch_size], ...
        'PatchesPerImage', amd_patches, ...
        'DataAugmentation', aug);

    % cnv.png 微调 datastore
    temp_label = fullfile(tempdir, 'temp_cnv_label.png');
    imwrite(uint8(gtMask) * 255, temp_label);
    temp_img = fullfile(tempdir, 'temp_cnv_gray.png');
    imwrite(gray, temp_img);

    imds_fine = imageDatastore(temp_img);
    pxds_fine = pixelLabelDatastore(temp_label, class_names, [0; 255]);
    aug_fine = imageDataAugmenter(...
        'RandRotation', [-15 15], ...
        'RandXTranslation', [-5 5], ...
        'RandYTranslation', [-5 5], ...
        'RandXReflection', true);
    ds_fine = randomPatchExtractionDatastore(imds_fine, pxds_fine, ...
        [patch_size patch_size], ...
        'PatchesPerImage', fine_patches, ...
        'DataAugmentation', aug_fine);
end
