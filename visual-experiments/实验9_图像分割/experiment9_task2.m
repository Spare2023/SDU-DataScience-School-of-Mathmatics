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

%% 实验9: 图像分割 — 任务2: CNV OCT图像分割
% 课程: 视觉与数据计算
% 重点函数: multithresh, graythresh
clear all;
close all;
clc;
fig_dir = fullfile(fileparts(mfilename('fullpath')), 'figs_task2');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
global DEMO_MODE;
DEMO_MODE = false;
fprintf('========================================\n');
fprintf('          实验9: 图像分割 — 任务2\n');
fprintf('========================================\n\n');

%% ========================================================================
%  任务2: CNV OCT图像分割
%  ========================================================================
fprintf('\n【任务2】CNV OCT图像分割\n');
fprintf('  Part A: 传统方法 (参考 chap9_2.m + 论文启发改进)\n');
fprintf('  Part B: 深度学习 U-Net (patch-based 训练)\n');
demoPause('任务2: 切换到 CNV OCT 图像 — 脉络膜新生血管分割');
fprintf('----------------------------------------\n');
% 读入cnv图像
cnv_path = 'cnv.png';
cnv = imread(cnv_path);
if size(cnv, 3) == 3
    gray = rgb2gray(cnv);
else
    gray = cnv;
end
fprintf('使用图像: %s (%dx%d)\n', cnv_path, size(gray,2), size(gray,1));

%% 加载GT（共用）
gtMask = imread('mask_cnv.png');
if size(gtMask, 3) == 3
    gtMask = rgb2gray(gtMask);
end
gtMask = gtMask > 0;

%% [交互ROI] 手动圈定分割区域
USE_ROI = false;  % true=交互圈定, false=用默认裁剪
roi_mask = true(size(gray));
if USE_ROI
    fprintf('\n--- [交互ROI] 请圈定 CNV 病灶区域 ---\n');
    fprintf('用鼠标画一个闭合区域包住病灶 (双击确认)\n');
    figure('Name', '圈定分割区域', 'NumberTitle', 'off');
    imshow(gray, []); title('画闭合区域包住 CNV 病灶 (双击确认)');
    h_roi = drawfreehand;
    roi_mask = h_roi.createMask();
    saveas(gcf, fullfile(fig_dir, 'exp9_task2_fig1.png'));
    fprintf('图片已保存: exp9_task2_fig1.png\n');
    close(gcf);
    roi_mask = imdilate(roi_mask, strel('disk', 10));
    fprintf('ROI 已圈定: 区域内 %%dpx (占图像 %%.1f%%)\n', ...
        sum(roi_mask(:)), 100 * sum(roi_mask(:)) / numel(roi_mask));
else
    % 默认裁剪区域 (参考 chap9_2.m: r=140:250, c=100:320)
    default_rect = [100, 140, 221, 111];
    roi_mask = false(size(gray));
    roi_mask(default_rect(2):default_rect(2)+default_rect(4)-1, ...
             default_rect(1):default_rect(1)+default_rect(3)-1) = true;
    fprintf('  默认裁剪区域: [%d:%d, %d:%d]\n', ...
        default_rect(2), default_rect(2)+default_rect(4)-1, ...
        default_rect(1), default_rect(1)+default_rect(3)-1);
end

%% ============================
%  Part A: 传统方法
%  ============================
fprintf('\n===== Part A: 传统方法（改进版） =====\n');

%% 核心 pipeline
fprintf(' [核心] multithresh 双阈值分割...\n');
gray_uint8 = im2uint8(gray);
gray_uint8(~roi_mask) = 0;
th = multithresh(gray_uint8(roi_mask), 2);
T1 = th(1); T2 = th(2);
fprintf('   T1=%d, T2=%d, 阈值(T2-35)=%d\n', T1, T2, T2-35);
BW = gray_uint8 >= (T2 - 35) & roi_mask;

%% 增强形态学后处理
fprintf(' [改进] 增强形态学...\n');
BW_filled = imfill(BW, 'holes');
fprintf('   孔洞填充: +%dpx\n', sum(BW_filled(:))-sum(BW(:)));

CC = bwconncomp(BW_filled);
numPixels = cellfun(@numel, CC.PixelIdxList);
BW_max = false(size(BW_filled));
if ~isempty(numPixels)
    [~, idx] = max(numPixels);
    BW_max(CC.PixelIdxList{idx}) = true;
end
fprintf('   最大连通域: %dpx\n', sum(BW_max(:)));

se = strel('disk', 3);
BW_erode = ~imerode(~BW_max, se);
fprintf('   黑区腐蚀: -%dpx\n', sum(BW_max(:))-sum(BW_erode(:)));
resultMask_A = bwareaopen(BW_erode, 20);
fprintf('   去除小区域: -%dpx\n', sum(BW_erode(:))-sum(resultMask_A(:)));

%% Part A 评估
intersection_A = sum(resultMask_A(:) & gtMask(:));
union_area_A   = sum(resultMask_A(:) | gtMask(:));
IoU_A = intersection_A / union_area_A;
dice_A = 2 * intersection_A / (sum(resultMask_A(:)) + sum(gtMask(:)));
fprintf('\n>> Part A 结果: IoU = %.4f, Dice = %.4f\n', IoU_A, dice_A);

%% ============================
%  Part B: U-Net (加载预训练模型 + 最优参数推理)
%  ============================
fprintf('\n===== Part B: U-Net (加载预训练模型 + 最优参数推理) =====\n');
fprintf('  模型来源: task2_partB_unet.m 训练 (含组B调参)\n');
fprintf('  后处理参数: tune_unet_params_groupA.m 网格搜索 (组A调参)\n\n');

img_u8   = im2uint8(gray);
ROOT_DIR = fileparts(mfilename('fullpath'));

%% 加载模型
model_path = fullfile(ROOT_DIR, 'unet_cnv_model.mat');
if ~exist(model_path, 'file')
    fprintf(' [提示] 模型文件不存在，请先运行 task2_partB_unet.m 训练\n');
    fprintf('   回退到 Part A 仅对比模式\n');
    resultMask_B = [];
    IoU_B = NaN; dice_B = NaN;
else
    load(model_path, 'net');
    fprintf(' [模型] 已加载: %s\n', model_path);

    %% 读取训练参数（PATCH_SIZE 等）
    paramsB_path = fullfile(ROOT_DIR, 'best_params_groupB.mat');
    if exist(paramsB_path, 'file')
        load(paramsB_path, 'best_params_B');
        if isfield(best_params_B, 'PATCH_SIZE')
            STRIDE = 16; PATCH_SIZE = best_params_B.PATCH_SIZE;
        else
            STRIDE = 16; PATCH_SIZE = 64;
        end
        fprintf(' [参数] 加载训练参数: PATCH_SIZE=%d, STRIDE=%d\n', PATCH_SIZE, STRIDE);
    else
        STRIDE = 16; PATCH_SIZE = 64;
        fprintf(' [参数] 未找到best_params_groupB，使用默认: PATCH_SIZE=%d\n', PATCH_SIZE);
    end
    fprintf(' [推理] 滑窗预测 (stride=%d, patch=%d)...\n', STRIDE, PATCH_SIZE);
    [h_b, w_b] = size(img_u8);
    score_map = zeros(h_b, w_b);
    weight_map = zeros(h_b, w_b);

    for r = 1:STRIDE:h_b-PATCH_SIZE+1
        for c = 1:STRIDE:w_b-PATCH_SIZE+1
            patch = single(img_u8(r:r+PATCH_SIZE-1, c:c+PATCH_SIZE-1));
            p = predict(net, patch);
            s = p(:,:,2);
            score_map(r:r+PATCH_SIZE-1,c:c+PATCH_SIZE-1) = ...
                score_map(r:r+PATCH_SIZE-1,c:c+PATCH_SIZE-1) + s;
            weight_map(r:r+PATCH_SIZE-1,c:c+PATCH_SIZE-1) = ...
                weight_map(r:r+PATCH_SIZE-1,c:c+PATCH_SIZE-1) + 1;
        end
    end
    weight_map(weight_map == 0) = 1;
    score_map = score_map ./ weight_map;

    %% 高斯平滑 + 二值化 + 后处理
    score_map = imgaussfilt(score_map, 1.5);
    resultMask_B = score_map > 0.5;
    resultMask_B = bwareaopen(resultMask_B, 50);           % 去小噪点
    resultMask_B = imclose(resultMask_B, strel('disk', 3)); % 闭运算弥合小洞
    resultMask_B = imfill(resultMask_B, 'holes');           % 填充孔洞
    resultMask_B = resultMask_B & roi_mask;                 % 限制在 ROI 内
    L_B = bwlabel(resultMask_B);                            % 取最大连通域
    s_B = regionprops(L_B, 'Area');
    if ~isempty(s_B)
        [~, mi] = max([s_B.Area]);
        resultMask_B = (L_B == mi);
    end

    %% 评估
    inter_B = sum(resultMask_B(:) & gtMask(:));
    union_B = sum(resultMask_B(:) | gtMask(:));
    IoU_B   = inter_B / union_B;
    dice_B  = 2 * inter_B / (sum(resultMask_B(:)) + sum(gtMask(:)));
    fprintf(' [评估] >> Part B: IoU = %.4f, Dice = %.4f\n', IoU_B, dice_B);
end

%% 对比显示
fprintf('\nPart A: IoU=%.4f, Dice=%.4f\n', IoU_A, dice_A);
if exist('resultMask_B','var') && ~isempty(resultMask_B)
    fprintf('Part B: IoU=%.4f, Dice=%.4f\n', IoU_B, dice_B);
end

figure('Name','任务2 CNV分割','NumberTitle','off','Position',[100 100 1200 500]);
subplot(1,3,1); imshow(gray,[]); title('原图像');
subplot(1,3,2);
imshow(labeloverlay(gray,resultMask_A,'Transparency',0.6));
title(sprintf('Part A: 传统方法 IoU=%.4f',IoU_A));
if exist('resultMask_B','var') && ~isempty(resultMask_B)
    subplot(1,3,3);
    imshow(labeloverlay(gray,resultMask_B,'Transparency',0.6));
    title(sprintf('Part B: U-Net IoU=%.4f',IoU_B));
else
    subplot(1,3,3); imshow(gray,[]); title('U-Net不可用');
end
saveas(gcf, fullfile(fig_dir, 'exp9_task2_fig2.png'));
fprintf('图片已保存: exp9_task2_fig2.png\n');
fprintf('\n【任务2完成】\n');
fprintf('========================================\n');
fprintf('        任务2 完成!\n');
fprintf('========================================\n');

%% ========================================================================
%  辅助函数
% ========================================================================
function iou = computeIoU(mask, gt)
    % COMPUTEIOU 计算二值掩模与 GT 的 Intersection-over-Union
    inter = sum(mask(:) & gt(:));
    union = sum(mask(:) | gt(:));
    if union > 0
        iou = inter / union;
    else
        iou = -1;
    end
end
function gt = loadGroundTruth(path)
    % LOADGROUNDTRUTH 加载并二值化 GT 掩模
    %   文件不存在时返回 [] 并打印警告
    if ~exist(path, 'file')
        fprintf('注意: 未找到ground truth (%s)，跳过IoU计算\n', path);
        gt = [];
        return;
    end
    gt = imread(path) > 128;
    if size(gt, 3) > 1
        gt = rgb2gray(gt) > 128;
    end
end
function demoPause(step_name)
    global DEMO_MODE;
    fprintf('\n=== [演示步进] %s ===\n', step_name);
    if DEMO_MODE
        fprintf('按任意键继续...\n');
        pause;
    end
end
