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

%% 实验9: 图像分割 — 任务1: 遥感图控点分割
% 课程: 视觉与数据计算
% 重点函数: edge, houghlines, graythresh, multithresh, otsuthresh, adaptthresh
%          watershed, imsegkmeans, superpixels, grabcut, activecontour
clear all;
close all;
clc;
fig_dir = fullfile(fileparts(mfilename('fullpath')), 'figs_task1');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
global DEMO_MODE;
DEMO_MODE = false;
fprintf('========================================\n');
fprintf('          实验9: 图像分割 — 任务1\n');
fprintf('========================================\n\n');

%% ========================================================================
%  任务1: 遥感图控点分割
%  ========================================================================
fprintf('【任务1】遥感图控点分割\n');
fprintf('----------------------------------------\n');
% 读入control图像（彩色，L型标记线为红色）
control_path = 'control.jpg';
control = imread(control_path);
demoPause('任务1: 加载图像 — 遥感图控点（红色 L 型标记线）');

%% 步骤1a: HSV 颜色阈值分割
% 注意：原方法使用R/(G+B)比值>1.5的阈值，但实际标记线
% 的R/(G+B)最大值仅为1.07(均值0.74)，导致阈值过高，
% 分割掩模为空，Hough变换无法检测到任何直线。
% 改进：使用R-max(G,B)色差，同时抑制G和B通道，比R-G更纯净。
% 标记线与背景的分离度约为3倍。
control_double = im2double(control);
R = control_double(:,:,1);
G = control_double(:,:,2);
B = control_double(:,:,3);

%% 方法1: HSV颜色阈值分割（参考方案方法）
% HSV将颜色分解为色调(H)、饱和度(S)、明度(V)，红色在H≈0°/360°
% 判断红色像素时只看H是否在红色范围、S够高（非灰）、V够亮（非黑）
hsv_img = rgb2hsv(control);
H_ch = hsv_img(:,:,1);
S_ch = hsv_img(:,:,2);
V_ch = hsv_img(:,:,3);
% 红色范围：H∈(0.92,1.0]∪[0,0.08)，S>0.35排除灰色，V>0.2排除过暗
hsv_mask = (H_ch > 0.92 | H_ch < 0.08) & (S_ch > 0.35) & (V_ch > 0.2);
hsv_mask = morphPostProcess(hsv_mask, 0, 30, true);
% morphPostProcess: close(跳过) → bwareaopen(30) → imfill → bwareafilt(取最大连通域)
fprintf('HSV阈值法: H∈0.92~1|0~0.08 ∩ S>0.35 ∩ V>0.2, 前景像素=%d\n', sum(hsv_mask(:)));

%% 方法2: 基于R-max(G,B)色差的红色标记线提取（几何预处理步骤）
% R-max(G,B)同时抑制G和B通道，比R-G更纯净（避免蓝紫色误检为红）
% 与HSV阈值法构成"颜色法 vs 几何法"的对比
red_diff = R - max(G, B);
red_diff = max(red_diff, 0);  % 截断负值，非红色区域为0
% 使用自适应阈值：均值 + k * 标准差（基于标记线像素占比极小）
diff_mean = mean(red_diff(:));
diff_std = std(red_diff(:));
k_factor = 5.0;  % 提高阈值减少误检（原4.0）
threshold_rg = diff_mean + k_factor * diff_std;
fprintf('R-max(G,B)色差: 均值=%.4f, 标准差=%.4f, 阈值(μ+%.1fσ)=%.4f\n', ...
    diff_mean, diff_std, k_factor, threshold_rg);

%% 步骤1b: R-max(G,B) 色差阈值分割
% R-max(G,B)同时抑制G和B通道，比R-G更纯净
demoPause('任务1-1: HSV 阈值法 + R-max(G,B) 色差法 — 两种红色标记线提取');
% 阈值分割提取红色标记线
red_mask = red_diff > threshold_rg;
% 形态学后处理：闭运算连接断裂线段，去除小噪点
red_mask = morphPostProcess(red_mask, 1, 50, false);
% morphPostProcess: close(disk=1) → bwareaopen(50)

%% 步骤1c: HSV+Lab双空间融合分割（补充方法）
demoPause('任务1-2: HSV+Lab 双空间融合 — 4种策略自动评分选择最优');
% 在R-G色差基础上，增加HSV色调和Lab颜色空间的多策略融合
fprintf('\n--- HSV+Lab双空间融合分割 ---\n');
[fusion_mask, fusion_label, fusion_results, fusion_metrics] = ...
    HSVLabFusionSeg(control_double, false);
fprintf('4种策略评分:\n');
f_names = fieldnames(fusion_results);
for fi = 1:length(f_names)
    nm = f_names{fi};
    star = '';
    if strcmp(nm, fusion_label), star = ' ★'; end
    fprintf('  %s: 总分=%.4f (紧致=%.3f, 边缘=%.3f, 面积=%.3f, Dice=%.3f, 平滑=%.3f)%s\n', ...
        nm, fusion_metrics.(nm).total_score, ...
        fusion_metrics.(nm).compactness, ...
        fusion_metrics.(nm).edge_overlap, ...
        fusion_metrics.(nm).area_score, ...
        fusion_metrics.(nm).dice_w_candidate, ...
        fusion_metrics.(nm).smoothness, star);
end
fprintf('自动选择最优策略: 「%s」\n', fusion_label);
% 计算融合结果与GT的IoU（后面有GT加载时一起计算）

%% 步骤1d: 对比方法 — 边缘检测算子 + 阈值分割
demoPause('任务1-3: 对比方法 — Canny/Sobel/LoG 边缘检测 + Otsu/自适应阈值分割');

%% 对比方法（在灰度图上）
control_gray = rgb2gray(control);
control_gray = im2double(control_gray);
% 方法1: Canny边缘检测
edges_canny = edge(control_gray, 'canny', [0.05 0.15]);
% 方法2: Sobel边缘检测
edges_sobel = edge(control_gray, 'sobel', 0.05);
% 方法3: Otsu阈值分割
level = graythresh(control_gray);
bw_otsu = imbinarize(control_gray, level);
% 方法4: 自适应阈值
bw_adaptive = imbinarize(control_gray, 'adaptive', 'Sensitivity', 0.4);
% 方法5: LoG (Laplacian of Gaussian) 边缘检测
% 零交叉检测器，对二阶导数过零点定位边缘，对噪声敏感
edges_log = edge(control_gray, 'log');

%% otsuthresh演示：从直方图计算Otsu阈值（与graythresh对比）
[counts, ~] = imhist(control_gray);
T_otsu = otsuthresh(counts);  % 返回[0,1]归一化阈值
fprintf('  otsuthresh阈值=%.4f (graythresh阈值=%.4f, 差值=%.4f)\n', ...
    T_otsu, level, abs(T_otsu - level));

%% 步骤1e: imsegkmeans 聚类分割（在彩色原图上）
demoPause('任务1-4: imsegkmeans 聚类 + superpixels 超像素 — 无监督分割方法');
fprintf('\n--- imsegkmeans聚类分割 ---\n');
k_clusters = 4;  % 用更多簇分离细小的红色标记线
kmeans_labels = imsegkmeans(control, k_clusters);
% 自动识别标记线类：R/(G+B)比值最高 + 面积最小的类（标记线为细线）
cluster_redness = zeros(k_clusters, 1);
cluster_area = zeros(k_clusters, 1);
for ci = 1:k_clusters
    px_mask = (kmeans_labels == ci);
    cluster_area(ci) = sum(px_mask(:));
    cluster_redness(ci) = mean(R(px_mask)) / (mean(G(px_mask)) + mean(B(px_mask)) + eps);
end
% 标记线应该是红色比值高且面积小的类
redness_rank = tiedrank(-cluster_redness);  % 红色比值排名（1=最红）
area_rank = tiedrank(cluster_area);           % 面积排名（1=最小）
[~, marker_cluster] = min(redness_rank + area_rank);  % 综合排名
mask_kmeans = (kmeans_labels == marker_cluster);
% 形态学清理
mask_kmeans = morphPostProcess(mask_kmeans, 1, 30, false);
% morphPostProcess: close(disk=1) → bwareaopen(30) → imfill
fprintf('聚类数=%d, 标记线类ID=%d, 前景像素=%d\n', ...
    k_clusters, marker_cluster, sum(mask_kmeans(:)));
for ci = 1:k_clusters
    fprintf('  类%d: 红色比值=%.3f, 面积=%dpx\n', ci, ...
        cluster_redness(ci), cluster_area(ci));
end

%% superpixels 超像素分割
fprintf('\n--- superpixels超像素分割 ---\n');
num_superpixels = 500;
[sp_labels, sp_num] = superpixels(control, num_superpixels);
fprintf('请求超像素数=%d, 实际生成=%d\n', num_superpixels, sp_num);

%% 步骤1f: Hough直线检测 + L型标记线配准
demoPause('任务1-5: Canny+Hough 直线检测 + 角度聚类贪心配准 L 型标记');

%% 方法2续: Canny边缘 + Hough直线检测（HSV引导几何法）
% 用HSV掩模膨胀后约束Canny边缘（颜色法引导几何法），
% 只保留红色区域附近的边缘线段，再通过HSV重叠验证过滤误检
rdiff_norm = red_diff / (max(red_diff(:)) + eps);  % 归一化到[0,1]
% Canny边缘检测（适度阈值，配合HSV引导去除背景纹理）
edgeR = edge(rdiff_norm, 'canny', [0.02 0.10]);
% 用HSV结果适度膨胀后约束Canny边缘——disk(5)覆盖标记线周边合理区域
hsv_guide = imdilate(hsv_mask, strel('disk', 5));
edgeMask = edgeR & hsv_guide;
% Hough变换检测直线（提高峰值阈值减少假直线）
[H_mat, theta, rho] = hough(edgeMask, 'Theta', -90:0.5:89.5);
peaks = houghpeaks(H_mat, 15, 'NHoodSize', [51, 51], 'Threshold', 0.3*max(H_mat(:)));
lines = houghlines(edgeMask, theta, rho, peaks, 'FillGap', 20, 'MinLength', 15);
fprintf('使用图像: %s\n', control_path);
fprintf('改进方法: R-max(G,B)色差 + Canny + HSV引导 + Hough + L型配准\n');
fprintf('Canny边缘（HSV引导后）: 前景=%d, 检测到 %d 条线段\n', sum(edgeMask(:)), length(lines));
% 只保留与HSV掩模有实际重叠的线段（颜色法二次验证）
[M_h, N_h] = size(hsv_mask);
valid_l = false(length(lines), 1);
for k = 1:length(lines)
    xs = round(linspace(lines(k).point1(1), lines(k).point2(1), 50));
    ys = round(linspace(lines(k).point1(2), lines(k).point2(2), 50));
    xs = max(1, min(N_h, xs));
    ys = max(1, min(M_h, ys));
    valid_l(k) = any(hsv_mask(sub2ind([M_h, N_h], ys, xs)));
end
lines = lines(valid_l);
fprintf('HSV重叠验证后保留 %d 条线段\n', length(lines));

%% L型标记线配准：将Hough线段配为完整的L型标记
% 利用Hough检测到的线段，通过角度聚类将线段分为水平和垂直两组，
% 用贪心匹配算法对所有水平线和垂直线进行配对，组装为完整L型标记。
% 用贪心匹配算法：对每条水平线找最近的未配垂直线，组装为完整L型标记。
% 容短缺臂，失败时回退到边界法。
l_markers = [];  % [corner_x, corner_y, h_idx, v_idx, end_dist]
corner_points = [];
corner_quality = -1;
n_lines = length(lines);
used_h = false(n_lines, 1);
used_v = false(n_lines, 1);
if ~isempty(lines)
    % 从Hough线段计算方向角
    line_angles = zeros(n_lines, 1);
    line_pts = zeros(n_lines, 4);  % [x1, y1, x2, y2]
    line_lens = zeros(n_lines, 1);
    for k = 1:n_lines
        p1 = lines(k).point1;
        p2 = lines(k).point2;
        line_pts(k, :) = [p1, p2];
        dx = p2(1) - p1(1);
        dy = p2(2) - p1(2);
        line_angles(k) = atan2d(dy, dx);  % [-180, 180]
        line_lens(k) = sqrt(dx^2 + dy^2);
    end
    % 角度归一化到[0,180)用于聚类
    % L型标记线由一组近似水平的臂(→0°或←180°)和一组近似垂直的臂组成
    angles_mod = mod(line_angles, 180);
    % 水平组：[0,45)或(135,180); 垂直组：[45,135]
    % 使用更宽松的角度范围（原[0,30)∪(150,180]和[60,120]），
    % 以适应控制点图像中L型标记线可能有旋转的情况
    horiz_idx = (angles_mod < 45) | (angles_mod > 135);
    vert_idx  = (angles_mod >= 45) & (angles_mod <= 135);
    if any(horiz_idx) && any(vert_idx)
        % --- 贪心匹配：每条水平线配最近的未用垂直线 ---
        for hi = find(horiz_idx)'
            if used_h(hi), continue; end
            hp1 = line_pts(hi, 1:2);
            hp2 = line_pts(hi, 3:4);
            best_dist = inf;
            best_vi = -1;
            best_corner_pt = [];
            for vi = find(vert_idx)'
                if used_v(vi), continue; end
                vp1 = line_pts(vi, 1:2);
                vp2 = line_pts(vi, 3:4);
                % 计算4个端点间的最小距离
                dd = [norm(hp1 - vp1), norm(hp1 - vp2), norm(hp2 - vp1), norm(hp2 - vp2)];
                d = min(dd);
                if d < best_dist && d < 25
                    best_dist = d;
                    best_vi = vi;
                    % 角点取最近的水平端点和垂直端点的平均
                    [~, min_idx] = min(dd);
                    if min_idx <= 2
                        h_pt = hp1;
                    else
                        h_pt = hp2;
                    end
                    best_corner_pt = (h_pt + (vp1 + vp2) / 2) / 2;
                end
            end
            if best_vi > 0
                used_h(hi) = true;
                used_v(best_vi) = true;
                l_markers = [l_markers; best_corner_pt, hi, best_vi, best_dist]; %#ok<AGROW>
                fprintf('  L型标记#%d: 角点=(%.1f,%.1f), 端距=%.1fpx\n', ...
                    size(l_markers,1), best_corner_pt(1), best_corner_pt(2), best_dist);
            end
        end
        corner_points = l_markers(:, 1:2);  % 所有角点
        if ~isempty(corner_points)
            corner_quality = mean(1 - l_markers(:,5) / 25);  % 平均端距质量 [0,1]
            fprintf('检测到 %d 个完整L型标记线 (未配对的水平线=%d, 垂直线=%d)\n', ...
                size(l_markers,1), sum(horiz_idx & ~used_h), sum(vert_idx & ~used_v));
        end
    end
end
% 回退方案：无配对的线时使用边界法
if isempty(l_markers) && sum(red_mask(:)) > 50
    boundaries = bwboundaries(red_mask);
    if ~isempty(boundaries)
        boundary_sizes = cellfun(@(x) size(x, 1), boundaries);
        [~, max_idx] = max(boundary_sizes);
        boundary = boundaries{max_idx};
        rows_b = double(boundary(:,1));
        cols_b = double(boundary(:,2));
        rows_norm = (rows_b - min(rows_b)) / (max(rows_b) - min(rows_b) + eps);
        cols_norm = (cols_b - min(cols_b)) / (max(cols_b) - min(cols_b) + eps);
        scores = (1 - rows_norm) + cols_norm;
        [~, corner_idx] = max(scores);
        corner_points = [cols_b(corner_idx), rows_b(corner_idx)];
        fprintf('  [回退边界法] 角点=(%.1f,%.1f)\n', ...
            corner_points(1), corner_points(2));
    end
end

%% 步骤1g: 评估指标计算 — IoU + Precision + Recall
demoPause('任务1-6: 评估指标 — IoU/Precision/Recall + 方法间对比 + Hausdorff距离');

%% 计算IoU与评估指标
% 对于细线目标，标准IoU过于严苛，同时计算膨胀GT后的IoU
iou_raw = -1;
iou_dilated = -1;
precision_val = -1;
recall_val = -1;
iou_fusion = -1;       % HSV+Lab融合结果IoU
iou_kmeans = -1;       % imsegkmeans结果IoU
gt1 = loadGroundTruth('mask_l.png');
if ~isempty(gt1)
    % GT为轮廓图（仅125px），imfill填充内部得到真实标记区域
    gt1 = imfill(gt1, 'holes');
    % 各方法的IoU（基于computeIoU复用）
    iou_hsv_gt  = computeIoU(hsv_mask, gt1);
    iou_raw     = computeIoU(red_mask, gt1);
    iou_fusion  = computeIoU(fusion_mask, gt1);
    iou_kmeans  = computeIoU(mask_kmeans, gt1);
    % 膨胀GT后的IoU（1px容差，对填充GT做边界补偿）
    gt1_dilated = imdilate(gt1, strel('disk', 1));
    iou_dilated = computeIoU(red_mask, gt1_dilated);
    % Precision与Recall
    [precision_val, recall_val] = computePrecisionRecall(red_mask, gt1);
    % Hausdorff距离 (各方法与GT的边界偏差)
    hd_hsv_gt  = hausdorff_dist(logical(hsv_mask), gt1);
    hd_rg_gt   = hausdorff_dist(logical(red_mask), gt1);
    hd_fusion_gt = hausdorff_dist(logical(fusion_mask), gt1);
    hd_kmeans_gt = hausdorff_dist(logical(mask_kmeans), gt1);
    fprintf('IoU (HSV阈值 vs GT):  %.4f  HD=%.1fpx\n', iou_hsv_gt, hd_hsv_gt);
    fprintf('IoU (R-G色差 vs GT):  %.4f  HD=%.1fpx\n', iou_raw, hd_rg_gt);
    fprintf('IoU (GT膨胀3×3):     %.4f\n', iou_dilated);
    fprintf('IoU (HSV+Lab融合):   %.4f  HD=%.1fpx\n', iou_fusion, hd_fusion_gt);
    fprintf('IoU (imsegkmeans):   %.4f  HD=%.1fpx\n', iou_kmeans, hd_kmeans_gt);
    fprintf('Precision:          %.4f\n', precision_val);
    fprintf('Recall:             %.4f\n', recall_val);
    fprintf('Hausdorff距离 (HSV vs GT):  %.1fpx\n', hd_hsv_gt);
    fprintf('Hausdorff距离 (R-G vs GT):  %.1fpx\n', hd_rg_gt);
    fprintf('Hausdorff距离 (融合 vs GT): %.1fpx\n', hd_fusion_gt);
    fprintf('Hausdorff距离 (K-means vs GT): %.1fpx\n', hd_kmeans_gt);
else
    iou_hsv_gt = -1;
end

%% 方法间IoU对比：HSV阈值法 vs R-G色差法（不同原理方法的互相印证）
iou_method12 = computeIoU(hsv_mask, red_mask);
fprintf('\n--- 方法间对比（颜色法 vs 几何法） ---\n');
fprintf('IoU (HSV阈值 vs R-G色差): %.4f\n', iou_method12);
if ~isempty(gt1)
    fprintf('IoU (HSV vs GT): %.4f | IoU (R-G vs GT): %.4f | 方法间: %.4f\n', ...
        iou_hsv_gt, iou_raw, iou_method12);
end
% Hausdorff距离：HSV掩模与R-G色差掩模的边界最大偏差
hd_hsv_rg = hausdorff_dist(logical(hsv_mask), logical(red_mask));
fprintf('Hausdorff距离 (HSV vs R-G色差): %.2fpx\n', hd_hsv_rg);

%% 步骤1h: 结果可视化
demoPause('任务1-7: 可视化 — 各方法分割结果对比 + L型标记线检测展示');

%% 可视化：3×4子图，增加LoG、imsegkmeans和superpixels
figure('Name', '任务1: 遥感图控点分割', 'NumberTitle', 'off', ...
    'Position', [50, 50, 1600, 900]);
% === 第1行: 改进方法的完整流程 ===
subplot(3, 4, 1);
imshow(control);
title('原图像（彩色）');
subplot(3, 4, 2);
imshow(hsv_mask, []);
title(sprintf('HSV阈值法 (IoU=%.3f)', iou_hsv_gt));
subplot(3, 4, 3);
imshow(red_mask, []);
title(sprintf('R-max(G,B)色差 (IoU=%.3f)', iou_raw));
subplot(3, 4, 4);
imshow(control);
hold on;
% 绿色边界 = HSV阈值掩模轮廓
B_hsv = bwboundaries(hsv_mask);
for k = 1:length(B_hsv)
    plot(B_hsv{k}(:,2), B_hsv{k}(:,1), 'g-', 'LineWidth', 1.5);
end
% 红色边界 = R-G色差掩模轮廓
B_rg = bwboundaries(red_mask);
for k = 1:length(B_rg)
    plot(B_rg{k}(:,2), B_rg{k}(:,1), 'r-', 'LineWidth', 1.5);
end
% 画每个完整的L型标记（彩色臂+黄色角点）
if ~isempty(l_markers)
    marker_colors = jet(size(l_markers, 1));
    for mi = 1:size(l_markers, 1)
        hi = l_markers(mi, 3);
        vi = l_markers(mi, 4);
        cx = l_markers(mi, 1);
        cy = l_markers(mi, 2);
        % 画水平臂（加粗）
        plot(line_pts(hi, [1,3]), line_pts(hi, [2,4]), 'Color', marker_colors(mi,:), 'LineWidth', 3);
        % 画垂直臂（加粗）
        plot(line_pts(vi, [1,3]), line_pts(vi, [2,4]), 'Color', marker_colors(mi,:), 'LineWidth', 3);
        % 画角点（黄色圈，示例风格）
        plot(cx, cy, 'yo', 'MarkerSize', 10, 'LineWidth', 2);
        % 标注L1, L2, ...
        text(cx+8, cy-8, sprintf('L%d', mi), 'Color', 'y', ...
            'FontWeight', 'bold', 'FontSize', 12);
    end
end
% 画未配对的线（灰色虚线，淡化显示）
for k = 1:length(lines)
    if ~used_h(k) && ~used_v(k)
        xy = [line_pts(k,1:2); line_pts(k,3:4)];
        plot(xy(:,1), xy(:,2), '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1);
    end
end
title(sprintf('绿=HSV 红=R-G 黄=L角点 IoU=%.3f HD=%.1fpx', iou_method12, hd_hsv_rg));
hold off;
% === 第2行: 边缘检测算子对比 ===
subplot(3, 4, 5);
imshow(edges_canny, []);
title('Canny边缘检测');
subplot(3, 4, 6);
imshow(edges_sobel, []);
title('Sobel边缘检测');
subplot(3, 4, 7);
imshow(edges_log, []);
title('LoG边缘检测');
subplot(3, 4, 8);
imshow(bw_adaptive, []);
title('自适应阈值分割');
% === 第3行: 阈值分割 + 聚类 + 超像素 ===
subplot(3, 4, 9);
imshow(bw_otsu, []);
title(sprintf('Otsu阈值 (otsuthresh=%.3f)', T_otsu));
subplot(3, 4, 10);
imshow(mask_kmeans, []);
title(sprintf('imsegkmeans聚类 (IoU=%.3f)', iou_kmeans));
subplot(3, 4, 11);
sp_boundary = boundarymask(sp_labels);
imshow(imoverlay(control, sp_boundary, 'cyan'));
title(sprintf('superpixels超像素边界 (N=%d)', sp_num));
subplot(3, 4, 12);
% IoU条状图对比各方法（颜色法 vs 几何法 + 融合/聚类）
methods_names = {'HSV', 'R-max(G,B)', 'HSV+Lab', 'K-means'};
iou_vals = [iou_hsv_gt; iou_raw; iou_fusion; iou_kmeans];
% 方法间IoU用虚线标注
valid_idx = iou_vals > 0;
if any(valid_idx)
    bar(categorical(methods_names(valid_idx)), iou_vals(valid_idx));
    ylabel('IoU'); ylim([0 1]); grid on;
    title('各方法IoU对比');
else
    imshow(control); title('IoU对比 (无GT)');
end
saveas(gcf, fullfile(fig_dir, 'exp9_task1_fig1.png'));
fprintf('图片已保存: exp9_task1_fig1.png\n');

%% 步骤1h(续): HSV+Lab双空间融合分割对比图
demoPause('任务1-8: HSV+Lab 融合分割可视化 — 4种策略在原图上叠加边界');

%% HSV+Lab双空间融合分割对比图
figure('Name', 'HSV+Lab双空间融合分割 (任务1)', 'NumberTitle', 'off', ...
    'Position', [100, 100, 1300, 700]);
% 第1行: 4种策略的掩模
subplot(2, 4, 1); imshow(fusion_results.HSV);
title(sprintf('① HSV最优 S=%.3f', fusion_metrics.HSV.total_score));
subplot(2, 4, 2); imshow(fusion_results.Lab);
title(sprintf('② Lab Otsu S=%.3f', fusion_metrics.Lab.total_score));
subplot(2, 4, 3); imshow(fusion_results.HSV_n_Lab);
title(sprintf('③ HSV∩Lab S=%.3f', fusion_metrics.HSV_n_Lab.total_score));
subplot(2, 4, 4); imshow(fusion_results.HSV_u_Lab);
title(sprintf('④ HSV∪Lab S=%.3f', fusion_metrics.HSV_u_Lab.total_score));
% 第2行: 在原图上叠加掩模边界
subplot(2, 4, 5);
imshow(control); hold on;
visboundaries(bwperim(fusion_results.HSV), 'Color', 'r', 'LineWidth', 1.5);
title('① HSV最优'); hold off;
subplot(2, 4, 6);
imshow(control); hold on;
visboundaries(bwperim(fusion_results.Lab), 'Color', 'g', 'LineWidth', 1.5);
title('② Lab Otsu'); hold off;
subplot(2, 4, 7);
imshow(control); hold on;
visboundaries(bwperim(fusion_results.HSV_n_Lab), 'Color', 'b', 'LineWidth', 1.5);
title('③ HSV∩Lab'); hold off;
subplot(2, 4, 8);
imshow(control); hold on;
visboundaries(bwperim(fusion_results.HSV_u_Lab), 'Color', 'm', 'LineWidth', 1.5);
visboundaries(bwperim(fusion_results.(fusion_label)), 'Color', 'y', 'LineWidth', 2.5);
title(sprintf('④ HSV∪Lab (★最优=%s IoU=%.3f)', fusion_label, iou_fusion));
hold off;
sgtitle(sprintf('HSV+Lab双空间融合分割 — 自动选择「%s」策略 (IoU=%.3f)', fusion_label, iou_fusion));
saveas(gcf, fullfile(fig_dir, 'exp9_task1_fig2.png'));
fprintf('图片已保存: exp9_task1_fig2.png');
fprintf('\n分割方法比较完成\n');
fprintf('  [方法1 HSV阈值] 基于颜色空间的分割：H通道精准定位红色(0.92~1|0~0.08)，\n');
fprintf('          S>0.35排除灰白区域，V>0.2排除暗区。取最大连通域排除误检\n');
fprintf('  [方法2 R-max(G,B)色差] 同时抑制G/B通道，比R-G更纯净(避免蓝紫色误检)。\n');
fprintf('          经Canny+HSV引导→Hough变换检测线段，贪心配准组装为完整L型标记\n');
fprintf('  Canny: 边缘连续性好，但对细小红线的响应较弱(灰度图中红色变暗)\n');
fprintf('  Sobel: 计算简单，边缘较粗\n');
fprintf('  LoG: 二阶导数零交叉检测，对噪声敏感但定位精确\n');
fprintf('  Otsu: 自动确定阈值，但对细线目标分割效果有限\n');
fprintf('  自适应: 适合光照不均匀的图像\n');
fprintf('  imsegkmeans: K-means颜色聚类，自动识别红色标记线类\n');
fprintf('  superpixels: SLIC超像素分割，保持L型边界的拓扑结构\n');

%% ========================================================================
%  辅助函数
% ========================================================================
function [best_mask, best_label, results, metrics] = HSVLabFusionSeg(rgb_img, debug_viz)
    % HSV+Lab双空间融合分割
    % 在最优HSV参数确定后，自动用Lab b*通道做Otsu自适应分割，
    % 生成4种策略（HSV最优 / Lab Otsu / HSV∩Lab / HSV∪Lab）并自动选出最优。
    %
    % 输入: rgb_img - 彩色图像 (double, [0,1])
    %       debug_viz - 是否显示中间过程图 (默认false)
    % 输出: best_mask - 自动选出的最优掩模
    %       best_label - 最优策略名称
    %       results - 4种策略的掩模结构体
    %       metrics - 每种策略的评分指标
    %
    % 自动选择标准(加权): 紧致度(0.3)+边缘强度(0.2)+面积适中(0.2)+区域连通性(0.3)
    if nargin < 2, debug_viz = false; end
    img_double = im2double(rgb_img);

    %% ── 步骤1: HSV 空间 ──
    hsv = rgb2hsv(img_double);
    H = hsv(:,:,1);
    S = hsv(:,:,2);
    V_ch = hsv(:,:,3);  % Value通道，用于过滤暗像素
    % 自动检测目标颜色的色调范围
    % 用R-G色差粗略定位目标区域，再从中统计H分布
    R = img_double(:,:,1);
    G = img_double(:,:,2);
    rg_diff = R - G;
    % 自适应阈值找到候选像素
    th_rg = mean(rg_diff(:)) + 2.5 * std(rg_diff(:));
    candidate_mask = rg_diff > th_rg;
    % 如果候选区太小，放宽阈值
    if sum(candidate_mask(:)) < 500
        candidate_mask = rg_diff > (mean(rg_diff(:)) + 1.5 * std(rg_diff(:)));
    end
    % 从候选像素统计H通道分布
    candidate_H = H(candidate_mask);
    if ~isempty(candidate_H)
        % 处理红色环绕（H≈0和H≈1都是红色）
        candidate_H_wrapped = [candidate_H; candidate_H + 1; candidate_H - 1];
        h_low = prctile(candidate_H_wrapped, 5);
        h_high = prctile(candidate_H_wrapped, 95);
        % 归约到[0,1]
        h_range = [mod(h_low, 1), mod(h_high, 1)];
        if h_range(1) > h_range(2)
            % 跨越0点（红色区域）
            mask_hsv = (H >= h_range(1) | H <= h_range(2)) & S > 0.1;
        else
            mask_hsv = (H >= h_range(1) & H <= h_range(2)) & S > 0.1;
        % 增加V通道过滤：排除过暗像素（阴影）和过亮像素（高光）
        bright_enough = V_ch > 0.15 & V_ch < 0.95;
        mask_hsv = mask_hsv & bright_enough;
        end
    else
        mask_hsv = false(size(H));
    end
    % 形态学清理
    mask_hsv = imclose(mask_hsv, strel('disk', 1));
    mask_hsv = bwareaopen(mask_hsv, 30);

    %% ── 步骤2: Lab空间 Otsu 自适应分割 ──
    lab = rgb2lab(img_double);
    a_ch = lab(:,:,2);  % a*: 红-绿轴
    b_ch = lab(:,:,3);  % b*: 蓝-黄轴
    % 红色在a*通道为正，b*通道接近0
    % 对a*通道做Otsu分割
    level_a = graythresh(mat2gray(a_ch));
    mask_lab_a = imbinarize(mat2gray(a_ch), level_a);
    % 对b*通道做Otsu分割，取绝对值（红色在b*≈0附近，可做反向）
    level_b = graythresh(mat2gray(abs(b_ch)));
    mask_lab_b = imbinarize(mat2gray(abs(b_ch)), level_b);
    % a*正阈值（红色区域a*>0）
    a_pos = mat2gray(max(a_ch, 0));
    level_a_pos = graythresh(a_pos(a_pos > 0.01));
    if ~isnan(level_a_pos) && level_a_pos > 0
        mask_lab_a_pos = imbinarize(a_pos, level_a_pos);
    else
        mask_lab_a_pos = mask_lab_a;
    end
    % Lab融合：a*正区域 ∩ b*接近0区域（典型的红色特征）
    mask_lab = mask_lab_a_pos & ~mask_lab_b;
    mask_lab = imclose(mask_lab, strel('disk', 1));
    mask_lab = bwareaopen(mask_lab, 30);

    %% ── 步骤3: 生成4种策略 ──
    % 策略1: HSV最优
    s1 = mask_hsv;
    % 策略2: Lab Otsu
    s2 = mask_lab;
    % 策略3: HSV ∩ Lab
    s3 = mask_hsv & mask_lab;
    % 策略4: HSV ∪ Lab
    s4 = mask_hsv | mask_lab;
    results = struct('HSV', s1, 'Lab', s2, 'HSV_n_Lab', s3, 'HSV_u_Lab', s4);

    %% ── 步骤4: 自动选择最优策略 ──
    names = fieldnames(results);
    n_strat = length(names);
    metrics = struct();
    scores = zeros(n_strat, 1);
    % 权重
    w_compact = 0.25;   % 紧致度（周长面积比，越小越好）
    w_edge    = 0.20;   % 边缘强度（落在图像边缘上的比例）
    w_area    = 0.15;   % 面积适中（不能太大也不能太小）
    w_fill    = 0.20;   % 填充度（与R-G色差候选区的重叠）
    w_smooth  = 0.20;   % 轮廓平滑度
    for i = 1:n_strat
        mask = results.(names{i});
        if sum(mask(:)) < 20
            scores(i) = -inf;
            continue;
        end
        % (a) 紧致度: perimeter²/area, 越小越紧致
        perim = bwperim(mask);
        p_len = sum(perim(:));
        a_area = sum(mask(:));
        compactness = p_len^2 / (a_area + eps);
        compact_score = exp(-compactness / 100);  % [0,1]
        % (b) 边缘强度: 掩模边界与图像边缘的吻合程度
        gray_img = rgb2gray(img_double);
        img_edges = edge(gray_img, 'canny', [0.05 0.15]);
        edge_overlap = sum(perim(:) & img_edges(:)) / (p_len + eps);
        % (c) 面积适中: 与候选区面积比，太接近1或0都不好
        area_ratio = a_area / (sum(candidate_mask(:)) + eps);
        area_score = exp(-(area_ratio - 1)^2 / 0.5);
        % (d) 与R-G色差候选区的重叠 (Dice)
        inter_c = sum(mask(:) & candidate_mask(:));
        dice_c = 2 * inter_c / (sum(mask(:)) + sum(candidate_mask(:)) + eps);
        % (e) 轮廓平滑度: 边界点的曲率方差小
        B = bwboundaries(mask, 8, 'noholes');
        if ~isempty(B)
            [~, max_idx] = max(cellfun(@(x) size(x,1), B));
            boundary = B{max_idx};
            if size(boundary, 1) > 10
                % 计算曲率
                dx = gradient(boundary(:,2));
                dy = gradient(boundary(:,1));
                ddx = gradient(dx);
                ddy = gradient(dy);
                curvature = abs(ddx .* dy - dx .* ddy) ./ ((dx.^2 + dy.^2).^(1.5) + eps);
                smoothness = exp(-std(curvature) * 10);
            else
                smoothness = 0.5;
            end
        else
            smoothness = 0;
        end
        % 加权综合评分
        scores(i) = w_compact * compact_score + ...
                    w_edge    * edge_overlap + ...
                    w_area    * area_score + ...
                    w_fill    * dice_c + ...
                    w_smooth  * smoothness;
        % 保存各维度指标
        metrics.(names{i}) = struct(...
            'compactness', compact_score, ...
            'edge_overlap', edge_overlap, ...
            'area_score', area_score, ...
            'dice_w_candidate', dice_c, ...
            'smoothness', smoothness, ...
            'total_score', scores(i));
    end
    % 选出最优
    [~, best_idx] = max(scores);
    best_label = names{best_idx};
    best_mask = results.(best_label);

    %% ── 步骤5: 可视化 ──
    if debug_viz
        figure('Name', 'HSV+Lab双空间融合分割', 'NumberTitle', 'off', ...
            'Position', [100, 100, 1200, 700]);
        % 第一行: 4种策略的掩模
        subplot(2, 4, 1); imshow(results.HSV);
        title(sprintf('① HSV最优\nS=%.3f', metrics.HSV.total_score));
        subplot(2, 4, 2); imshow(results.Lab);
        title(sprintf('② Lab Otsu\nS=%.3f', metrics.Lab.total_score));
        subplot(2, 4, 3); imshow(results.HSV_n_Lab);
        title(sprintf('③ HSV∩Lab\nS=%.3f', metrics.HSV_n_Lab.total_score));
        subplot(2, 4, 4); imshow(results.HSV_u_Lab);
        title(sprintf('④ HSV∪Lab\nS=%.3f', metrics.HSV_u_Lab.total_score));
        % 第二行: 4种策略在原图上的分割结果
        subplot(2, 4, 5);
        imshow(img_double); hold on;
        visboundaries(bwperim(results.HSV), 'Color', 'r', 'LineWidth', 1.5);
        title('① HSV最优');
        hold off;
        subplot(2, 4, 6);
        imshow(img_double); hold on;
        visboundaries(bwperim(results.Lab), 'Color', 'g', 'LineWidth', 1.5);
        title('② Lab Otsu');
        hold off;
        subplot(3, 4, 7);
        imshow(img_double); hold on;
        visboundaries(bwperim(results.HSV_n_Lab), 'Color', 'b', 'LineWidth', 1.5);
        title('③ HSV∩Lab');
        hold off;
        subplot(3, 4, 8);
        imshow(img_double); hold on;
        visboundaries(bwperim(results.HSV_u_Lab), 'Color', 'm', 'LineWidth', 1.5);
        title('④ HSV∪Lab');
        hold off;
        sgtitle(sprintf('HSV+Lab双空间融合分割 — 自动选择: 「%s」策略', best_label));
        saveas(gcf, fullfile(fileparts(mfilename('fullpath')), 'figs_task1', 'exp9_task1_fig3.png'));
        fprintf('图片已保存: exp9_task1_fig3.png\n');
    end
end
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
function [precision, recall] = computePrecisionRecall(mask, gt)
    % COMPUTEPRECISIONRECALL 计算精确率和召回率
    tp = sum(mask(:) & gt(:));
    fp = sum(mask(:) & ~gt(:));
    fn = sum(~mask(:) & gt(:));
    if (tp + fp) > 0
        precision = tp / (tp + fp);
    else
        precision = 0;
    end
    if (tp + fn) > 0
        recall = tp / (tp + fn);
    else
        recall = 0;
    end
end
function metrics = computeBoundaryMetrics(edge_bw, gt_perim, tolerance)
    % COMPUTEBOUNDARYMETRICS 边缘检测的三指标评估
    %   返回结构体: strictIoU / tolerantIoU / recall
    gt_perim_d1 = imdilate(gt_perim, strel('disk', tolerance));
    inter_s = sum(edge_bw(:) & gt_perim(:));
    union_s = sum(edge_bw(:) | gt_perim(:));
    if union_s > 0, metrics.strictIoU = inter_s / union_s;
    else,           metrics.strictIoU = -1; end
    inter_d1 = sum(edge_bw(:) & gt_perim_d1(:));
    union_d1 = sum(edge_bw(:) | gt_perim_d1(:));
    if union_d1 > 0, metrics.tolerantIoU = inter_d1 / union_d1;
    else,            metrics.tolerantIoU = -1; end
    metrics.recall = sum(edge_bw(:) & gt_perim_d1(:)) / max(sum(gt_perim_d1(:)), 1);
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
function bw = morphPostProcess(bw, disk_radius, min_area, keep_largest)
    % MORPHPOSTPROCESS 形态学后处理管道
    %   imclose → bwareaopen → imfill → 可选 bwareafilt(最大连通域)
    if nargin < 2 || isempty(disk_radius), disk_radius = 1; end
    if nargin < 3 || isempty(min_area),    min_area = 30;   end
    if nargin < 4 || isempty(keep_largest), keep_largest = false; end
    bw = imclose(bw, strel('disk', disk_radius));
    bw = bwareaopen(bw, min_area);
    bw = imfill(bw, 'holes');
    if keep_largest
        bw = bwareafilt(logical(bw), 1);
    end
end
function hd = hausdorff_dist(BW1, BW2)
    % HAUSDORFFDIST 计算两个二值掩模边界之间的 Hausdorff 距离
    b1 = bwperim(BW1);
    b2 = bwperim(BW2);
    if ~any(b1(:)) || ~any(b2(:))
        hd = Inf;
        return;
    end
    D1 = bwdist(b1);
    D2 = bwdist(b2);
    hd = max(max(D2(b1)), max(D1(b2)));
end
function demoPause(step_name)
    % DEMOPAUSE 课堂演示步进暂停函数
    %   由 global DEMO_MODE 控制是否暂停。
    %   DEMO_MODE = true  → 每步暂停，按任意键继续
    %   DEMO_MODE = false → 仅打印步骤名，不暂停
    global DEMO_MODE;
    fprintf('\n=== [演示步进] %s ===\n', step_name);
    if DEMO_MODE
        fprintf('按任意键继续...\n');
        pause;
    end
end
