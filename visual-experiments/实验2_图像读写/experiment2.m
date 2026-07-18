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

%% 实验2: 图像读写与基本操作
% 课程: 视觉与数据计算
% 重点函数: imread, imshow, imwrite, imcrop, imresize, mean2, std2, psnr, ssim
%
% 结构:
%   第一部分 — 基本任务 (实验要求 1-4)
%   第二部分 — 拓展分析 (五种插值方法对比 + IBP超分辨率)

clear all;
close all;
clc;

% 创建图片保存目录
fig_dir = fullfile(fileparts(mfilename('fullpath')), 'figs');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

fprintf('========================================\n');
fprintf('      实验2: 图像读写与基本操作\n');
fprintf('========================================\n\n');

%% 导入图像并统一为double
image_path = 'lena.png';

% 检查文件是否存在
if ~exist(image_path, 'file')
    error('找不到图像文件: %s\n请将 lena.png 放在当前目录下.', image_path);
end

img = imread(image_path);
if size(img, 3) == 3
    img = rgb2gray(img);
end
I = im2double(img);   % [0,1] double，避免uint8截断误差

fprintf('图像加载完成: %s (%d x %d)\n', image_path, size(I, 1), size(I, 2));

%% ================================================================
%  第一部分: 基本任务 (实验要求, 见 full.md 13.2)
%% ================================================================

%% 任务1: 读入图像, 选取子图像, 显示子图像及其第100行曲线
fprintf('\n【任务1】图像裁剪与行数据可视化\n');
fprintf('----------------------------------------\n');

% 截取子图像 (行: 50-170, 列: 60-180)
row_start = 50; row_end = 170;
col_start = 60; col_end = 180;
sub_img = I(row_start:row_end, col_start:col_end);

fprintf('原图像大小: %d x %d\n', size(I, 1), size(I, 2));
fprintf('子图像范围: 行[%d:%d], 列[%d:%d]\n', row_start, row_end, col_start, col_end);
fprintf('子图像大小: %d x %d\n', size(sub_img, 1), size(sub_img, 2));

% 检查子图像是否有第100行 (稳健性)
row_disp = min(100, size(sub_img, 1));
if size(sub_img, 1) < 100
    warning('子图像仅 %d 行, 不足100行, 将显示最后一行代替.', size(sub_img, 1));
end

figure('Name', '任务1: 图像裁剪与行数据', 'NumberTitle', 'off', 'Position', [100, 100, 1200, 400]);

% 第1幅: 原图像 + 子图像位置标注
subplot(1, 3, 1);
imshow(I, []);
hold on;
rectangle('Position', [col_start, row_start, col_end-col_start+1, row_end-row_start+1], ...
          'EdgeColor', 'r', 'LineWidth', 2);
title('原图像 (红框 = 子图像位置)');
hold off;

% 第2幅: 子图像
subplot(1, 3, 2);
imshow(sub_img, []);
title(sprintf('子图像 [%d:%d, %d:%d]', row_start, row_end, col_start, col_end));

% 第3幅: 子图像第100行 (或最后一行) 灰度曲线
subplot(1, 3, 3);
row_data = sub_img(row_disp, :);
plot(row_data * 255, 'b-', 'LineWidth', 1.5);
xlabel('列号'); ylabel('灰度值');
title(sprintf('子图像第%d行灰度曲线', row_disp));
grid on;

% 保存图片
saveas(gcf, fullfile(fig_dir, 'exp2_task1_subimage.png'));
fprintf('图片已保存: exp2_task1_subimage.png\n');

%% 任务2: 图像减采样与放大
fprintf('\n【任务2】图像减采样与放大\n');
fprintf('----------------------------------------\n');

% 减采样4倍 (imresize 默认使用抗混叠滤波器)
lena4s = imresize(I, 0.25);
fprintf('减采样: %d x %d → %d x %d\n', ...
    size(I,1), size(I,2), size(lena4s,1), size(lena4s,2));

% Lanczos3 插值放大4倍回原尺寸 (保存为 lena4l.bmp)
lena4l = imresize(lena4s, size(I), 'lanczos3');
fprintf('放大:   %d x %d → %d x %d (Lanczos3)\n', ...
    size(lena4s,1), size(lena4s,2), size(lena4l,1), size(lena4l,2));

% 保存文件
imwrite(im2uint8(lena4s), 'lena4s.bmp');
imwrite(im2uint8(lena4l), 'lena4l.bmp');
fprintf('已保存: lena4s.bmp (减采样), lena4l.bmp (Lanczos3放大)\n');

%% 任务3: 图像质量评价
fprintf('\n【任务3】图像质量评价\n');
fprintf('----------------------------------------\n');

% 原图像和放大图像的均值
mean_orig = mean2(I);
mean_upsampled = mean2(lena4l);
fprintf('原图像均值:  %.4f (%.2f/255)\n', mean_orig, mean_orig * 255);
fprintf('放大图像均值: %.4f (%.2f/255)\n', mean_upsampled, mean_upsampled * 255);

% 质量指标 PSNR 和 SSIM
psnr_val = psnr(lena4l, I);
ssim_val = ssim(lena4l, I);
fprintf('PSNR = %.4f dB,  SSIM = %.4f\n', psnr_val, ssim_val);

% 一个窗口显示 lena 和 lena4l
figure('Name', '任务3: 原图像 vs 放大图像', 'NumberTitle', 'off', 'Position', [100, 100, 900, 400]);

subplot(1, 2, 1);
imshow(I, []);
title(sprintf('原图像 lena (均值 = %.4f)', mean_orig));

subplot(1, 2, 2);
imshow(lena4l, []);
title({sprintf('放大图像 lena4l (均值 = %.4f)', mean_upsampled); ...
       sprintf('PSNR = %.2f dB,  SSIM = %.4f', psnr_val, ssim_val)});

% 保存图片
saveas(gcf, fullfile(fig_dir, 'exp2_task3_comparison.png'));
fprintf('图片已保存: exp2_task3_comparison.png\n');

%% 任务4: 差图像分析
fprintf('\n【任务4】差图像分析\n');
fprintf('----------------------------------------\n');

% 计算差图像
diff_img = abs(lena4l - I);
fprintf('差图像统计 (值域 [0,1]):  均值 = %.4f,  标准差 = %.4f\n', ...
    mean2(diff_img), std2(diff_img(:)));

% 为清晰显示, 使用 imadjust 拉伸对比度 (stretchlim 自动确定拉伸范围)
diff_display = imadjust(diff_img, stretchlim(diff_img), []);

figure('Name', '任务4: 差图像', 'NumberTitle', 'off', 'Position', [100, 100, 1200, 400]);

subplot(1, 3, 1);
imshow(I, []);
title('原图像 (lena)');

subplot(1, 3, 2);
imshow(lena4l, []);
title(sprintf('放大图像 (lena4l)\nPSNR = %.2f dB', psnr_val));

subplot(1, 3, 3);
imshow(diff_display, []);
title({sprintf('差图像 |lena4l − lena| (对比度增强)'); ...
       sprintf('均值 = %.4f, 标准差 = %.4f', mean2(diff_img), std2(diff_img(:)))});
colorbar;

% 保存图片
saveas(gcf, fullfile(fig_dir, 'exp2_task4_diff.png'));
fprintf('图片已保存: exp2_task4_diff.png\n');

% 分析图像放大的关键和挑战
fprintf('\n【分析】图像放大的关键和挑战:\n');
fprintf('1. 信息损失: 减采样4倍后仅保留 1/16 像素, 放大无法完整恢复丢失细节\n');
fprintf('2. 插值精度: 插值核越大(邻域越宽), 重建质量越高, 计算量也越大\n');
fprintf('   nearest(1x1) < bilinear(2x2) < bicubic(4x4) < lanczos3(6x6)\n');
fprintf('3. 本实验 PSNR = %.2f dB: 反映 Lanczos3 插值重建的总体误差水平\n', psnr_val);
fprintf('4. 残余误差集中在边缘/纹理等高频区域 (见差图像), 这是降采样信息\n');
fprintf('   损失不可完全恢复的根本原因, 也是超分辨率重建试图解决的问题\n');

%% ================================================================
%  第二部分: 拓展分析 (课堂演示补充 — 五种插值方法对比)
%% ================================================================
fprintf('\n========================================\n');
fprintf('  拓展分析: 五种插值 / 超分辨率方法对比\n');
fprintf('========================================\n');

%% 拓展1: 定义方法集合并统一计算
fprintf('\n【拓展】PSNR / SSIM 对比\n');
fprintf('----------------------------------------\n');

% 使用 cell 数组统一管理: {方法名, 上采样函数句柄}
methods = {
    '最近邻',     @(lr) imresize(lr, size(I), 'nearest');
    '双线性',     @(lr) imresize(lr, size(I), 'bilinear');
    '双三次',     @(lr) imresize(lr, size(I), 'bicubic');
    'Lanczos3',   @(lr) imresize(lr, size(I), 'lanczos3');
    'IBP超分辨率', @(lr) ibp_upsample(lr, size(I), 'lanczos3')
};
nM = size(methods, 1);

% 批量计算: 上采样 → PSNR/SSIM → 差图像
upsampled = cell(1, nM);
psnr_vals = zeros(1, nM);
ssim_vals = zeros(1, nM);
diffs     = cell(1, nM);

for i = 1:nM
    upsampled{i} = methods{i,2}(lena4s);
    psnr_vals(i) = psnr(upsampled{i}, I);
    ssim_vals(i) = ssim(upsampled{i}, I);
    diffs{i}     = abs(upsampled{i} - I);
end

% 汇总表格 (MATLAB table 类型, 较 fprintf 更直观)
results = table(methods(:,1), psnr_vals', ssim_vals', ...
    'VariableNames', {'方法', 'PSNR_dB', 'SSIM'});
disp(results);

fprintf('IBP超分辨率 较 Lanczos3 提升: +%.4f dB\n', ...
    psnr_vals(5) - psnr_vals(4));

%% 拓展2: 五种方法效果并列对比
figure('Name', '拓展: 五种方法效果对比', 'NumberTitle', 'off', ...
    'Position', [100, 100, 2100, 350]);

for i = 1:nM
    subplot(1, nM, i);
    imshow(upsampled{i}, []);
    title({methods{i,1}; sprintf('PSNR = %.2f, SSIM = %.4f', ...
        psnr_vals(i), ssim_vals(i))});
end
sgtitle('五种插值 / 超分辨率方法效果对比');

% 保存图片
saveas(gcf, fullfile(fig_dir, 'exp2_ext_methods_compare.png'));
fprintf('图片已保存: exp2_ext_methods_compare.png\n');

%% 拓展3: 差图像对比 (2行 x 5列)
figure('Name', '拓展: 五种差图像对比', 'NumberTitle', 'off', ...
    'Position', [100, 100, 2100, 500]);

for i = 1:nM
    % 第一行: 重建图像
    subplot(2, nM, i);
    imshow(upsampled{i}, []);
    title(methods{i,1});

    % 第二行: 差图像 (stretchlim 增强对比度)
    subplot(2, nM, i + nM);
    diff_show = imadjust(diffs{i}, stretchlim(diffs{i}), []);
    imshow(diff_show, []);
    title({sprintf('差图 均值 = %.4f', mean2(diffs{i})); ...
           sprintf('标准差 = %.4f', std2(diffs{i}(:)))});
    colorbar;
end
sgtitle('五种方法差图像对比 (对比度已增强)');

% 保存图片
saveas(gcf, fullfile(fig_dir, 'exp2_ext_diff_compare.png'));
fprintf('图片已保存: exp2_ext_diff_compare.png\n');

%% 拓展4: 差图像列均值曲线
figure('Name', '拓展: 差图像列均值曲线', 'NumberTitle', 'off', ...
    'Position', [100, 100, 900, 500]);

hold on;
colors    = {'r-', 'g-', 'b-', 'c-', 'm-'};
lw        = [1, 1, 1, 1, 1.5];   % IBP 曲线略粗以突出
for i = 1:nM
    plot(mean(diffs{i}, 1), colors{i}, 'LineWidth', lw(i));
end
hold off;
xlabel('列号'); ylabel('平均绝对误差');
title('五种差图像列均值曲线对比');
legend(methods(:,1), 'Location', 'northwest');
grid on;

% 保存图片
saveas(gcf, fullfile(fig_dir, 'exp2_ext_curve.png'));
fprintf('图片已保存: exp2_ext_curve.png\n');

%% 拓展5: 质量指标排序
fprintf('\n【拓展】按 PSNR 降序排列:\n');
[~, order] = sort(psnr_vals, 'descend');
for i = 1:nM
    fprintf('  %d. %s: PSNR = %.4f dB,  SSIM = %.4f\n', ...
        i, methods{order(i),1}, psnr_vals(order(i)), ssim_vals(order(i)));
end

%% 完成输出
fprintf('\n========================================\n');
fprintf('        实验2 完成!\n');
fprintf('========================================\n');

close all;
