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

%% 实验8: 图像压缩与数字水印
% 课程: 视觉与数据计算
% 重点函数: im2jpeg(jpeg2im), im2jpeg2k(jpeg2k2im), VideoReader, montage
% 参考教材: Digital Image Processing Using MATLAB (Gonzalez, Woods, Eddins)
%
% 【课堂演示说明】
%   运行前设置 DEMO_MODE = true 可进入步进演示模式,
%   每张关键图表显示后暂停, 按 Enter 继续.
%   设置 DEMO_MODE = false 则全自动运行（与原版一致）.
%
% 【文件结构】
%   配置 → 任务1~4（核心实验）→ 拓展1~2（自主拓展）→ 辅助函数
%   每个 %% 为 MATLAB Cell, 可 Ctrl+Enter 单独执行.
%
%   提取 calc_quality_metrics / show_diff_map / show_comparison_12 /
%   show_rd_curve / pause_step 五个辅助函数消除重复代码,
%   增加 DEMO_MODE 步进演示支持.
clear all; close all; clc;

%% ========== 配置与初始化 ==========
% --- 步进演示开关 ---
% true  → 每张图表暂停, 按 Enter 继续, 适合课堂讲解
% false → 全自动运行, 一次性显示所有结果
DEMO_MODE = false;
% 创建图片保存目录
fig_dir = fullfile(fileparts(mfilename('fullpath')), 'figs');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
% 添加 DIPUM 工具箱路径（提供 im2jpeg, jpeg2im, im2jpeg2k, jpeg2k2im 等）
addpath('../dipum_toolbox_2.0.2');
fprintf('========================================\n');
fprintf('      实验8: 图像压缩与数字水印\n');
fprintf('========================================\n\n');
fprintf('演示模式: %s\n\n', conditional(DEMO_MODE, '步进 (按 Enter 继续)', '全自动'));

%% ==================================================
%  任务1: JPEG 压缩
%  核心要求: 读入 lena, 1×2 窗口显示原始和重建图像
%  拓展: 多质量因子分析、率失真曲线、差异图

%% ==================================================
fprintf('【任务1】JPEG压缩 (DIPUM 工具箱)\n');
fprintf('----------------------------------------\n');
% 读入 lena 图像
img_path = 'lena.png';
img = imread(img_path);
if size(img, 3) == 3
    img_gray = rgb2gray(img);
else
    img_gray = img;
end
fprintf('原图像大小: %d x %d\n', size(img_gray, 1), size(img_gray, 2));
img_uint8 = im2uint8(img_gray);

%% 1.0 测试默认质量参数（验证 DIPUM im2jpeg 基本行为）
fprintf('  测试 im2jpeg 默认行为:\n');
c_def = im2jpeg(img_uint8, 1.0);
img_def = jpeg2im(c_def);
psnr_def = psnr(img_def, img_uint8);
fprintf('    quality=1.0: PSNR=%.4f dB, 压缩比=%.1f:1\n', psnr_def, imratio(img_uint8, c_def));
c_def = im2jpeg(img_uint8, 0.1);
img_def = jpeg2im(c_def);
psnr_def = psnr(img_def, img_uint8);
fprintf('    quality=0.1: PSNR=%.4f dB, 压缩比=%.1f:1\n', psnr_def, imratio(img_uint8, c_def));
% 标准 imwrite 参照
fprintf('  参考: 标准 imwrite JPEG:\n');
imwrite(img_uint8, 'temp_ref.jpg', 'Quality', 100);
ref = imread('temp_ref.jpg');
fprintf('    Quality=100: PSNR=%.4f dB, 文件大小=%d KB\n', ...
        psnr(ref, img_uint8), round(dir('temp_ref.jpg').bytes/1024));
imwrite(img_uint8, 'temp_ref.jpg', 'Quality', 10);
ref = imread('temp_ref.jpg');
fprintf('    Quality=10:  PSNR=%.4f dB, 文件大小=%d KB\n', ...
        psnr(ref, img_uint8), round(dir('temp_ref.jpg').bytes/1024));
delete('temp_ref.jpg');
fprintf('\n');
pause_step(DEMO_MODE, 'JPEG 默认质量测试完成');

%% 1.1 多质量因子 JPEG 压缩与重建
quality_factors = [5, 10, 20, 30, 50, 70, 90];
n_q = length(quality_factors);
psnr_vals   = zeros(1, n_q);
ssim_vals   = zeros(1, n_q);
cr_vals     = zeros(1, n_q);
entropy_vals = zeros(1, n_q);
best_psnr_jpeg = -inf;
best_recon_jpeg = [];
best_q_jpeg = [];
best_cr_jpeg = [];
figure('Name', '任务1: JPEG压缩 (DIPUM)', 'NumberTitle', 'off', ...
       'Position', [50, 50, 1600, 900]);
subplot(3, n_q+1, 1);
imshow(img_gray);
title('原图像', 'FontSize', 10);
for i = 1:n_q
    q = quality_factors(i);
    % DIPUM 压缩与重建
    c = im2jpeg(img_uint8, q / 100);
    img_recon = jpeg2im(c);
    % 计算质量指标（函数提取）
    [psnr_val, ssim_val, cr, ent_val] = calc_quality_metrics(img_uint8, c, img_recon);
    psnr_vals(i)   = psnr_val;
    ssim_vals(i)   = ssim_val;
    cr_vals(i)     = cr;
    entropy_vals(i) = ent_val;
    if psnr_val > best_psnr_jpeg
        best_psnr_jpeg = psnr_val;
        best_recon_jpeg = img_recon;
        best_q_jpeg = q;
        best_cr_jpeg = cr;
        best_ssim_jpeg = ssim_val;
    end
    fprintf('质量因子 %2d: 压缩比=%.1f:1, PSNR=%.4f dB, SSIM=%.4f, 熵=%.2f bits/pixel\n', ...
            q, cr, psnr_val, ssim_val, ent_val);
    % 重建图像
    subplot(3, n_q+1, i+1);
    imshow(img_recon);
    title({sprintf('JPEG Q=%d', q), ...
           sprintf('CR=%.1f:1, PSNR=%.4f dB', cr, psnr_val)}, 'FontSize', 9);
    % 差异图（函数提取）
    subplot(3, n_q+1, (n_q+1) + i+1);
    show_diff_map(img_uint8, img_recon, gca, 5, sprintf('差异图 (×5) Q=%d', q));
    % 局部放大（注: ROI 因图而异, 保持内联）
    subplot(3, n_q+1, 2*(n_q+1) + i+1);
    roi_y = 200:399; roi_x = 200:399;
    imshow(img_recon(roi_y, roi_x));
    title(sprintf('局部放大 Q=%d', q), 'FontSize', 9);
end
subplot(3, n_q+1, n_q+2);
imshow(img_gray);
title(sprintf('原图熵=%.2f', ntrop(img_uint8)), 'FontSize', 10);
subplot(3, n_q+1, 2*(n_q+1)+1);
imshow(img_gray(roi_y, roi_x));
title('原图局部', 'FontSize', 10);
pause_step(DEMO_MODE, 'JPEG 多质量分析完成');

%% 1.2 率失真曲线（函数提取）
ann_labels = arrayfun(@(x) sprintf('Q=%d', x), quality_factors, 'UniformOutput', false);
show_rd_curve(cr_vals, psnr_vals, ssim_vals, entropy_vals, ann_labels, ...
              '率失真曲线 (Rate-Distortion)', 'JPEG');
pause_step(DEMO_MODE, 'JPEG 率失真曲线完成');

%% 1.3 核心实验: 原始 vs 重建 1×2 对比窗口
% 满足 full.md 要求: "在一个窗口(1行2列)中显示原始和重建图像"
show_comparison_12(img_gray, best_recon_jpeg, ...
                   '原始图像 (lena)', ...
                   {sprintf('重建图像 Q=%d', best_q_jpeg); ...
                    sprintf('CR=%.1f:1, PSNR=%.4f dB, SSIM=%.4f', ...
                    best_cr_jpeg, best_psnr_jpeg, best_ssim_jpeg)}, ...
                   '任务1: JPEG压缩对比 (最优质量)');
fprintf('\nJPEG压缩分析:\n');
fprintf('  质量因子越低，压缩比越高，但图像质量下降\n');
fprintf('  压缩后熵值降低，说明空间冗余被有效去除\n');
fprintf('  差异图中亮度区域反映压缩损失的空间分布\n');
fprintf('  注: DIPUM im2jpeg 在 quality<0.3(即质量因子<30)时存在异常:\n');
fprintf('      Q=5~20 的 PSNR 仅 12~15dB, SSIM 0.19~0.53, 压缩比也异常偏低(1.9~4.1:1),\n');
fprintf('      这与标准 JPEG 的低质量高压缩比特性相悖(标准imwrite在Q=10时PSNR=30.41dB),\n');
fprintf('      这是教学简化实现 im2jpeg 在边界参数下的编码缺陷.\n');
pause_step(DEMO_MODE, 'JPEG 核心实验完成 → 进入任务2');

%% ==================================================
%  任务2: JPEG2000 压缩
%  核心要求: 读入 house, 1×2 窗口显示原始和重建图像
%  拓展: 多参数分析、率失真曲线、JPEG vs JPEG2000 对比

%% ==================================================
fprintf('\n【任务2】JPEG2000压缩 (DIPUM 工具箱)\n');
fprintf('----------------------------------------\n');
house_path = 'house.png';
house = imread(house_path);
if size(house, 3) == 3
    house_gray = rgb2gray(house);
else
    house_gray = house;
end
fprintf('原图像大小: %d x %d\n', size(house_gray, 1), size(house_gray, 2));
house_uint8 = im2uint8(house_gray);

%% 2.1 多压缩比 JPEG2000 压缩与重建
jp2_configs = [
    6,   8.5;
    8,   8.0;
    10,  7.5;
    12,  7.0;
    15,  6.5;
    20,  6.0
];
n_jp2 = size(jp2_configs, 1);
n_scales = 5;
psnr_jp2   = zeros(1, n_jp2);
ssim_jp2   = zeros(1, n_jp2);
cr_jp2     = zeros(1, n_jp2);
entropy_jp2 = zeros(1, n_jp2);
best_psnr_jp2 = -inf;
best_recon_jp2 = [];
best_mu_jp2 = [];
best_cr_jp2 = [];
figure('Name', '任务2: JPEG2000压缩 (DIPUM)', 'NumberTitle', 'off', ...
       'Position', [50, 50, 1600, 900]);
subplot(3, n_jp2+1, 1);
imshow(house_gray);
title('原图像 (house)', 'FontSize', 10);
for i = 1:n_jp2
    mu = jp2_configs(i, 1);
    eps = jp2_configs(i, 2);
    c = im2jpeg2k(house_uint8, n_scales, [mu, eps]);
    img_recon = jpeg2k2im(c);
    % 计算质量指标（函数提取）
    [psnr_val, ssim_val, cr, ent_val] = calc_quality_metrics(house_uint8, c, img_recon);
    psnr_jp2(i)   = psnr_val;
    ssim_jp2(i)   = ssim_val;
    cr_jp2(i)     = cr;
    entropy_jp2(i) = ent_val;
    if psnr_val > best_psnr_jp2
        best_psnr_jp2 = psnr_val;
        best_recon_jp2 = img_recon;
        best_mu_jp2 = mu;
        best_cr_jp2 = cr;
        best_ssim_jp2 = ssim_val;
    end
    fprintf('[mu=%.0f eps=%.1f]: 压缩比=%.1f:1, PSNR=%.4f dB, SSIM=%.4f, 熵=%.2f\n', ...
            mu, eps, cr, psnr_val, ssim_val, ent_val);
    % 重建图像
    subplot(3, n_jp2+1, i+1);
    imshow(img_recon);
    title({sprintf('JP2 μ=%.0f ε=%.1f', mu, eps), ...
           sprintf('CR=%.1f:1, PSNR=%.4f dB', cr, psnr_val)}, 'FontSize', 9);
    % 差异图（函数提取）
    subplot(3, n_jp2+1, (n_jp2+1) + i+1);
    show_diff_map(house_uint8, img_recon, gca, 5, sprintf('差异图 (×5)'));
    % 局部放大
    subplot(3, n_jp2+1, 2*(n_jp2+1) + i+1);
    roi_y = 64:191; roi_x = 64:191;
    imshow(img_recon(roi_y, roi_x));
    title('局部放大', 'FontSize', 9);
end
subplot(3, n_jp2+1, n_jp2+2);
imshow(house_gray);
title(sprintf('原图熵=%.2f', ntrop(house_uint8)), 'FontSize', 10);
subplot(3, n_jp2+1, 2*(n_jp2+1)+1);
imshow(house_gray(roi_y, roi_x));
title('原图局部', 'FontSize', 10);
pause_step(DEMO_MODE, 'JPEG2000 多参数分析完成');

%% 2.2 率失真曲线（函数提取, 不含熵子图）
ann_labels_jp2 = arrayfun(@(x) sprintf('μ=%.0f', x), jp2_configs(:,1), 'UniformOutput', false);
show_rd_curve(cr_jp2, psnr_jp2, ssim_jp2, [], ann_labels_jp2, ...
              'JPEG2000 率失真曲线', 'JPEG2000');
pause_step(DEMO_MODE, 'JPEG2000 率失真曲线完成');

%% 2.3 核心实验: 原始 vs 重建 1×2 对比窗口
% 满足 full.md 要求: "在一个窗口(1行2列)中显示原始和重建图像"
show_comparison_12(house_gray, best_recon_jp2, ...
                   '原始图像 (house)', ...
                   {sprintf('重建图像 μ=%.0f', best_mu_jp2); ...
                    sprintf('CR=%.1f:1, PSNR=%.4f dB, SSIM=%.4f', ...
                    best_cr_jp2, best_psnr_jp2, best_ssim_jp2)}, ...
                   '任务2: JPEG2000压缩对比 (最优质量)');
fprintf('\nJPEG2000压缩分析:\n');
fprintf('  基于小波变换，在高压缩比下仍能保持较好的视觉质量\n');
fprintf('  差异图中振铃效应(ringing)集中在边缘附近\n');
pause_step(DEMO_MODE, 'JPEG2000 核心实验完成');

%% 2.4 JPEG vs JPEG2000 率失真对比（使用同一图像: lena）
fprintf('\n【对比】JPEG vs JPEG2000 率失真对比\n');
fprintf('----------------------------------------\n');
q_list = [10, 20, 30, 50, 70, 90];
jp2_mu_list = [6, 8, 10, 12, 15, 18];
cr_j = zeros(size(q_list));
psnr_j = zeros(size(q_list));
cr_k = zeros(size(jp2_mu_list));
psnr_k = zeros(size(jp2_mu_list));
cr_std = zeros(size(q_list));
psnr_std = zeros(size(q_list));
for i = 1:length(q_list)
    c = im2jpeg(img_uint8, q_list(i)/100);
    img_r = jpeg2im(c);
    cr_j(i) = imratio(img_uint8, c);
    psnr_j(i) = psnr(img_r, img_uint8);
    imwrite(img_uint8, 'temp_rd.jpg', 'Quality', q_list(i));
    img_r = imread('temp_rd.jpg');
    finfo = dir('temp_rd.jpg');
    cr_std(i) = numel(img_uint8) * 8 / (finfo.bytes * 8);
    psnr_std(i) = psnr(img_r, img_uint8);
end
delete('temp_rd.jpg');
for i = 1:length(jp2_mu_list)
    c = im2jpeg2k(img_uint8, 5, [jp2_mu_list(i), 7.5]);
    img_r = jpeg2k2im(c);
    cr_k(i) = imratio(img_uint8, c);
    psnr_k(i) = psnr(img_r, img_uint8);
end
figure('Name', 'JPEG vs JPEG2000 对比', 'NumberTitle', 'off', ...
       'Position', [100, 100, 1000, 400]);
subplot(1, 2, 1);
plot(cr_j, psnr_j, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 6); hold on;
plot(cr_k, psnr_k, 'r-s', 'LineWidth', 1.5, 'MarkerSize', 6);
plot(cr_std, psnr_std, 'k--^', 'LineWidth', 1.5, 'MarkerSize', 6);
xlabel('压缩比', 'FontSize', 11);
ylabel('PSNR (dB)', 'FontSize', 11);
title('JPEG vs JPEG2000 率失真曲线 (lena)', 'FontSize', 12);
legend({'DIPUM JPEG', 'JPEG2000', '标准JPEG (imwrite)'}, 'Location', 'best');
grid on; hold off;
subplot(1, 2, 2);
plot(cr_j, psnr_j, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 6); hold on;
plot(cr_k, psnr_k, 'r-s', 'LineWidth', 1.5, 'MarkerSize', 6);
xlabel('压缩比', 'FontSize', 11);
ylabel('PSNR (dB)', 'FontSize', 11);
title('DIPUM: JPEG vs JPEG2000 (放大)', 'FontSize', 12);
legend({'DIPUM JPEG', 'JPEG2000'}, 'Location', 'best');
grid on; hold off;
fprintf('分析: DIPUM JPEG 是教学简化实现, PSNR 低于标准 JPEG\n');
fprintf('  JPEG2000 在相同压缩比下 PSNR 更高, 尤其高压缩比时优势明显\n');
pause_step(DEMO_MODE, 'JPEG vs JPEG2000 对比完成 → 进入任务3');

%% ==================================================
%  任务3: 视频帧提取
%  核心要求: 读取 gsalesman, montage(2×2) 显示第(20-22,35)帧
%  注: 本文无重复代码, 保持原样

%% ==================================================
fprintf('\n【任务3】视频帧提取\n');
fprintf('----------------------------------------\n');
video_path = 'gsalesman.avi';
v = VideoReader(video_path);
frame_nums = [20, 21, 22, 35];
frames = cell(length(frame_nums), 1);
for i = 1:length(frame_nums)
    frames{i} = read(v, frame_nums(i));
end
fprintf('视频信息:\n');
fprintf('  文件名: %s\n', video_path);
fprintf('  分辨率: %d x %d\n', v.Width, v.Height);
fprintf('  帧率: %.2f fps\n', v.FrameRate);
fprintf('  总帧数: %d\n', floor(v.Duration * v.FrameRate));
figure('Name', '任务3: 视频帧显示', 'NumberTitle', 'off', ...
       'Position', [50, 50, 1200, 600]);
frame_array = zeros(v.Height, v.Width, 1, length(frame_nums), 'uint8');
for i = 1:length(frame_nums)
    if size(frames{i}, 3) == 3
        frame_gray = rgb2gray(frames{i});
    else
        frame_gray = frames{i};
    end
    frame_array(:,:,1,i) = im2uint8(frame_gray);
end
montage(frame_array, 'Size', [2 2], 'DisplayRange', []);
title(sprintf('视频帧: %d, %d, %d, %d', frame_nums(1), frame_nums(2), ...
      frame_nums(3), frame_nums(4)));
pause_step(DEMO_MODE, '视频帧提取完成 → 进入任务4');

%% ==================================================
%  任务4: DCT 域不可见鲁棒水印 (Example 8.30)
%  核心要求: 读入 lena, 1×2 显示原始和水印图像, 尝试水印提取
%  拓展: DCT系数分析、检测窗口、鲁棒性测试
%  算法来源: Digital Image Processing Using MATLAB, 2/e
%            Gonzalez, Woods, Eddins — Example 8.30
%
%  嵌入: 全局 DCT → 选取 K 个最大系数 → 乘性嵌入高斯序列
%  检测: 皮尔逊相关系数阈值判决

%% ==================================================
fprintf('\n【任务4】DCT域不可见鲁棒水印 (Example 8.30)\n');
fprintf('----------------------------------------\n');

%% 4.1 水印嵌入
img = imread(img_path);
if size(img, 3) == 3
    img_gray_wm = rgb2gray(img);
else
    img_gray_wm = img;
end
img_wm_orig = im2double(img_gray_wm);
alpha = 0.1;        % 水印强度
% 读取水印图像 (例 8.30 不可见水印, 嵌入内容改为实际图像)
[wm_idx, wm_map] = imread('watermark.bmp');
wm_gray = im2double(ind2gray(wm_idx, wm_map)); % 转为灰度 double [0,1]
[wm_h, wm_w] = size(wm_gray);         % 保存尺寸供后续重构显示
omega = wm_gray(:);                    % 展平为 N×1 水印序列
K = numel(omega);                     % 自动适配 DCT 系数个数
fprintf('  水印图像: %d×%d, 总计 %d 像素\n', wm_w, wm_h, K);
C = dct2(img_wm_orig);
% 按幅值选取 K 个最大系数
[C_sorted, idx_sorted] = sort(abs(C(:)), 'descend');
positions = idx_sorted(1:K);
c_i = C(positions);
% 乘性嵌入: c'_i = c_i * (1 + alpha * omega_i)
c_i_modified = c_i .* (1 + alpha .* omega);
C_wm = C;
C_wm(positions) = c_i_modified;
img_watermarked = idct2(C_wm);
img_watermarked = max(0, min(1, img_watermarked));

%% 4.2 水印检测（无攻击）
C_test = dct2(img_watermarked);
c_hat = C_test(positions);
omega_hat = real((c_hat - c_i) ./ (alpha .* c_i));
omega = double(omega);
omega_hat = double(omega_hat);
rho = corrcoef(omega, omega_hat);
gamma = rho(1, 2);
fprintf('水印嵌入完成:\n');
fprintf('  K=%d, alpha=%.2f\n', K, alpha);
fprintf('  含水印图像与原始水印相关系数 γ = %.4f\n', gamma);
% 水印图像重构显示
omega_hat_img = reshape(omega_hat, wm_h, wm_w);
omega_img = reshape(omega, wm_h, wm_w);
figure('Name', '任务4: 水印图像提取结果', 'NumberTitle', 'off', ...
       'Position', [200, 200, 800, 350]);
subplot(1, 2, 1);
imshow(omega_img, []);
title('原始水印图像 (watermark.bmp)', 'FontSize', 12);
subplot(1, 2, 2);
imshow(omega_hat_img, []);
title({'提取水印图像'; sprintf('γ=%.4f, PSNR=%.4f dB, SSIM=%.4f', ...
       gamma, psnr(omega_hat_img, omega_img), ssim(omega_hat_img, omega_img))}, ...
       'FontSize', 11);
% 误检对照: 随机水印应与提取序列不相关
rng(999);
omega_wrong = randn(K, 1);
rho_wrong = corrcoef(omega_wrong, omega_hat);
gamma_wrong = rho_wrong(1, 2);
fprintf('  随机序列与提取序列的相关系数 γ_wrong = %.4f\n', gamma_wrong);
psnr_wm = psnr(img_watermarked, img_wm_orig);
ssim_wm = ssim(img_watermarked, img_wm_orig);
fprintf('  水印图像 PSNR = %.4f dB, SSIM = %.4f\n', psnr_wm, ssim_wm);
pause_step(DEMO_MODE, '水印嵌入与检测完成');

%% 4.3 核心实验: 原始 vs 水印图像 1×2 对比窗口
% 满足 full.md 要求: "在一个窗口(1行2列)中显示原始和水印图像"
show_comparison_12(img_wm_orig, img_watermarked, ...
                   '原图像', ...
                   {sprintf('水印图像'); sprintf('PSNR=%.4f dB, SSIM=%.4f', psnr_wm, ssim_wm)}, ...
                   '任务4: 原始图像 vs 水印图像');
pause_step(DEMO_MODE, '水印核心实验完成');

%% 4.4 DCT 系数与水印分析
figure('Name', '任务4: DCT系数分析', 'NumberTitle', 'off', ...
       'Position', [50, 100, 1200, 350]);
subplot(1, 3, 1);
imshow(abs(C_wm) > 0.01);
title(sprintf('修改的DCT系数位置 (K=%d)', K), 'FontSize', 11);
subplot(1, 3, 2);
bar(1:K, abs(c_i_modified - c_i) ./ abs(c_i));
title('DCT系数相对修改幅度', 'FontSize', 11);
xlabel('系数索引', 'FontSize', 10); ylabel('|Δc|/|c|', 'FontSize', 10);
subplot(1, 3, 3);
C_shift = fftshift(abs(C));
C_shift = log(C_shift + 1);
imshow(C_shift, []);
title('DCT频谱 (对数尺度)', 'FontSize', 11);

%% 4.5 水印提取与检测可视化
figure('Name', '任务4: 水印提取与检测', 'NumberTitle', 'off', ...
       'Position', [100, 150, 1000, 800]);
subplot(2, 2, 1);
imshow(img_watermarked);
title({sprintf('含水印图像'); sprintf('PSNR=%.4f dB, SSIM=%.4f', psnr_wm, ssim_wm)}, ...
      'FontSize', 11);
subplot(2, 2, 2);
imshow(omega_img, []);
title({'原始水印图像 (watermark.bmp)'; sprintf('%d×%d, γ=%.4f', wm_w, wm_h, gamma)}, ...
      'FontSize', 11);
subplot(2, 2, 3);
imshow(omega_hat_img, []);
wm_psnr_extract = psnr(omega_hat_img, omega_img);
wm_ssim_extract = ssim(omega_hat_img, omega_img);
title({'提取水印图像'; sprintf('γ=%.4f, PSNR=%.4f dB, SSIM=%.4f', ...
       gamma, wm_psnr_extract, wm_ssim_extract)}, 'FontSize', 11);
if gamma > 0.5
    verdict = '✓ 水印存在';
else
    verdict = '✗ 水印未检出';
end
text(0.5, -0.08, {sprintf('相关系数 γ = %.4f (阈值: 0.5)', gamma); ...
                  sprintf('误检对照 γ_{wrong} = %.4f', gamma_wrong); verdict}, ...
     'Units', 'normalized', 'FontSize', 10, 'Color', 'k', ...
     'HorizontalAlignment', 'center');
subplot(2, 2, 4);
show_diff_map(img_wm_orig, img_watermarked, gca, 20, '差异图像 (×20)');

%% 4.6 水印鲁棒性测试
fprintf('\n水印鲁棒性测试:\n');
% 攻击1: JPEG 压缩
img_attacked = im2uint8(img_watermarked);
imwrite(img_attacked, 'temp_attack.jpg', 'Quality', 50);
img_attacked = im2double(imread('temp_attack.jpg'));
delete('temp_attack.jpg');
[gamma_jpeg, omega_hat_jpeg] = detect_watermark(img_attacked, positions, c_i, alpha, omega);
omega_hat_jpeg_img = reshape(omega_hat_jpeg, wm_h, wm_w);
% 攻击2: 高斯噪声
img_attacked = img_watermarked + 0.02 * randn(size(img_watermarked));
img_attacked = max(0, min(1, img_attacked));
[gamma_noise, omega_hat_noise] = detect_watermark(img_attacked, positions, c_i, alpha, omega);
omega_hat_noise_img = reshape(omega_hat_noise, wm_h, wm_w);
% 攻击3: 裁剪（右下角 25%）
img_attacked = img_watermarked;
half_h = round(size(img_attacked, 1) / 2);
half_w = round(size(img_attacked, 2) / 2);
img_attacked(half_h:end, half_w:end) = 0.5;
[gamma_crop, omega_hat_crop] = detect_watermark(img_attacked, positions, c_i, alpha, omega);
omega_hat_crop_img = reshape(omega_hat_crop, wm_h, wm_w);
% 攻击4: 缩放攻击（缩小→放大）
img_attacked = imresize(imresize(img_watermarked, 0.5), 2);
if size(img_attacked, 1) ~= size(img_watermarked, 1)
    img_attacked = imresize(img_attacked, [size(img_watermarked,1), size(img_watermarked,2)]);
end
[gamma_scale, omega_hat_scale] = detect_watermark(img_attacked, positions, c_i, alpha, omega);
omega_hat_scale_img = reshape(omega_hat_scale, wm_h, wm_w);
fprintf('  JPEG压缩攻击(Q=50):    γ = %.4f\n', gamma_jpeg);
fprintf('  高斯噪声攻击(σ=0.02):  γ = %.4f\n', gamma_noise);
fprintf('  裁剪攻击(25%%):        γ = %.4f\n', gamma_crop);
fprintf('  缩放攻击(0.5x→2x):    γ = %.4f\n', gamma_scale);
% 鲁棒性柱状图 + 提取水印对比
figure('Name', '水印鲁棒性测试', 'NumberTitle', 'off', ...
       'Position', [50, 50, 1200, 900]);
% 第一行: 柱状图
subplot(2, 3, 1);
attacks = {'无攻击', 'JPEG压缩', '高斯噪声', '裁剪25%', '缩放攻击'};
gamma_vals = [gamma, gamma_jpeg, gamma_noise, gamma_crop, gamma_scale];
bar(gamma_vals, 0.5);
set(gca, 'XTickLabel', attacks, 'XTickLabelRotation', 30);
ylabel('相关系数 γ', 'FontSize', 12);
title('水印鲁棒性测试', 'FontSize', 12);
ylim([0, 1.1]);
grid on;
for i = 1:length(gamma_vals)
    text(i, gamma_vals(i)+0.03, sprintf('%.4f', gamma_vals(i)), ...
         'HorizontalAlignment', 'center', 'FontSize', 10);
end
% 第2~6子图: 原始水印 + 各攻击后提取的水印
subplot(2, 3, 2);
imshow(omega_img, []);
title('原始水印', 'FontSize', 11);
subplot(2, 3, 3);
imshow(omega_hat_jpeg_img, []);
title({sprintf('JPEG压缩 γ=%.4f', gamma_jpeg); ...
       sprintf('PSNR=%.2f', psnr(omega_hat_jpeg_img, omega_img))}, 'FontSize', 10);
subplot(2, 3, 4);
imshow(omega_hat_noise_img, []);
title({sprintf('高斯噪声 γ=%.4f', gamma_noise); ...
       sprintf('PSNR=%.2f', psnr(omega_hat_noise_img, omega_img))}, 'FontSize', 10);
subplot(2, 3, 5);
imshow(omega_hat_crop_img, []);
title({sprintf('裁剪25%% γ=%.4f', gamma_crop); ...
       sprintf('PSNR=%.2f', psnr(omega_hat_crop_img, omega_img))}, 'FontSize', 10);
subplot(2, 3, 6);
imshow(omega_hat_scale_img, []);
title({sprintf('缩放攻击 γ=%.4f', gamma_scale); ...
       sprintf('PSNR=%.2f', psnr(omega_hat_scale_img, omega_img))}, 'FontSize', 10);
sgtitle('水印鲁棒性测试: 攻击后提取水印对比', 'FontSize', 13);
fprintf('\nDCT水印分析:\n');
fprintf('  基于全局DCT最大系数的乘性嵌入，具有良好的不可见性\n');
fprintf('  相关系数检测对常见的图像处理操作具有一定的鲁棒性\n');
fprintf('  水印为实际图像(watermark.bmp, %d×%d), 攻击后仍可辨识内容\n', wm_w, wm_h);
fprintf('  裁剪攻击失效原因: 水印乘性嵌入在全局DCT的前K=%d个最大幅值系数中,\n', K);
fprintf('  这些系数对应图像的低频成分且分布在全图范围。裁剪右下角25%%区域\n');
fprintf('  在空间域直接删除了部分像素信息,经DCT变换后对应的频率分量被破坏,\n');
fprintf('  导致提取的水印序列受损(γ≈%.4f)。这是空间域裁剪攻击对频域水印\n', gamma_crop);
fprintf('  算法的固有局限——水印若嵌入在空间域特定区域则可抗裁剪但不可见性降低.\n');
pause_step(DEMO_MODE, '水印鲁棒性测试完成 → 进入拓展分析');

%% ==================================================
%  ========== 以下为课程拓展分析 ==========

%% ==================================================

%% ==================================================
%  拓展1: DCT 系数能量分布与渐进重建

%% ==================================================
fprintf('\n【拓展1】DCT 能量分布与渐进重建\n');
fprintf('----------------------------------------\n');
block = img_uint8(33:40, 33:40);
block_center = double(block) - 128;
dct_block = dct2(block_center);
figure('Name', '拓展: DCT能量分布', 'NumberTitle', 'off', ...
       'Position', [50, 50, 1400, 500]);
subplot(2, 4, 1);
imshow(block);
title('原始 8×8 块', 'FontSize', 11);
subplot(2, 4, 2);
imshow(abs(dct_block), []);
title('DCT 系数幅值', 'FontSize', 11);
colormap(gca, 'hot');
subplot(2, 4, 3);
imagesc(abs(dct_block)); colorbar;
title('DCT系数 (伪彩色)', 'FontSize', 11);
axis square;
% 累计能量
coeffs = abs(dct_block(:));
coeffs_sorted = sort(coeffs, 'descend');
cum_energy = cumsum(coeffs_sorted.^2) / sum(coeffs_sorted.^2) * 100;
subplot(2, 4, 4);
plot(1:64, cum_energy, 'b-', 'LineWidth', 1.5);
xlabel('系数个数 (按幅值降序)', 'FontSize', 10);
ylabel('累计能量百分比 (%)', 'FontSize', 10);
title('DCT 能量集中度', 'FontSize', 11);
grid on;
xline(10, '--r', '10个系数');
% Zigzag 顺序
Z = [
    1,  2,  6,  7, 15, 16, 28, 29;
    3,  5,  8, 14, 17, 27, 30, 43;
    4,  9, 13, 18, 26, 31, 42, 44;
   10, 12, 19, 25, 32, 41, 45, 54;
   11, 20, 24, 33, 40, 46, 53, 55;
   21, 23, 34, 39, 47, 52, 56, 61;
   22, 35, 38, 48, 51, 57, 60, 62;
   36, 37, 49, 50, 58, 59, 63, 64
];
retain_counts = [1, 6, 20, 64];
% 渐进重建
[p_psnr_recon, p_retain_counts] = ...
    progressive_reconstruct(dct_block, Z, retain_counts, block);
for k = 1:length(retain_counts)
    subplot(2, 4, 4 + k);
    n = retain_counts(k);
    [~, ~, block_recon] = progressive_reconstruct(dct_block, Z, n, block);
    imshow(block_recon);
    title(sprintf('保留 %d 系数', n), 'FontSize', 10);
end
sgtitle('DCT 渐进重建 (Zigzag顺序)', 'FontSize', 12);
% 累计能量 + 重建质量
figure('Name', 'DCT 能量与重建质量', 'NumberTitle', 'off', ...
       'Position', [100, 100, 800, 400]);
subplot(1, 2, 1);
plot(1:64, cum_energy, 'b-', 'LineWidth', 1.5); hold on;
xline(10, '--r', '10个系数');
xlabel('系数个数', 'FontSize', 11);
ylabel('累计能量 (%)', 'FontSize', 11);
title('DCT 系数能量集中度', 'FontSize', 12);
grid on;
for k = 1:length(retain_counts)
    e = cum_energy(min(retain_counts(k), 64));
    plot(retain_counts(k), e, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
    text(retain_counts(k), e+2, sprintf('%d→%.1f%%', retain_counts(k), e), 'FontSize', 9);
end
subplot(1, 2, 2);
plot(p_retain_counts, p_psnr_recon, 'g-s', 'LineWidth', 1.5, 'MarkerSize', 8);
xlabel('保留系数个数', 'FontSize', 11);
ylabel('块重建 PSNR (dB)', 'FontSize', 11);
title('渐进重建质量', 'FontSize', 12);
grid on;
pause_step(DEMO_MODE, '拓展1: DCT能量分析完成');

%% ==================================================
%  拓展2: JPEG 标准量化表可视化

%% ==================================================
fprintf('\n【拓展2】量化表演示\n');
fprintf('----------------------------------------\n');
Q_luminance_50 = [
    16, 11, 10, 16,  24,  40,  51,  61;
    12, 12, 14, 19,  26,  58,  60,  55;
    14, 13, 16, 24,  40,  57,  69,  56;
    14, 17, 22, 29,  51,  87,  80,  62;
    18, 22, 37, 56,  68, 109, 103,  77;
    24, 35, 55, 64,  81, 104, 113,  92;
    49, 64, 78, 87, 103, 121, 120, 101;
    72, 92, 95, 98, 112, 100, 103,  99
];
Q_chrominance_50 = [
    17, 18, 24, 47, 99, 99, 99, 99;
    18, 21, 26, 66, 99, 99, 99, 99;
    24, 26, 56, 99, 99, 99, 99, 99;
    47, 66, 99, 99, 99, 99, 99, 99;
    99, 99, 99, 99, 99, 99, 99, 99;
    99, 99, 99, 99, 99, 99, 99, 99;
    99, 99, 99, 99, 99, 99, 99, 99;
    99, 99, 99, 99, 99, 99, 99, 99
];
qualities_demo = [10, 50, 90];
figure('Name', '拓展: 量化表', 'NumberTitle', 'off', ...
       'Position', [50, 50, 1200, 800]);
for idx = 1:3
    qf = qualities_demo(idx);
    s = max(1, min(255, round(5000 / qf - 50)));
    Q = Q_luminance_50 * s / 50;
    Q = max(1, round(Q));
    subplot(2, 3, idx);
    imagesc(Q); colorbar; axis square;
    title(sprintf('亮度量化表 Q=%d', qf), 'FontSize', 11);
    for r = 1:8
        for c = 1:8
            text(c, r, num2str(Q(r, c)), 'HorizontalAlignment', 'center', ...
                 'FontSize', 7, 'Color', 'w');
        end
    end
    Q_c = Q_chrominance_50 * s / 50;
    Q_c = max(1, round(Q_c));
    subplot(2, 3, 3+idx);
    imagesc(Q_c); colorbar; axis square;
    title(sprintf('色度量表 Q=%d', qf), 'FontSize', 11);
    for r = 1:8
        for c = 1:8
            text(c, r, num2str(Q_c(r, c)), 'HorizontalAlignment', 'center', ...
                 'FontSize', 7, 'Color', 'w');
        end
    end
end
sgtitle('JPEG 标准量化表 (亮度 vs 色度)', 'FontSize', 13);
pause_step(DEMO_MODE, '全部实验完成');

%% 清理临时文件
fclose all;

%% 结果汇总
fprintf('\n========================================\n');
fprintf('           实验结果分析\n');
fprintf('========================================\n');
fprintf('1. JPEG压缩: DCT + 量化 + 熵编码, 质量因子控制压缩率\n');
fprintf('   使用了 DIPUM 的 im2jpeg/jpeg2im, 符合课程要求\n');
fprintf('   注: DIPUM im2jpeg 在 quality<0.3 时存在编码异常(PSNR/CR反常偏低),\n');
fprintf('       与标准 JPEG 对比可验证其边界缺陷.\n');
fprintf('2. JPEG2000: 基于小波变换, 高压缩比下优于 JPEG\n');
fprintf('   使用了 DIPUM 的 im2jpeg2k/jpeg2k2im\n');
fprintf('3. 率失真曲线: 直观展示压缩比与质量指标的权衡关系\n');
fprintf('4. 差异图: 显示压缩损失的空间分布\n');
fprintf('5. DCT域水印: Example 8.30 算法, 全局DCT最大系数乘性嵌入,\n');
fprintf('   相关系数检测, 对常见攻击有一定鲁棒性\n');
fprintf('   裁剪攻击失效原因: 水印分布在全局DCT系数中,空间域裁剪破坏对应频率分量.\n');
fprintf('6. DCT能量分布: 前少数系数集中大部分能量\n');
fprintf('7. 量化表: 亮度vs色度, 不同质量因子下的步长差异\n');
% 保存所有图片
figs = findobj(0, 'Type', 'figure');
for f_i = 1:length(figs)
    saveas(figs(f_i), fullfile(fig_dir, sprintf('exp8_fig%d.png', f_i)));
end

fprintf('\n========================================\n');
fprintf('        实验8 完成!\n');
fprintf('========================================\n');

%% ============================================================
%  辅助函数
%  功能: 减少重复代码, 保持主流程清晰
%  说明: MATLAB 允许脚本末尾定义局部函数 (R2016b+)

%% ============================================================
function [psnr_val, ssim_val, cr, ent_val] = calc_quality_metrics(orig_uint8, c_struct, recon_u)
    % calc_quality_metrics 一次性计算四个质量指标
    %   输入: orig_uint8 = uint8 原图
    %         c_struct   = im2jpeg/im2jpeg2k 返回的压缩结构体
    %         recon_u    = uint8 重建图
    %   输出: psnr_val, ssim_val, cr, ent_val
    cr = imratio(orig_uint8, c_struct);
    psnr_val = psnr(recon_u, orig_uint8);
    ssim_val = ssim(recon_u, orig_uint8);
    ent_val = ntrop(recon_u);
end
function show_diff_map(orig, recon, ax_h, scale, title_str)
    % show_diff_map 绘制差异图（×5 放大 + hot 色表）
    %   输入: orig, recon = 图像矩阵 (uint8 或 double)
    %         ax_h   = axes 句柄 (如 gca)
    %         scale  = 差异放大倍数 (默认 5)
    %         title_str = 标题文本
    diff_img = abs(double(orig) - double(recon));
    axes(ax_h);  % 切换当前坐标轴
    imshow(diff_img * scale, []);
    title({title_str; sprintf('MAD=%.1f', mean(diff_img(:)))}, 'FontSize', 9);
    colormap(gca, 'hot');
end
function show_comparison_12(orig, recon, left_t, right_t, fig_name)
    % show_comparison_12 创建 1×2 标准对比窗口（实验要求格式）
    %   输入: orig, recon = 图像矩阵
    %         left_t, right_t = 标题 (字符串或 cell)
    %         fig_name = 窗口标题
    figure('Name', fig_name, 'NumberTitle', 'off', 'Position', [50, 50, 900, 400]);
    subplot(1, 2, 1);
    imshow(orig);
    title(left_t, 'FontSize', 12);
    subplot(1, 2, 2);
    imshow(recon);
    title(right_t, 'FontSize', 11);
end
function show_rd_curve(cr, psnr, ssim, entropy, labels, fig_name, prefix)
    % show_rd_curve 绘制率失真曲线
    %   输入: cr, psnr, ssim = 等长向量
    %         entropy = [] 时只显示 1×2 子图, 否则 1×3
    %         labels  = cell 数组, 每个数据点的标注文本
    %         fig_name = Figure 窗口标题
    %         prefix   = 子图标题前缀 (如 'JPEG' / 'JPEG2000')
    n_plots = 2 + ~isempty(entropy);
    figure('Name', fig_name, 'NumberTitle', 'off', 'Position', [100, 100, 400*n_plots, 400]);
    subplot(1, n_plots, 1);
    plot(cr, psnr, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 6);
    xlabel('压缩比', 'FontSize', 11); ylabel('PSNR (dB)', 'FontSize', 11);
    title(sprintf('%s 率失真曲线 (CR-PSNR)', prefix), 'FontSize', 11);
    grid on;
    for i = 1:length(cr)
        text(cr(i)+0.5, psnr(i)-0.5, labels{i}, 'FontSize', 8);
    end
    subplot(1, n_plots, 2);
    plot(cr, ssim, 'r-s', 'LineWidth', 1.5, 'MarkerSize', 6);
    xlabel('压缩比', 'FontSize', 11); ylabel('SSIM', 'FontSize', 11);
    title(sprintf('%s SSIM 曲线', prefix), 'FontSize', 11);
    grid on;
    for i = 1:length(cr)
        text(cr(i)+0.5, ssim(i)-0.005, labels{i}, 'FontSize', 8);
    end
    if ~isempty(entropy)
        subplot(1, n_plots, 3);
        plot(cr, entropy, 'g-d', 'LineWidth', 1.5, 'MarkerSize', 6);
        xlabel('压缩比', 'FontSize', 11); ylabel('熵 (bits/pixel)', 'FontSize', 11);
        title(sprintf('%s 压缩后熵值变化', prefix), 'FontSize', 11);
        grid on;
    end
end
function pause_step(enabled, msg)
    % pause_step 步进演示辅助
    %   输入: enabled = true 时暂停等待 Enter
    %                   false 时无条件继续
    %         msg     = 提示文本 (可选, 默认 '继续演示')
    %   在 enabled 状态下同时关闭所有图窗, 减少屏幕杂乱
    if enabled
        if nargin < 2, msg = '继续演示'; end
        fprintf(sprintf('◆ (%s) [auto-continue] ', msg));
        % batch: input(sprintf('\n  ◆ (%s) [Enter] ', msg), 's');
        close all;
        fprintf('\n');
    end
end
function [gamma, omega_hat] = detect_watermark(img_attacked, positions, c_i, alpha, omega)
    % detect_watermark 从攻击后的图像中提取水印并计算相关系数
    %   输入: img_attacked = double 攻击后图像
    %         positions    = DCT 系数选中位置的线性索引
    %         c_i          = 原 DCT 系数值
    %         alpha        = 水印强度
    %         omega        = 原始水印序列
    %   输出: gamma    = 皮尔逊相关系数
    %         omega_hat = 提取的水印序列 (可用于重构水印图像)
    C_test = dct2(img_attacked);
    c_hat = C_test(positions);
    omega_hat = real((c_hat - c_i) ./ (alpha .* c_i));
    rho = corrcoef(double(omega), double(omega_hat));
    gamma = rho(1, 2);
end
function [psnr_vals, retain_counts, block_recon] = progressive_reconstruct(dct_block, Z, n, orig_block)
    % progressive_reconstruct 按 Zigzag 顺序渐进重建 8×8 块
    %   输入: dct_block  = 8×8 DCT 系数
    %         Z          = 8×8 Zigzag 扫描表 (值 1~64)
    %         n          = 保留系数个数 (标量) 或保留数组
    %         orig_block = uint8 原始块 (用于计算 PSNR)
    %   输出: psnr_vals  = 各保留步长的 PSNR 数组
    %         retain_counts = 输入 n (向量化时)
    %         block_recon   = 重建块 (n 为标量时返回单个块)
    %
    %   向量化模式: n 为数组时返回 psnr_vals 和 retain_counts
    %   单值模式:   n 为标量时返回 block_recon
    if isscalar(n)
        dct_masked = zeros(8, 8);
        for idx = 1:n
            [r, c] = find(Z == idx);
            dct_masked(r, c) = dct_block(r, c);
        end
        block_recon = idct2(dct_masked) + 128;
        block_recon = uint8(max(0, min(255, round(block_recon))));
        psnr_vals = psnr(block_recon, orig_block);
        retain_counts = n;
    else
        psnr_vals = zeros(size(n));
        retain_counts = n;
        for k = 1:length(n)
            nk = n(k);
            dct_masked = zeros(8, 8);
            for idx = 1:nk
                [r, c] = find(Z == idx);
                dct_masked(r, c) = dct_block(r, c);
            end
            block_recon = idct2(dct_masked) + 128;
            block_recon = uint8(max(0, min(255, round(block_recon))));
            psnr_vals(k) = psnr(block_recon, orig_block);
        end
        block_recon = [];
    end
end
function s = conditional(cond, tstr, fstr)
    % 内联条件函数 (替代三元运算符)
    if cond
        s = tstr;
    else
        s = fstr;
    end
end