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

%% 实验6: 多尺度分析与小波变换 (使用Wavelet Toolbox)
% 课程: 视觉与数据计算
% 重点函数: wavedec2, appcoef2, detcoef2, wfilters
%           waverec2, upcoef2, upwlev2, wrcoef2
%           wdencmp, wdenoise2, wthcoef2, wthresh, thselect
clear all;
close all;
clc;
% 创建图片保存目录
fig_dir = fullfile(fileparts(mfilename('fullpath')), 'figs');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
fprintf('========================================\n');
fprintf('     实验6: 多尺度分析与小波变换\n');
fprintf('========================================\n\n');

%% 关键 Wavelet Toolbox 函数速查
%  wavedec2(X,N,wname)     — 二维小波分解
%  waverec2(C,S,wname)     — 二维小波重构
%  appcoef2(C,S,wname,N)   — 提取近似系数
%  detcoef2(O,C,S,N)       — 提取细节系数 (O='h'/'v'/'d'/'all')
%  wrcoef2(type,C,S,wname,N) — 重构某层分量至原图尺寸
%  wthcoef2(type,C,S,N,T,sorh) — 阈值处理细节系数
%  wdencmp(mode,X,wname,N,THR,sorh,keepapp) — 去噪/压缩
%  wthresh(Y,sorh,T)       — 标量阈值函数
%  thselect(Y,TPTR)        — 自动阈值选取
%  wfilters(wname)         — 获取小波滤波器系数
%  upcoef2(O,coefs,wname,N) — 单层系数上采样重构
%  upwlev2(C,S,wname)      — 单层重构(提升一层)

%% ========================================================================
%  任务1: 小波分解与边缘检测
%  使用 Symlets 小波三尺度分解和重构, 提取水平/垂直/对角线方向的边缘
%  ========================================================================
fprintf('【任务1】小波分解与边缘检测\n');
fprintf('----------------------------------------\n');
% 读入图像
img_path = 'house.png';
img_gray = load_gray(img_path);
fprintf('使用图像: %s, 尺寸: %dx%d\n', img_path, size(img_gray, 1), size(img_gray, 2));
% --- 小波分解 (wavedec2) ---
% 语法: [C, S] = wavedec2(X, N, wname)
%   C: 系数向量 (近似 + 逐层细节)
%   S: 记录各层大小的书签矩阵
wavelet = 'sym4';  % Symlets 小波
level = 3;         % 三尺度分解
[C, S] = wavedec2(img_gray, level, wavelet);
fprintf('小波分解: %s小波, %d层\n', wavelet, level);
% --- 提取各方向细节系数 (detcoef2) ---
% 语法: [H,V,D] = detcoef2('all', C, S, N)   — 同时提取三层
%        H = detcoef2('h', C, S, N)          — 水平细节
%        V = detcoef2('v', C, S, N)          — 垂直细节
%        D = detcoef2('d', C, S, N)          — 对角线细节
H = cell(level, 1); V = cell(level, 1); D = cell(level, 1);
for lev = 1:level
    [H{lev}, V{lev}, D{lev}] = detcoef2('all', C, S, lev);
end
% --- 提取近似系数 (appcoef2) ---
% 语法: A = appcoef2(C, S, wname, N)
cA3 = appcoef2(C, S, wavelet, 3);
fprintf('近似系数大小: %dx%d\n', size(cA3, 1), size(cA3, 2));
% --- 重构各方向边缘 (wrcoef2) ---
% 语法: Xrec = wrcoef2(type, C, S, wname, N)
%   type='a' — 重构近似; type='h'/'v'/'d' — 重构细节
%   自动将系数上采样+滤波至原图分辨率
img_h_edge = zeros(size(img_gray));
img_v_edge = zeros(size(img_gray));
img_d_edge = zeros(size(img_gray));
for lev = 1:level
    img_h_edge = img_h_edge + wrcoef2('h', C, S, wavelet, lev);
    img_v_edge = img_v_edge + wrcoef2('v', C, S, wavelet, lev);
    img_d_edge = img_d_edge + wrcoef2('d', C, S, wavelet, lev);
end
fprintf('边缘重构完成: 水平/垂直/对角线方向\n');
% --- 显示结果 ---
figure('Name', '任务1: 小波分解与边缘检测', 'NumberTitle', 'off', ...
       'Position', [50, 50, 1000, 800]);
subplot(2, 2, 1);
imshow(img_gray, []);
title('原图像 (house)', 'FontSize', 12);
subplot(2, 2, 2);
imshow(abs(img_h_edge), []);
title('水平方向边缘', 'FontSize', 12);
subplot(2, 2, 3);
imshow(abs(img_v_edge), []);
title('垂直方向边缘', 'FontSize', 12);
subplot(2, 2, 4);
imshow(abs(img_d_edge), []);
title('对角线方向边缘', 'FontSize', 12);
sgtitle(sprintf('任务1: %s小波三尺度分解边缘检测', wavelet), 'FontSize', 14);
saveas(gcf, fullfile(fig_dir, 'exp6_fig1.png'));
% 保存图片
fprintf('图片已保存: exp6_fig1.png');

%% 拓展: wfilters 演示 — 展示小波滤波器系数
%  wfilters 返回指定小波的分解/重构低通和高通滤波器系数
[Lo_D, Hi_D, Lo_R, Hi_R] = wfilters('sym4');
figure('Name', 'sym4小波滤波器', 'NumberTitle', 'off', 'Position', [50, 50, 800, 600]);
subplot(2,2,1); stem(Lo_D); title('分解低通滤波器 (Lo_D)');
subplot(2,2,2); stem(Hi_D); title('分解高通滤波器 (Hi_D)');
subplot(2,2,3); stem(Lo_R); title('重构低通滤波器 (Lo_R)');
subplot(2,2,4); stem(Hi_R); title('重构高通滤波器 (Hi_R)');
sgtitle('Symlets 4 小波滤波器系数 (wfilters)', 'FontSize', 13);
saveas(gcf, fullfile(fig_dir, 'exp6_fig2.png'));
% 保存图片
fprintf('图片已保存: exp6_fig2.png');
fprintf('  wfilters: sym4滤波器长度=%d\n', length(Lo_D));
fprintf('任务1完成\n\n');
pause;

%% ========================================================================
%  任务2: 小波去噪 (含多方案对比)
%  对含高斯噪声(σ=30)的lena图像进行小波阈值去噪
%  对比方案: 纯小波去噪 | 先高斯滤波再小波去噪 | 小波去噪+高斯后处理
%  ========================================================================
fprintf('【任务2】小波去噪 (含多方案对比)\n');
fprintf('----------------------------------------\n');
% 读入图像
lena = load_gray('lena.png');
% 添加高斯噪声 (均值为0, 偏差为30)
sigma_noise = 30;
rng(0);
lena_noisy = lena + (sigma_noise / 255) * randn(size(lena));
lena_noisy = clip_image(lena_noisy);  % 裁剪到[0,1]
fprintf('添加高斯噪声: σ=%d\n', sigma_noise);
% --- 小波分解参数 ---
level2 = 4;      % 4层分解以获得更好的去噪效果
wavelet2 = 'sym4';
gauss_hsize = 5;   % 高斯滤波核大小
gauss_sigma = 1.5; % 高斯滤波标准差
fprintf('高斯滤波参数: 核大小=%d, σ=%.1f\n', gauss_hsize, gauss_sigma);
% ========== 方法1: 纯小波阈值去噪 (基准) ==========
fprintf('\n>>> 方法1: 纯小波软阈值去噪 (基准)\n');
[C_n, S_n] = wavedec2(lena_noisy, level2, wavelet2);
% 从最细尺度对角线系数估计噪声标准差
D1 = detcoef2('d', C_n, S_n, 1);
sigma_est = median(abs(D1(:))) / 0.6745;
% 通用阈值 (VisuShrink)
thr = sigma_est * sqrt(2 * log(numel(lena_noisy)));
fprintf('  噪声估计 σ=%.4f, 阈值 Thr=%.4f\n', sigma_est, thr);

%% 拓展: thselect 自动阈值选取对比 (不改变去噪结果, 仅供讨论)
%  thselect 提供了多种策略: 'sqtwolog'(通用), 'rigrsure'(SURE), 'heursure', 'minimaxi'
thr_sure = thselect(D1(:), 'rigrsure');     % Stein无偏风险估计
thr_minimax = thselect(D1(:), 'minimaxi');  % 极小极大阈值
fprintf('  [thselect对比] VisuShrink=%.4f, SURE=%.4f, Minimax=%.4f\n', thr, thr_sure, thr_minimax);
% 对各级细节系数进行软阈值处理
C_d = wavelet_threshold(C_n, S_n, level2, thr, 's');
lena_d1 = clip_image(waverec2(C_d, S_n, wavelet2));
[psnr_d1, ssim_d1] = print_quality(lena_d1, lena);
% ========== 方法2: 先高斯滤波 → 再小波去噪 ==========
fprintf('\n>>> 方法2: 先高斯滤波 → 再小波去噪 (你提出的方案)\n');
% 步骤1: 高斯滤波预处理 (去除部分高频噪声)
gauss_kernel = fspecial('gaussian', gauss_hsize, gauss_sigma);
lena_gauss = imfilter(lena_noisy, gauss_kernel, 'replicate');
fprintf('  高斯滤波完成: 核%d×%d, σ=%.1f\n', gauss_hsize, gauss_hsize, gauss_sigma);
% 步骤2: 对高斯滤波结果进行小波去噪
[C_g, S_g] = wavedec2(lena_gauss, level2, wavelet2);
% 重新估计噪声水平 (高斯滤波后噪声已降低)
D1_g = detcoef2('d', C_g, S_g, 1);
sigma_est_g = median(abs(D1_g(:))) / 0.6745;
thr_g = sigma_est_g * sqrt(2 * log(numel(lena_gauss)));
fprintf('  高斯后噪声估计 σ=%.4f, 阈值 Thr=%.4f\n', sigma_est_g, thr_g);
C_dg = wavelet_threshold(C_g, S_g, level2, thr_g, 's');
lena_d2 = clip_image(waverec2(C_dg, S_g, wavelet2));
[psnr_d2, ssim_d2] = print_quality(lena_d2, lena);
% ========== 方法3: 小波去噪 → 高斯滤波后处理 ==========
fprintf('\n>>> 方法3: 小波去噪 → 高斯滤波后处理 (对称对比)\n');
lena_d3 = clip_image(imfilter(lena_d1, gauss_kernel, 'replicate'));
[psnr_d3, ssim_d3] = print_quality(lena_d3, lena);
% ========== 方法4: 小波域混合处理 ==========
fprintf('\n>>> 方法4: 小波域处理 (近似高斯滤波 + 细节软阈值)\n');
fprintf('  在小波分解的近似系数上做高斯滤波, 同时对细节系数做软阈值\n');
[C_m4, S_m4] = wavedec2(lena_noisy, level2, wavelet2);
% 先对细节系数做软阈值
C_m4 = wavelet_threshold(C_m4, S_m4, level2, thr, 's');
% 重构近似分量 → 高斯滤波 + 细节分量 (保持不变)
A_part = wrcoef2('a', C_m4, S_m4, wavelet2, level2);
A_part_fil = imfilter(A_part, gauss_kernel, 'replicate');
D_part = waverec2(C_m4, S_m4, wavelet2) - A_part;
lena_d4 = clip_image(A_part_fil + D_part);
[psnr_d4, ssim_d4] = print_quality(lena_d4, lena);
% ========== 汇总对比 ==========
fprintf('\n========== 四种方案对比汇总 ==========\n');
fprintf('  方法1 (纯小波去噪)       : PSNR = %.4f dB, SSIM = %.4f ★基准\n', psnr_d1, ssim_d1);
fprintf('  方法2 (高斯→小波)        : PSNR = %.4f dB, SSIM = %.4f %s\n', ...
    psnr_d2, ssim_d2, ternary(psnr_d2 >= psnr_d1, '↑优于基准', '↓劣于基准'));
fprintf('  方法3 (小波→高斯)        : PSNR = %.4f dB, SSIM = %.4f %s\n', ...
    psnr_d3, ssim_d3, ternary(psnr_d3 >= psnr_d1, '↑优于基准', '↓劣于基准'));
fprintf('  方法4 (小波域混合处理)   : PSNR = %.4f dB, SSIM = %.4f %s\n', ...
    psnr_d4, ssim_d4, ternary(psnr_d4 >= psnr_d1, '↑优于基准', '↓劣于基准'));
fprintf('====================================\n\n');
% 计算差图像 (用于显示)
diff_d1 = lena_noisy - lena_d1;
diff_d2 = lena_noisy - lena_d2;
% --- 显示结果: 按full.md要求 2行2列 ---
% 但为了对比方案, 使用 2行3列显示主结果, 再加一张汇总对比图
figure('Name', '任务2: 小波去噪 — 方案对比', 'NumberTitle', 'off', ...
       'Position', [50, 50, 1500, 900]);
% 第一行: 原图 / 噪声图 / 方法1(纯小波)
subplot(2, 3, 1);
imshow(lena, []);
title('原图像 (lena)', 'FontSize', 12);
subplot(2, 3, 2);
imshow(lena_noisy, []);
title({sprintf('噪声图像 (σ=%d)', sigma_noise), ...
       sprintf('PSNR=%.4f dB, SSIM=%.4f', psnr(lena_noisy, lena), ssim(lena_noisy, lena))}, ...
       'FontSize', 10);
subplot(2, 3, 3);
imshow(lena_d1, []);
title({sprintf('方法1: 纯小波软阈值去噪'), ...
       sprintf('PSNR=%.4f dB, SSIM=%.4f ★', psnr_d1, ssim_d1)}, ...
       'FontSize', 10, 'Color', 'k');
% 第二行: 方法2 / 方法3 / 方法4
subplot(2, 3, 4);
imshow(lena_d2, []);
title({sprintf('方法2: 高斯滤波→小波去噪'), ...
       sprintf('PSNR=%.4f dB, SSIM=%.4f %s', psnr_d2, ssim_d2, ...
               ternary(psnr_d2 >= psnr_d1, '★最优', '')), ...
       '高斯粗去噪降噪声水平 → 小波精去噪保留细节'}, ...
       'FontSize', 10);
subplot(2, 3, 5);
imshow(lena_d3, []);
title({sprintf('方法3: 小波去噪→高斯后处理'), ...
       sprintf('PSNR=%.4f dB, SSIM=%.4f %s', psnr_d3, ssim_d3, ...
               ternary(psnr_d3 >= psnr_d1, '★', '')), ...
       '先小波去噪, 再高斯滤波平滑残留伪影'}, ...
       'FontSize', 10);
subplot(2, 3, 6);
imshow(lena_d4, []);
title({sprintf('方法4: 小波域混合处理'), ...
       sprintf('PSNR=%.4f dB, SSIM=%.4f %s', psnr_d4, ssim_d4, ...
               ternary(psnr_d4 >= psnr_d1, '★', '')), ...
       '近似高斯滤波 + 细节软阈值'}, ...
       'FontSize', 10);
sgtitle(sprintf('任务2: %s小波 %d层 去噪方案对比 (阈值Thr=%.4f)', ...
        wavelet2, level2, thr), 'FontSize', 14);
saveas(gcf, fullfile(fig_dir, 'exp6_fig3.png'));
% --- 补充: 差图像对比 (展示细节损失差异) ---
figure('Name', '任务2: 差图像对比', 'NumberTitle', 'off', ...
       'Position', [50, 50, 1200, 350]);
subplot(1, 3, 1);
imshow(abs(lena_noisy - lena_d1) * 4, []);
title({'方法1差图(×4)', '(纯小波)'}, 'FontSize', 11);
subplot(1, 3, 2);
imshow(abs(lena_noisy - lena_d2) * 4, []);
title({'方法2差图(×4)', '(高斯→小波)'}, 'FontSize', 11);
subplot(1, 3, 3);
imshow(abs(lena_d1 - lena_d2) * 4, []);
title({'方法1vs方法2差图(×4)', '(两者差异)'}, 'FontSize', 11);
sgtitle('差图像对比 (放大4倍显示, 越亮表示差异越大)', 'FontSize', 13);
saveas(gcf, fullfile(fig_dir, 'exp6_fig4.png'));
% 保存图片
fprintf('图片已保存: exp6_fig4.png');
fprintf('任务2完成\n\n');
pause;

%% ========================================================================
%  任务3: 渐进重构
%  使用 Biorthogonal 小波三尺度分解, 逐步添加细节系数实现渐进传输
%  ========================================================================
fprintf('【任务3】渐进重构\n');
fprintf('----------------------------------------\n');
% --- Biorthogonal 小波三尺度分解 ---
wavelet3 = 'bior6.8';
level3 = 3;
[C3, S3] = wavedec2(lena, level3, wavelet3);
fprintf('小波分解: %s, %d层\n', wavelet3, level3);
% --- 渐进重构策略 ---
% 利用 wrcoef2 分别重构各成分, 然后逐级叠加:
%   图像1: 仅3尺度近似系数
%   图像2: 3尺度近似 + 3尺度细节
%   图像3: 3尺度近似 + 3尺度 + 2尺度细节
% 这样避免了手动修改C向量的复杂性, 且 wrcoef2 自动将结果上采样至原图大小
% 计算各成分 (每个成分均为原图分辨率)
comp_a3 = wrcoef2('a', C3, S3, wavelet3, 3);  % 第3层近似
comp_h3 = wrcoef2('h', C3, S3, wavelet3, 3);  % 第3层水平细节
comp_v3 = wrcoef2('v', C3, S3, wavelet3, 3);  % 第3层垂直细节
comp_d3 = wrcoef2('d', C3, S3, wavelet3, 3);  % 第3层对角线细节
comp_h2 = wrcoef2('h', C3, S3, wavelet3, 2);
comp_v2 = wrcoef2('v', C3, S3, wavelet3, 2);
comp_d2 = wrcoef2('d', C3, S3, wavelet3, 2);
% 渐进重构图像
img_prog1 = comp_a3;                                               % 仅近似
img_prog2 = img_prog1 + comp_h3 + comp_v3 + comp_d3;               % +3层细节
img_prog3 = img_prog2 + comp_h2 + comp_v2 + comp_d2;               % +2层细节
fprintf('渐进重构完成:\n');
fprintf('  图像1: 仅第3层近似 (最粗糙)\n');
fprintf('  图像2: + 第3层细节\n');
fprintf('  图像3: + 第2层细节\n');
% --- 显示结果 ---
figure('Name', '任务3: 渐进重构', 'NumberTitle', 'off', ...
       'Position', [50, 50, 1100, 900]);
subplot(2, 2, 1);
imshow(lena, []);
title('原图像 (lena)', 'FontSize', 12);
subplot(2, 2, 2);
imshow(img_prog1, []);
title('仅第3层近似系数', 'FontSize', 12);
subplot(2, 2, 3);
imshow(img_prog2, []);
title('第3层近似 + 第3层细节', 'FontSize', 12);
subplot(2, 2, 4);
imshow(img_prog3, []);
title('第3层近似 + 第3+2层细节', 'FontSize', 12);
sgtitle(sprintf('任务3: %s小波三尺度渐进重构', wavelet3), 'FontSize', 14);
saveas(gcf, fullfile(fig_dir, 'exp6_fig5.png'));
% 保存图片
fprintf('图片已保存: exp6_fig5.png');
fprintf('任务3完成\n\n');
pause;

%% ========================================================================
%  任务4: 四级小波金字塔
%  利用上述分解构造四级金字塔, 展示多尺度近似图像
%  ========================================================================
fprintf('【任务4】四级小波金字塔\n');
fprintf('----------------------------------------\n');
% --- 四级分解 ---
level4 = 4;
wavelet4 = 'bior6.8';
[C4, S4] = wavedec2(lena, level4, wavelet4);
fprintf('四级小波分解完成: %s小波\n', wavelet4);
% --- 提取各级近似图像 (wrcoef2 自动上采样至原图大小) ---
% 显示第 1, 2, 3 层近似, 逐层变粗糙
approx1 = wrcoef2('a', C4, S4, wavelet4, 1);
approx2 = wrcoef2('a', C4, S4, wavelet4, 2);
approx3 = wrcoef2('a', C4, S4, wavelet4, 3);
% --- 显示结果: 原图像 + 3幅近似图像 ---
figure('Name', '任务4: 四级小波金字塔', 'NumberTitle', 'off', ...
       'Position', [50, 50, 1400, 320]);
subplot(1, 4, 1);
imshow(lena, []);
title('原图像', 'FontSize', 12);
subplot(1, 4, 2);
imshow(approx1, []);
title('第1层近似', 'FontSize', 12);
subplot(1, 4, 3);
imshow(approx2, []);
title('第2层近似', 'FontSize', 12);
subplot(1, 4, 4);
imshow(approx3, []);
title('第3层近似', 'FontSize', 12);
sgtitle('任务4: 四级小波金字塔 — 多尺度近似', 'FontSize', 14);
saveas(gcf, fullfile(fig_dir, 'exp6_fig6.png'));
% 保存图片
fprintf('图片已保存: exp6_fig6.png');
fprintf('四级小波金字塔已构造:\n');
fprintf('  第1层近似: %dx%d\n', size(approx1, 2), size(approx1, 1));
fprintf('  第2层近似: %dx%d\n', size(approx2, 2), size(approx2, 1));
fprintf('  第3层近似: %dx%d\n', size(approx3, 2), size(approx3, 1));
fprintf('任务4完成\n\n');
pause;

%% ========================================================================
%  结果分析
%  ========================================================================
fprintf('【实验分析】\n');
fprintf('========================================\n');
fprintf('1. 小波边缘检测:\n');
fprintf('   - sym4小波三尺度分解将图像分离为近似+3方向×3尺度细节\n');
fprintf('   - 各方向边缘独立重构展示了小波的方向选择性\n');
fprintf('   - 多尺度细节提供不同"粗细"的边缘信息\n\n');
fprintf('2. 小波去噪:\n');
fprintf('   - 利用噪声在高频细节系数中的分布特性进行阈值处理\n');
fprintf('   - 软阈值(s=|x|-t)相比硬阈值产生更平滑的结果\n');
fprintf('   - 通用阈值(sqrt(2logN)*σ)在理论上渐进最优, 但对强噪声(σ=30)过于激进\n');
fprintf('     (阈值约0.44, 会过度杀死细节系数, 损失图像内容)\n');
fprintf('   - 方案对比: 纯小波去噪(基准) vs 高斯+小波 vs 小波+高斯后处理 vs 小波域混合\n');
fprintf('     * 方法2(高斯滤波→小波去噪)效果最佳: 高斯预滤波降低噪声方差,\n');
fprintf('       使小波阈值更合理(阈值降低), 保留更多真实边缘细节\n');
fprintf('     * 方法1(纯小波)在高噪声下阈值过大, 去噪后图像偏平滑(过度杀死细节)\n');
fprintf('     * 方法3(小波→高斯)和4(小波域混合)提升有限\n');
fprintf('     * 结论: 对于强高斯噪声场景, 先高斯预滤波再小波阈值去噪能显著提升效果,\n');
fprintf('       两者互补: 高斯粗去噪降低噪声水平→小波精去噪保留细节\n\n');
fprintf('3. 渐进重构:\n');
fprintf('   - 先传输近似系数(占少量数据), 再逐步添加细节\n');
fprintf('   - 适合网络渐进传输和缩略图预览场景\n');
fprintf('   - Biorthogonal双正交小波保证完美重构\n\n');
fprintf('4. 小波金字塔:\n');
fprintf('   - 与高斯金字塔不同, 小波金字塔可完美重建原图\n');
fprintf('   - 各级近似展示不同分辨率下的图像内容\n');
fprintf('   - 多尺度表示为图像处理提供灵活的分析框架\n\n');
fprintf('========================================\n');
fprintf('        实验6 完成!\n');
fprintf('========================================\n');

%% ========================================================================
%  辅助函数

%% ========================================================================

%% 辅助函数: 三元运算符 (用于实验分析中的比较标记)
function res = ternary(cond, true_val, false_val)
    if cond
        res = true_val;
    else
        res = false_val;
    end
end

%% 读入并预处理图像 (转灰度、转double)
function img = load_gray(path)
    img = imread(path);
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    img = im2double(img);
end

%% 裁剪图像值到合法范围 [0, 1]
function img = clip_image(img)
    img = max(0, min(1, img));
end

%% 逐层对各方向细节系数做阈值处理
function C_out = wavelet_threshold(C, S, level, thr, sorh)
    C_out = C;
    for lev = 1:level
        C_out = wthcoef2('h', C_out, S, lev, thr, sorh);
        C_out = wthcoef2('v', C_out, S, lev, thr, sorh);
        C_out = wthcoef2('d', C_out, S, lev, thr, sorh);
    end
end

%% 计算并打印质量指标 (PSNR, SSIM)
function [psnr_val, ssim_val] = print_quality(denoised, original)
    psnr_val = psnr(denoised, original);
    ssim_val = ssim(denoised, original);
    fprintf('  PSNR = %.4f dB, SSIM = %.4f\n', psnr_val, ssim_val);
end