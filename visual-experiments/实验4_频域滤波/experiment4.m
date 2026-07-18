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

%% 实验4: 频域滤波
% 课程: 视觉与数据计算
% 重点函数: fft2, ifft2, fftshift, dct2, idct2, dctmtx

clear all;
close all;
clc;

% 创建图片保存目录
fig_dir = fullfile(fileparts(mfilename('fullpath')), 'figs');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

fprintf('========================================\n');
fprintf('          实验4: 频域滤波\n');
fprintf('========================================\n\n');

%% 读入图像
img_path = 'lena.png';

img = imread(img_path);
if size(img, 3) == 3
    img_gray = rgb2gray(img);
else
    img_gray = img;
end
img_gray = im2double(img_gray);

fprintf('使用图像: %s\n', img_path);
fprintf('图像大小: %d x %d\n\n', size(img_gray, 1), size(img_gray, 2));

%% 自动加载优化参数（由 tune_parameters.m 生成）
optimized_params_loaded = false;
if exist('optimal_params.mat', 'file')
    load('optimal_params.mat');
    fprintf('【自动加载】已应用优化参数: D0=%d, threshold倍率=%.1f, block_size=%d\n', ...
        optimal_D0, optimal_threshold_mult, optimal_block_size);
    optimized_params_loaded = true;
else
    fprintf('【注意】未找到 optimal_params.mat，使用默认参数\n');
    fprintf('  运行 tune_parameters.m 可生成优化参数文件\n\n');
end

%% 参数初始化（默认值，若已加载优化参数则覆盖）
D0 = 50;                % 高斯低通截止频率（默认）
threshold_mult = 2.5;   % 硬阈值倍率（默认）
block_size = 8;         % DCT块大小（默认）
if optimized_params_loaded
    D0 = optimal_D0;
    threshold_mult = optimal_threshold_mult;
    block_size = optimal_block_size;
end
fprintf('  参数: D0=%d, threshold倍率=%.1f, block_size=%d\n\n', D0, threshold_mult, block_size);

%% 任务1: 频谱和相角分析
fprintf('【任务1】频谱和相角分析\n');
fprintf('----------------------------------------\n');

% 计算二维FFT
F = fft2(img_gray);
F_shifted = fftshift(F);

% 计算频谱（幅度）和相角
magnitude = abs(F_shifted);
phase = angle(F_shifted);

% 对数变换用于显示频谱
magnitude_log = log(1 + magnitude);

% 只用频谱重建（相角为0）
F_mag_only = magnitude .* exp(1j * 0);
img_mag_only = real(ifft2(ifftshift(F_mag_only)));

% 只用相角重建（频谱为1）
F_phase_only = exp(1j * phase);
img_phase_only = real(ifft2(ifftshift(F_phase_only)));

% 显示结果
figure('Name', '任务1: 频谱和相角分析', 'NumberTitle', 'off', 'Position', [50, 50, 1200, 900]);

subplot(2, 2, 1);
imshow(img_gray, []);
title('原图像');

subplot(2, 2, 2);
imshow(phase, [-pi pi]);
title('相角图像');
colorbar;

subplot(2, 2, 3);
imshow(img_mag_only, []);
title('只用频谱重建 (相角=0)');

subplot(2, 2, 4);
imshow(img_phase_only, []);
title('只用相角重建 (频谱=1)');
sgtitle('任务1: 频谱和相角分析', 'FontSize', 14, 'FontWeight', 'bold');

% 保存图片
saveas(gcf, fullfile(fig_dir, 'exp4_task1_spectrum_phase.png'));
fprintf('图片已保存: exp4_task1_spectrum_phase.png\n');

fprintf('分析: 相角信息对图像结构更重要，只用相角重建的图像能辨认出轮廓\n');
fprintf('      而只用频谱重建的图像则失去结构信息\n\n');

%% 任务2: 高斯卷积与高斯低通滤波比较
fprintf('【任务2】高斯卷积与高斯低通滤波比较\n');
fprintf('----------------------------------------\n');

% 添加高斯噪声
sigma_noise = 30;
rng(0);
img_noisy = img_gray + (sigma_noise/255) * randn(size(img_gray));
img_noisy = max(0, min(1, img_noisy));

% 方法1: 空间域高斯卷积
h_gaussian = fspecial('gaussian', [15 15], 2);
img_gaussian_conv = imfilter(img_noisy, h_gaussian, 'replicate');

% 方法2: 频率域高斯低通滤波
% 创建高斯低通滤波器
[M, N] = size(img_gray);
[u, v] = meshgrid(1:N, 1:M);
u = u - N/2;
v = v - M/2;
D = sqrt(u.^2 + v.^2);
% D0 已在参数初始化块中设置
H_gaussian_lp = exp(-(D.^2) / (2 * D0^2));

% 频域滤波
F_noisy = fft2(img_noisy);
F_noisy_shifted = fftshift(F_noisy);
F_filtered = F_noisy_shifted .* H_gaussian_lp;
img_gaussian_freq = real(ifft2(ifftshift(F_filtered)));

% 计算质量指标
psnr_noisy = psnr(img_noisy, img_gray);
ssim_noisy = ssim(img_noisy, img_gray);

psnr_conv = psnr(img_gaussian_conv, img_gray);
ssim_conv = ssim(img_gaussian_conv, img_gray);

psnr_freq = psnr(img_gaussian_freq, img_gray);
ssim_freq = ssim(img_gaussian_freq, img_gray);

fprintf('噪声图像: PSNR=%.4f dB, SSIM=%.4f\n', psnr_noisy, ssim_noisy);
fprintf('高斯卷积: PSNR=%.4f dB, SSIM=%.4f\n', psnr_conv, ssim_conv);
fprintf('高斯低通: PSNR=%.4f dB, SSIM=%.4f\n', psnr_freq, ssim_freq);

% 显示结果
figure('Name', '任务2: 高斯卷积与频域滤波', 'NumberTitle', 'off', 'Position', [50, 50, 1200, 900]);

subplot(2, 2, 1);
imshow(img_gray, []);
title('原图像');

subplot(2, 2, 2);
imshow(img_noisy, []);
title({sprintf('噪声图像'), sprintf('PSNR=%.4f, SSIM=%.4f', psnr_noisy, ssim_noisy)});

subplot(2, 2, 3);
imshow(img_gaussian_conv, []);
title({sprintf('高斯卷积(空间域)'), sprintf('PSNR=%.4f, SSIM=%.4f', psnr_conv, ssim_conv)});

subplot(2, 2, 4);
imshow(img_gaussian_freq, []);
title({sprintf('高斯低通(频率域)'), sprintf('PSNR=%.4f, SSIM=%.4f', psnr_freq, ssim_freq)});
sgtitle(sprintf('任务2: 高斯卷积与频域低通滤波 (D0=%d)', D0), 'FontSize', 14, 'FontWeight', 'bold');

% 保存图片
saveas(gcf, fullfile(fig_dir, 'exp4_task2_conv_vs_freq.png'));
fprintf('图片已保存: exp4_task2_conv_vs_freq.png\n');

%% 任务3: 三维显示Fourier频谱
fprintf('\n【任务3】三维显示Fourier频谱\n');
fprintf('----------------------------------------\n');

% 预计算任务3所需的频谱（复用任务2的数据，避免重复FFT）
F_conv_shifted = fftshift(fft2(img_gaussian_conv));
F_freq_shifted = fftshift(fft2(img_gaussian_freq));
F_conv_mag = log(1 + abs(F_conv_shifted));
F_freq_mag = log(1 + abs(F_freq_shifted));

figure('Name', '任务3: 三维Fourier频谱', 'NumberTitle', 'off', 'Position', [50, 50, 1200, 900]);
colormap(jet);

% 原图像频谱
subplot(2, 2, 1);
surf(magnitude_log, 'EdgeColor', 'none', 'FaceAlpha', 0.85);
title('原图像频谱');
xlabel('u'); ylabel('v'); zlabel('log(1+|F|)');
view(45, 30); lighting gouraud; light;

% 噪声图像频谱（利用任务2已计算的 F_noisy_shifted，避免重复 FFT）
F_noisy_mag = log(1 + abs(F_noisy_shifted));
subplot(2, 2, 2);
surf(F_noisy_mag, 'EdgeColor', 'none', 'FaceAlpha', 0.85);
title('噪声图像频谱');
xlabel('u'); ylabel('v'); zlabel('log(1+|F|)');
view(45, 30); lighting gouraud; light;

% 高斯卷积结果频谱
subplot(2, 2, 3);
surf(F_conv_mag, 'EdgeColor', 'none', 'FaceAlpha', 0.85);
title('高斯卷积结果频谱');
xlabel('u'); ylabel('v'); zlabel('log(1+|F|)');
view(45, 30); lighting gouraud; light;

% 高斯低通结果频谱
subplot(2, 2, 4);
surf(F_freq_mag, 'EdgeColor', 'none', 'FaceAlpha', 0.85);
title('高斯低通结果频谱');
xlabel('u'); ylabel('v'); zlabel('log(1+|F|)');
view(45, 30); lighting gouraud; light;
sgtitle('任务3: 三维Fourier频谱', 'FontSize', 14, 'FontWeight', 'bold');

% 保存图片
saveas(gcf, fullfile(fig_dir, 'exp4_task3_3d_spectrum.png'));
fprintf('图片已保存: exp4_task3_3d_spectrum.png\n');

%% 任务4: DCT变换与反变换
fprintf('\n【任务4】DCT变换与反变换\n');
fprintf('----------------------------------------\n');

% 截取8×8图像块
block = img_gray(262:269, 262:269);

% 方法1: 使用dct2函数
dct_result = dct2(block);
block_reconstructed = idct2(dct_result);

% 方法2: 使用DCT变换矩阵
C8 = dctmtx(8);
dct_result_matrix = C8 * block * C8';
block_reconstructed_matrix = C8' * dct_result_matrix * C8;

% 计算重建误差
error_func = max(abs(block_reconstructed(:) - block(:)));
error_matrix = max(abs(block_reconstructed_matrix(:) - block(:)));

fprintf('图像块位置: (262:269, 262:269)\n');
fprintf('方法1 (dct2函数) 最大重建误差: %e\n', error_func);
fprintf('方法2 (变换矩阵) 最大重建误差: %e\n', error_matrix);

% 显示结果
figure('Name', '任务4: DCT变换与反变换', 'NumberTitle', 'off', 'Position', [50, 50, 1400, 400]);

subplot(1, 4, 1);
imshow(block, []);
title('原图像块 (8×8)');

subplot(1, 4, 2);
imshow(log(abs(dct_result) + 1), []);
title('DCT系数 (对数)');
colorbar;

subplot(1, 4, 3);
imshow(block_reconstructed, []);
title({sprintf('dct2重建'), sprintf('误差=%e', error_func)});

subplot(1, 4, 4);
imshow(block_reconstructed_matrix, []);
title({sprintf('矩阵法重建'), sprintf('误差=%e', error_matrix)});
sgtitle('任务4: DCT变换与反变换', 'FontSize', 14, 'FontWeight', 'bold');

% 保存图片
saveas(gcf, fullfile(fig_dir, 'exp4_task4_dct.png'));
fprintf('图片已保存: exp4_task4_dct.png\n');

%% 任务5: 3D DCT硬阈值去噪
fprintf('\n【任务5】3D DCT硬阈值去噪\n');
fprintf('----------------------------------------\n');

% 读入彩色图像
img_color = imread(img_path);
if size(img_color, 3) ~= 3
    % 灰度图复制成3个伪通道，仍然可以演示3D DCT的效果
    fprintf('注意: 输入为灰度图，复制为3个伪通道以演示3D DCT\n');
    img_color = repmat(im2double(img_color), [1, 1, 3]);
else
    img_color = im2double(img_color);
end

% 添加高斯噪声
rng(0);
noisy_color = img_color + (sigma_noise/255) * randn(size(img_color));
noisy_color = max(0, min(1, noisy_color));

% 3D DCT硬阈值去噪
% 将彩色图像视为三维张量进行3D DCT
% 对每个8x8x3块沿三个维度同时做DCT（真3D DCT）
[H, W, C] = size(noisy_color);
denoised_color = zeros(size(noisy_color));

% 噪声在像素域的std = sigma_noise/255
% DCT是正交变换，系数域的噪声std与像素域相同
% 硬阈值取 2.5 倍噪声std，平衡去噪与保细节
noise_std = sigma_noise / 255;
threshold = threshold_mult * noise_std;
% block_size 已在参数初始化块中设置

% 【性能优化】预计算 DCT 变换矩阵，用 BLAS 矩阵乘法替代 dct() 函数调用
% dct() 对小块有大量函数调用开销；矩阵乘法由多线程 BLAS 加速
DCT_mat = dctmtx(block_size);

for i = 1:block_size:H
    for j = 1:block_size:W
        % 提取块
        block_i_end = min(i+block_size-1, H);
        block_j_end = min(j+block_size-1, W);
        block = noisy_color(i:block_i_end, j:block_j_end, :);

        % 如果是完整的block_size块（对3通道张量做真3D DCT）
        if size(block, 1) == block_size && size(block, 2) == block_size
            % 【优化】2D DCT: DCT_mat * X * DCT_mat' (BLAS 矩阵乘法)
            % vs 旧方法: dct(dct(X, [], 1), [], 2) (FFT-based, 大量函数调用)
            for c = 1:C
                dct_block(:, :, c) = DCT_mat * block(:, :, c) * DCT_mat';
            end
            % 通道维DCT（C=3 很小，dct() 开销可忽略）
            dct_block = dct(dct_block, [], 3);

            % 硬阈值（将绝对值小于阈值的系数置零）
            dct_block(abs(dct_block) < threshold) = 0;

            % 【优化】3D逆变换: 通道维 IDCT + 2D IDCT 矩阵乘法
            block_denoised = idct(dct_block, [], 3);
            for c = 1:C
                block_denoised(:, :, c) = DCT_mat' * block_denoised(:, :, c) * DCT_mat;
            end
            denoised_color(i:i+block_size-1, j:j+block_size-1, :) = block_denoised;
        else
            % 不完整的块直接复制
            denoised_color(i:block_i_end, j:block_j_end, :) = block;
        end
    end
end

% 裁剪到有效范围
denoised_color = max(0, min(1, denoised_color));

% 计算质量指标
psnr_noisy_color = psnr(noisy_color, img_color);
ssim_noisy_color = ssim(noisy_color, img_color);

psnr_denoised = psnr(denoised_color, img_color);
ssim_denoised = ssim(denoised_color, img_color);

fprintf('彩色图像去噪结果:\n');
fprintf('  噪声图像: PSNR=%.4f dB, SSIM=%.4f\n', psnr_noisy_color, ssim_noisy_color);
fprintf('  去噪图像: PSNR=%.4f dB, SSIM=%.4f\n', psnr_denoised, ssim_denoised);

% 显示结果
figure('Name', '任务5: 3D DCT硬阈值去噪', 'NumberTitle', 'off', 'Position', [50, 50, 1200, 400]);

subplot(1, 3, 1);
imshow(img_color, []);
title('原彩色图像');

subplot(1, 3, 2);
imshow(noisy_color, []);
title({sprintf('噪声图像'), sprintf('PSNR=%.4f, SSIM=%.4f', psnr_noisy_color, ssim_noisy_color)});

subplot(1, 3, 3);
imshow(denoised_color, []);
title({sprintf('3D DCT去噪'), sprintf('PSNR=%.4f, SSIM=%.4f', psnr_denoised, ssim_denoised)});
sgtitle('任务5: 3D DCT硬阈值去噪', 'FontSize', 14, 'FontWeight', 'bold');

% 保存图片
saveas(gcf, fullfile(fig_dir, 'exp4_task5_3ddct.png'));
fprintf('图片已保存: exp4_task5_3ddct.png\n');

%% 结果分析
fprintf('\n【分析】\n');
fprintf('1. 频谱与相角 (图1): 相角(右上)包含结构信息→相角重建(右下)可辨轮廓;\n');
fprintf('   幅值重建(左下)丢失结构; 说明相角对图像结构更重要\n');
fprintf('2. 空间域vs频率域 (图2): 高斯卷积 PSNR=%.4f, 高斯低通 PSNR=%.4f;\n', psnr_conv, psnr_freq);
fprintf('   两者效果相近, 但频域滤波对应理想低通(无核截断), 空间域受核大小限制\n');
fprintf('3. 3D频谱 (图3): 噪声抬高高频, 两种滤波均衰减高频成分;\n');
fprintf('   比较四图可见低频(中心)保留, 高频(边缘)被抑制\n');
fprintf('4. DCT变换 (图4): dct2函数与矩阵法最大误差皆 ~10^{-15} (浮点精度);\n');
fprintf('   验证了 DCT 的正交完备性和变换矩阵 C_8 的正确性\n');
fprintf('5. 3D DCT去噪 (图5): 硬阈值 threshold=%.1f×σ_noise, 阈值=%.4f;\n', threshold_mult, threshold);
fprintf('   噪声 PSNR=%.2f→去噪 PSNR=%.2f (提升%.2f dB); SSIM=%.4f\n', ...
    psnr_noisy_color, psnr_denoised, psnr_denoised-psnr_noisy_color, ssim_denoised);
fprintf('   阈值越高去噪越强, 但细节(边缘/纹理)损失也越大, 需权衡\n');

fprintf('\n========================================\n');
fprintf('        实验4 完成!\n');
fprintf('========================================\n');

close all;
