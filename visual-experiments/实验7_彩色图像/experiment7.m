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

%% 实验7: 彩色图像处理
% 课程: 视觉与数据计算
% 重点函数: rgb2ycbcr, rgb2lab, rgb2hsv
clear all;
close all;
clc;
% 创建图片保存目录
fig_dir = fullfile(fileparts(mfilename('fullpath')), 'figs');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
% 添加 BM3D 算法路径
addpath('../BM3D');                    % CBM3D 核心函数
% 参数配置
cfg = struct();
cfg.sigma_noise = 30;                 % 高斯噪声标准差
cfg.med_window = [5 5];               % 中值滤波窗口
cfg.gauss_sigma = 1.5;                % 高斯滤波 sigma
cfg.n_denoisers = 6;                  % 去噪方法数
cfg.n_colorspaces = 3;                % 颜色空间数
fprintf('========================================\n');
fprintf('        实验7: 彩色图像处理\n');
fprintf('========================================\n\n');

%% 任务1: RGB到YCbCr颜色空间转换
fprintf('【任务1】RGB到YCbCr颜色空间转换\n');
fprintf('----------------------------------------\n');
% 读入彩色house图像
house_color_path = 'image_House256rgb.png';
house_color = imread(house_color_path);
% 转换到YCbCr
house_ycbcr = rgb2ycbcr(house_color);
% 分离通道
Y = house_ycbcr(:,:,1);   % 亮度
Cb = house_ycbcr(:,:,2);  % 蓝色色度
Cr = house_ycbcr(:,:,3);  % 红色色度
fprintf('YCbCr颜色空间:\n');
fprintf('  Y (亮度): 范围[16, 235]\n');
fprintf('  Cb (蓝色色度): 范围[16, 240]，128表示无色度\n');
fprintf('  Cr (红色色度): 范围[16, 240]，128表示无色度\n');
% 显示结果
figure('Name', '任务1: YCbCr颜色空间', 'NumberTitle', 'off', 'Position', [50, 50, 1400, 400]);
subplot(1, 4, 1);
imshow(house_color);
title('原RGB图像');
subplot(1, 4, 2);
imshow(Y, []);
title('Y (亮度)');
subplot(1, 4, 3);
imshow(Cb, []);
title('Cb (蓝色色度)');
subplot(1, 4, 4);
imshow(Cr, []);
title('Cr (红色色度)');

%% 任务2: RGB到Lab颜色空间转换及噪声分析
fprintf('\n【任务2】RGB到Lab颜色空间转换及噪声分析\n');
fprintf('----------------------------------------\n');
% 读入彩色lena图像
lena_color_path = 'image_lena512rgb.png';
lena_color = imread(lena_color_path);
lena_color = im2double(lena_color);
% 保存干净图像在Lab空间下的值（用于计算噪声的标准差）
lena_color_lab_clean = rgb2lab(lena_color);
% 添加高斯噪声
sigma_noise = cfg.sigma_noise;
rng(0);
lena_noisy = lena_color + (sigma_noise/255) * randn(size(lena_color));
lena_noisy = max(0, min(1, lena_noisy));
% 转换到Lab颜色空间
lena_lab = rgb2lab(lena_noisy);
% 分离通道
L = lena_lab(:,:,1);   % 亮度
a_ch = lena_lab(:,:,2);   % 绿-红轴
b_ch = lena_lab(:,:,3);   % 蓝-黄轴
fprintf('Lab颜色空间:\n');
fprintf('  L (亮度): 范围[0, 100]\n');
fprintf('  a (绿-红): 负值偏绿，正值偏红\n');
fprintf('  b (蓝-黄): 负值偏蓝，正值偏黄\n');
% 计算各通道噪声标准差（用噪声图像减去原始干净图像的Lab值，得到纯噪声分量）
L_clean = lena_color_lab_clean(:,:,1);
a_clean = lena_color_lab_clean(:,:,2);
b_clean = lena_color_lab_clean(:,:,3);
std_L_noise = std(L(:) - L_clean(:));
std_a_noise = std(a_ch(:) - a_clean(:));
std_b_noise = std(b_ch(:) - b_clean(:));
fprintf('\n各通道噪声标准差:\n');
fprintf('  L通道: %.4f\n', std_L_noise);
fprintf('  a通道: %.4f\n', std_a_noise);
fprintf('  b通道: %.4f\n', std_b_noise);
% 显示结果
figure('Name', '任务2: Lab颜色空间噪声分析', 'NumberTitle', 'off', 'Position', [50, 50, 1200, 900]);
subplot(2, 2, 1);
imshow(lena_noisy, []);
title('噪声RGB图像');
subplot(2, 2, 2);
imshow(L, [0 100]);
title({'L (亮度)', sprintf('std_noise=%.4f', std_L_noise)});
subplot(2, 2, 3);
imshow(a_ch, [-128 127]);
title({'a (绿-红)', sprintf('std_noise=%.4f', std_a_noise)});
subplot(2, 2, 4);
imshow(b_ch, [-128 127]);
title({'b (蓝-黄)', sprintf('std_noise=%.4f', std_b_noise)});

%% 任务3: 彩色图像去噪方法比较（增强版）
fprintf('\n【任务3】彩色图像去噪方法比较（增强版）\n');
fprintf('  直接方法: 6种去噪方式遍历RGB通道 → 选最优\n');
fprintf('  间接方法: 3种颜色空间 × 6种去噪方式 + CBM3D → 选最优\n');
fprintf('----------------------------------------\n');
psnr_noisy = psnr(lena_noisy, lena_color);
ssim_noisy = ssim(lena_noisy, lena_color);

%% ===== 一、直接方法：多种去噪方式遍历RGB三通道 =====
fprintf('\n--- 一、直接方法: 各去噪方式对比 ---\n');
direct_methods  = {'medfilt2', 'gaussian', 'bilateral', 'wiener', 'nlm', 'diffusion'};
direct_labels   = {'medfilt2[5×5]', 'imgaussfilt σ=1.5', 'imbilatfilt', ...
                   'wiener2[5×5]', 'imnlmfilt', 'imdiffusefilt'};
n_direct = length(direct_methods);
direct_results = cell(n_direct, 1);
direct_psnr = zeros(n_direct, 1);
direct_ssim = zeros(n_direct, 1);
for k = 1:n_direct
    method = direct_methods{k};
    denoised = zeros(size(lena_noisy));
    for c = 1:3
        denoised(:,:,c) = denoise_channel(lena_noisy(:,:,c), method);
    end
    direct_psnr(k) = psnr(denoised, lena_color);
    direct_ssim(k) = ssim(denoised, lena_color);
    direct_results{k} = denoised;
    fprintf('  直接: %-20s PSNR=%.4f dB, SSIM=%.4f\n', direct_labels{k}, direct_psnr(k), direct_ssim(k));
end
[best_d_psnr, best_d_idx] = max(direct_psnr);
best_direct_name = direct_labels{best_d_idx};
best_direct_result = direct_results{best_d_idx};
fprintf('  ★ 最优直接方法: %s  (PSNR=%.4f, SSIM=%.4f)\n', best_direct_name, best_d_psnr, direct_ssim(best_d_idx));

%% ===== 二、间接方法：颜色空间 × 去噪方式遍历 =====
fprintf('\n--- 二、间接方法: 颜色空间×去噪方式对比 ---\n');
color_spaces = {'YCbCr', 'Lab', 'HSV'};
cs_lum_idx   = [1, 1, 3];         % 亮度通道索引
denoisers    = {'medfilt2', 'gaussian', 'bilateral', 'wiener', 'nlm', 'diffusion'};
denoiser_labels = {'medfilt2[5×5]', 'imgaussfilt σ=1.5', 'imbilatfilt', ...
                   'wiener2[5×5]', 'imnlmfilt', 'imdiffusefilt'};
n_cs = length(color_spaces);
n_dn = length(denoisers);
indirect_results = cell(n_cs, n_dn);
indirect_psnr = zeros(n_cs, n_dn);
indirect_ssim = zeros(n_cs, n_dn);
for cs = 1:n_cs
    % 转换到目标颜色空间
    switch color_spaces{cs}
        case 'YCbCr', cs_img = rgb2ycbcr(lena_noisy);
        case 'Lab',   cs_img = rgb2lab(lena_noisy);
        case 'HSV',   cs_img = rgb2hsv(lena_noisy);
    end
    lum_idx = cs_lum_idx(cs);
    lum_ch  = cs_img(:,:,lum_idx);
    for d = 1:n_dn
        dl = denoise_channel(lum_ch, denoisers{d});
        cs_den = cs_img;
        cs_den(:,:,lum_idx) = dl;
        % 反变换回RGB
        switch color_spaces{cs}
            case 'YCbCr', rgb_den = ycbcr2rgb(cs_den);
            case 'Lab',   rgb_den = max(0, min(1, lab2rgb(cs_den)));
            case 'HSV',   rgb_den = hsv2rgb(cs_den);
        end
        indirect_psnr(cs, d) = psnr(rgb_den, lena_color);
        indirect_ssim(cs, d) = ssim(rgb_den, lena_color);
        indirect_results{cs, d} = rgb_den;
        fprintf('  间接: %-6s + %-17s PSNR=%.4f dB, SSIM=%.4f\n', ...
            color_spaces{cs}, denoiser_labels{d}, indirect_psnr(cs, d), indirect_ssim(cs, d));
    end
end

%% ---- CBM3D 加入间接方法对比 ----
% 同时计算 opp 和 yCbCr 两种颜色空间，opp 用于间接方法最优选择
[~, lena_cbm3d] = CBM3D(1, lena_noisy, sigma_noise, 'np', 0, 'opp');
[~, lena_cbm3d_ycbcr] = CBM3D(1, lena_noisy, sigma_noise, 'np', 0, 'yCbCr');
psnr_cbm3d = psnr(lena_cbm3d, lena_color);
ssim_cbm3d = ssim(lena_cbm3d, lena_color);
psnr_cbm3d_ycbcr = psnr(lena_cbm3d_ycbcr, lena_color);
ssim_cbm3d_ycbcr = ssim(lena_cbm3d_ycbcr, lena_color);
fprintf('  间接: CBM3D (opponent)    PSNR=%.4f dB, SSIM=%.4f\n', psnr_cbm3d, ssim_cbm3d);
fprintf('  间接: CBM3D (yCbCr)      PSNR=%.4f dB, SSIM=%.4f\n', psnr_cbm3d_ycbcr, ssim_cbm3d_ycbcr);
if psnr_cbm3d >= psnr_cbm3d_ycbcr
    cbm3d_best_space = 'opp';
else
    cbm3d_best_space = 'yCbCr';
end
fprintf('  → CBM3D 推荐颜色空间: %s\n', cbm3d_best_space);
% 综合所有间接方法（含CBM3D）选出最优
[best_i_psnr_grid, best_linear] = max(indirect_psnr(:));
[best_cs_idx, best_dn_idx] = ind2sub([n_cs, n_dn], best_linear);
if psnr_cbm3d > best_i_psnr_grid
    best_i_psnr   = psnr_cbm3d;
    best_i_ssim   = ssim_cbm3d;
    best_indirect_name = 'CBM3D (opponent)';
    best_indirect_result = lena_cbm3d;
else
    best_i_psnr   = best_i_psnr_grid;
    best_i_ssim   = indirect_ssim(best_cs_idx, best_dn_idx);
    best_indirect_name = sprintf('%s + %s', color_spaces{best_cs_idx}, denoiser_labels{best_dn_idx});
    best_indirect_result = indirect_results{best_cs_idx, best_dn_idx};
end
fprintf('  ★ 最优间接方法: %s  (PSNR=%.4f, SSIM=%.4f)\n', best_indirect_name, best_i_psnr, best_i_ssim);

%% ===== 三、保留参考方法及调参用变量 =====
% YCbCr 参考（Y双边+Cb/Cr高斯，与遍历方法不同——遍历只处理亮度通道）
ycbcr_work = rgb2ycbcr(lena_noisy);
y_bilateral = denoise_channel(ycbcr_work(:,:,1), 'bilateral');
cb_gauss    = denoise_channel(ycbcr_work(:,:,2), 'gaussian');
cr_gauss    = denoise_channel(ycbcr_work(:,:,3), 'gaussian');
lena_ycbcr_denoised = ycbcr2rgb(cat(3, y_bilateral, cb_gauss, cr_gauss));
psnr_ycbcr = psnr(lena_ycbcr_denoised, lena_color);
ssim_ycbcr = ssim(lena_ycbcr_denoised, lena_color);
% HSV 参考（已包含在遍历中，直接引用结果）
hsv_work = rgb2hsv(lena_noisy);
lena_hsv_denoised = indirect_results{3, 5};
psnr_hsv = indirect_psnr(3, 5);
ssim_hsv = indirect_ssim(3, 5);
% illumpca 参考（光照校正+PCA去噪，独立方法）
lin_img = rgb2lin(lena_noisy);
illuminant = illumpca(lin_img);
wb_img = chromadapt(lena_noisy, illuminant);
wb_ycbcr = rgb2ycbcr(wb_img);
wb_y  = imbilatfilt(wb_ycbcr(:,:,1));
wb_cb = imgaussfilt(wb_ycbcr(:,:,2), 2);
wb_cr = imgaussfilt(wb_ycbcr(:,:,3), 2);
lena_illumpca_denoised = ycbcr2rgb(cat(3, wb_y, wb_cb, wb_cr));
psnr_illum = psnr(lena_illumpca_denoised, lena_color);
ssim_illum = ssim(lena_illumpca_denoised, lena_color);

%% CBM3D 已在上面间接方法中计算, 不再重复
% 保留以下参考方法(YCbCr+bilateral, HSV+NLM, illumpca)供后续调参使用

%% ===== 四、汇总输出 =====
fprintf('\n========== 去噪结果汇总 ==========\n');
% 格式化表格输出（方案3）
fprintf('┌──────────────────────────────┬──────────┬──────────┬───────────────┐\n');
fprintf('│ 方法                         │ PSNR(dB) │  SSIM    │ 类型          │\n');
fprintf('├──────────────────────────────┼──────────┼──────────┼───────────────┤\n');
% 噪声行
fprintf('│ %-28s │ %8.4f │ %8.4f │ %-13s │\n', '噪声图像', psnr_noisy, ssim_noisy, '参考');
% 直接方法
for k = 1:n_direct
    tag = '直接';
    if k == best_d_idx, tag = '★直接'; end
    fprintf('│ %-28s │ %8.4f │ %8.4f │ %-13s │\n', direct_labels{k}, direct_psnr(k), direct_ssim(k), tag);
end
fprintf('├──────────────────────────────┼──────────┼──────────┼───────────────┤\n');
% 间接方法
for cs = 1:n_cs
    for d = 1:n_dn
        tag = '间接';
        n = sprintf('%s + %s', color_spaces{cs}, denoiser_labels{d});
        if strcmp(n, best_indirect_name), tag = '★间接'; end
        fprintf('│ %-28s │ %8.4f │ %8.4f │ %-13s │\n', n, indirect_psnr(cs,d), indirect_ssim(cs,d), tag);
    end
end
% CBM3D
cbm3d_tag = '间接';
if strcmp(best_indirect_name, 'CBM3D (opponent)'), cbm3d_tag = '★间接'; end
fprintf('│ %-28s │ %8.4f │ %8.4f │ %-13s │\n', 'CBM3D (opponent)', psnr_cbm3d, ssim_cbm3d, cbm3d_tag);
fprintf('├──────────────────────────────┼──────────┼──────────┼───────────────┤\n');
% 参考方法
fprintf('│ %-28s │ %8.4f │ %8.4f │ %-13s │\n', 'YCbCr+bilateral(ref)', psnr_ycbcr, ssim_ycbcr, '参考');
fprintf('│ %-28s │ %8.4f │ %8.4f │ %-13s │\n', 'HSV+NLM(ref)', psnr_hsv, ssim_hsv, '参考');
fprintf('│ %-28s │ %8.4f │ %8.4f │ %-13s │\n', 'illumpca(ref)', psnr_illum, ssim_illum, '参考');
fprintf('└──────────────────────────────┴──────────┴──────────┴───────────────┘\n');
fprintf('  ★ 最优直接: %s (PSNR=%.4f, SSIM=%.4f)\n', best_direct_name, best_d_psnr, direct_ssim(best_d_idx));
fprintf('  ★ 最优间接: %s (PSNR=%.4f, SSIM=%.4f)\n', best_indirect_name, best_i_psnr, best_i_ssim);
% 逐通道分析（方案4）
fprintf('\n--- 逐通道PSNR分析 ---\n');
fprintf('  通道   最优直接      最优间接      差值(直-间)\n');
ch_labels = {'R', 'G', 'B'};
for c = 1:3
    dp = psnr(best_direct_result(:,:,c), lena_color(:,:,c));
    ip = psnr(best_indirect_result(:,:,c), lena_color(:,:,c));
    diff_str = '';
    if dp > ip, diff_str = '直接优'; elseif ip > dp, diff_str = '间接优'; end
    fprintf('  %s       %.4f      %.4f      %+.4f (%s)\n', ...
        ch_labels{c}, dp, ip, dp-ip, diff_str);
end

%% ===== 五、显示结果 =====
% Figure 1: 最优直接 vs 最优间接 对比
figure('Name', '任务3: 直接方法 vs 间接方法', 'NumberTitle', 'off', 'Position', [50, 50, 1000, 450]);
subplot(1, 2, 1);
imshow(best_direct_result);
title({sprintf('★ 最优直接方法: %s', best_direct_name), ...
       sprintf('PSNR=%.4f dB, SSIM=%.4f', best_d_psnr, direct_ssim(best_d_idx))});
subplot(1, 2, 2);
imshow(best_indirect_result);
title({sprintf('★ 最优间接方法: %s', best_indirect_name), ...
       sprintf('PSNR=%.4f dB, SSIM=%.4f', best_i_psnr, best_i_ssim)});
sgtitle('直接方法 vs 间接方法 去噪对比', 'FontSize', 14);
% Figure 2: 综合对比概览
figure('Name', '任务3: 彩色图像去噪方法综合对比', 'NumberTitle', 'off', 'Position', [50, 50, 1800, 1100]);
subplot(3, 3, 1);
imshow(lena_noisy, []);
title({sprintf('噪声图像 (sigma=30)'), sprintf('PSNR=%.4f, SSIM=%.4f', psnr_noisy, ssim_noisy)});
subplot(3, 3, 2);
imshow(best_direct_result);
title({sprintf('★最优直接: %s', best_direct_name), sprintf('PSNR=%.4f, SSIM=%.4f', best_d_psnr, direct_ssim(best_d_idx))});
subplot(3, 3, 3);
imshow(best_indirect_result);
title({sprintf('★最优间接: %s', best_indirect_name), sprintf('PSNR=%.4f, SSIM=%.4f', best_i_psnr, best_i_ssim)});
subplot(3, 3, 4);
imshow(lena_ycbcr_denoised, []); title({'YCbCr+bilateral', sprintf('PSNR=%.4f, SSIM=%.4f', psnr_ycbcr, ssim_ycbcr)});
subplot(3, 3, 5);
imshow(lena_hsv_denoised, []);   title({'HSV+NLM', sprintf('PSNR=%.4f, SSIM=%.4f', psnr_hsv, ssim_hsv)});
subplot(3, 3, 6);
imshow(lena_illumpca_denoised, []); title({'illumpca', sprintf('PSNR=%.4f, SSIM=%.4f', psnr_illum, ssim_illum)});
subplot(3, 3, 7);
imshow(lena_cbm3d, []);          title({'CBM3D(间接方法)', sprintf('PSNR=%.4f, SSIM=%.4f', psnr_cbm3d, ssim_cbm3d)});
subplot(3, 3, [8 9]);
axis off;
text(0.05, 0.85, '去噪结果汇总:', 'FontSize', 14, 'FontWeight', 'bold');
text(0.05, 0.72, sprintf('★最优直接: %-18s PSNR=%.2f, SSIM=%.4f', best_direct_name, best_d_psnr, direct_ssim(best_d_idx)), 'FontSize', 11, 'Color', 'b');
text(0.05, 0.60, sprintf('★最优间接: %-18s PSNR=%.2f, SSIM=%.4f', best_indirect_name, best_i_psnr, best_i_ssim), 'FontSize', 11, 'Color', 'r');
text(0.05, 0.48, sprintf('YCbCr+bilateral:        PSNR=%.2f, SSIM=%.4f', psnr_ycbcr, ssim_ycbcr), 'FontSize', 11);
text(0.05, 0.38, sprintf('HSV+NLM:                PSNR=%.2f, SSIM=%.4f', psnr_hsv, ssim_hsv), 'FontSize', 11);
text(0.05, 0.28, sprintf('illumpca:               PSNR=%.2f, SSIM=%.4f', psnr_illum, ssim_illum), 'FontSize', 11);
text(0.05, 0.18, sprintf('CBM3D(间接方法):       PSNR=%.2f, SSIM=%.4f', psnr_cbm3d, ssim_cbm3d), 'FontSize', 11);
text(0.05, 0.08, sprintf('噪声:                   PSNR=%.2f, SSIM=%.4f', psnr_noisy, ssim_noisy), 'FontSize', 11);

%% Figure 3: 误差分布热力图（方案1）
figure('Name', '任务3: 去噪误差分布对比', 'NumberTitle', 'off', 'Position', [50, 50, 1400, 400]);
diff_direct   = abs(double(best_direct_result)   - double(lena_color));
diff_indirect = abs(double(best_indirect_result) - double(lena_color));
diff_diff     = double(diff_indirect) - double(diff_direct);  % 负=间接优, 正=直接优
subplot(1, 3, 1);
imshow(diff_direct, []);
title({sprintf('直接法误差: %s', best_direct_name), sprintf('MAE=%.4f', mean(diff_direct(:)))});
colormap(gca, jet); colorbar;
subplot(1, 3, 2);
imshow(diff_indirect, []);
title({sprintf('间接法误差: %s', best_indirect_name), sprintf('MAE=%.4f', mean(diff_indirect(:)))});
colormap(gca, jet); colorbar;
subplot(1, 3, 3);
imshow(diff_diff, []);
title({'间接 vs 直接 差异', '蓝=间接优  红=直接优'});
colormap(gca, jet); colorbar;
sgtitle('去噪误差分布对比', 'FontSize', 14);
saveas(gcf, fullfile(fig_dir, 'exp7_fig5.png'));
% 保存图片
fprintf('图片已保存: exp7_fig5.png\n');

%% 任务3调参: 中值滤波窗口大小扫描
fprintf('\n--- 任务3调参: 中值滤波窗口大小 ---\n');
med_window_sizes = [3, 5, 7, 9];
psnr_med = zeros(length(med_window_sizes), 1);
ssim_med = zeros(length(med_window_sizes), 1);
for k = 1:length(med_window_sizes)
    ws = med_window_sizes(k);
    med_denoised = zeros(size(lena_noisy));
    for c = 1:3
        med_denoised(:,:,c) = medfilt2(lena_noisy(:,:,c), [ws ws]);
    end
    psnr_med(k) = psnr(med_denoised, lena_color);
    ssim_med(k) = ssim(med_denoised, lena_color);
    fprintf('  窗口[%d %d]: PSNR=%.4f dB, SSIM=%.4f\n', ws, ws, psnr_med(k), ssim_med(k));
end

%% 任务3调参: 高斯滤波sigma扫描(间接法-L通道)
fprintf('\n--- 任务3调参: 高斯滤波sigma扫描(间接法) ---\n');
sigma_values = [0.5, 1, 1.5, 2, 3, 4, 5];
psnr_sig = zeros(length(sigma_values), 1);
ssim_sig = zeros(length(sigma_values), 1);
for k = 1:length(sigma_values)
    sig = sigma_values(k);
    lab_tmp = lena_lab;
    lab_tmp(:,:,1) = imfilter(lena_lab(:,:,1), fspecial('gaussian', [5 5], sig), 'replicate');
    sig_denoised = lab2rgb(lab_tmp);
    sig_denoised = max(0, min(1, sig_denoised));
    psnr_sig(k) = psnr(sig_denoised, lena_color);
    ssim_sig(k) = ssim(sig_denoised, lena_color);
    fprintf('  sigma=%.1f: PSNR=%.4f dB, SSIM=%.4f\n', sig, psnr_sig(k), ssim_sig(k));
end

%% 绘制调参曲线
figure('Name', '任务3调参: 去噪参数优化', 'NumberTitle', 'off', 'Position', [50, 50, 1200, 500]);
subplot(1, 2, 1);
yyaxis left;
plot(med_window_sizes, psnr_med, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 8);
ylabel('PSNR (dB)');
yyaxis right;
plot(med_window_sizes, ssim_med, 'r-s', 'LineWidth', 1.5, 'MarkerSize', 8);
ylabel('SSIM');
xlabel('中值滤波窗口大小'); title('直接法: 中值滤波窗口 vs 去噪指标');
legend({'PSNR', 'SSIM'}, 'Location', 'best'); grid on;
subplot(1, 2, 2);
yyaxis left;
plot(sigma_values, psnr_sig, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 8);
ylabel('PSNR (dB)');
yyaxis right;
plot(sigma_values, ssim_sig, 'r-s', 'LineWidth', 1.5, 'MarkerSize', 8);
ylabel('SSIM');
xlabel('高斯sigma'); title('间接法: 高斯sigma vs 去噪指标');
legend({'PSNR', 'SSIM'}, 'Location', 'best'); grid on;
sgtitle('任务3: 去噪参数优化', 'FontSize', 14);
saveas(gcf, fullfile(fig_dir, 'exp7_fig6.png'));
% 保存图片
fprintf('图片已保存: exp7_fig6.png\n');

%% 任务3调参: 最优参数汇总
[~, best_med_idx] = max(psnr_med);
best_med_win = med_window_sizes(best_med_idx);
[~, best_sig_idx] = max(psnr_sig);
best_sigma = sigma_values(best_sig_idx);
fprintf('\n★ 任务3最优参数汇总:\n');
fprintf('   直接法(中值滤波): 窗口[%d %d] → PSNR=%.4f dB, SSIM=%.4f\n', ...
    best_med_win, best_med_win, psnr_med(best_med_idx), ssim_med(best_med_idx));
fprintf('   间接法(高斯滤波): sigma=%.1f → PSNR=%.4f dB, SSIM=%.4f\n', ...
    best_sigma, psnr_sig(best_sig_idx), ssim_sig(best_sig_idx));
fprintf('  ──────────────────────────────────────────────\n');
fprintf('  ★ 全局遍历最优直接: %s  PSNR=%.4f, SSIM=%.4f\n', best_direct_name, best_d_psnr, direct_ssim(best_d_idx));
fprintf('  ★ 全局遍历最优间接: %s  PSNR=%.4f, SSIM=%.4f\n', best_indirect_name, best_i_psnr, best_i_ssim);
fprintf('\n分析:\n');
fprintf('  直接方法: 简单直接，但可能引入颜色失真\n');
fprintf('  间接方法: 保持颜色信息，更适合人眼感知\n');
fprintf('  颜色空间法: YCbCr/HSV/illumpca利用颜色分离特性实现自适应去噪\n');
fprintf('  误差图: 蓝色区域表示该处间接法误差更小，红色表示直接法更优\n');

%% 任务3调参增强: YCbCr双边滤波DegreeOfSmoothing扫描
fprintf('\n--- 任务3调参(增强): YCbCr法 双边滤波强度扫描 ---\n');
dof_values = [0.5, 1, 1.5, 2, 3];
psnr_ycbcr_tune = zeros(length(dof_values), 1);
ssim_ycbcr_tune = zeros(length(dof_values), 1);
for k = 1:length(dof_values)
    d = dof_values(k);
    y_tmp = imbilatfilt(ycbcr_work(:,:,1), 'DegreeOfSmoothing', d);
    ycbcr_tmp = ycbcr2rgb(cat(3, y_tmp, cb_gauss, cr_gauss));
    psnr_ycbcr_tune(k) = psnr(ycbcr_tmp, lena_color);
    ssim_ycbcr_tune(k) = ssim(ycbcr_tmp, lena_color);
    fprintf('  DegreeOfSmoothing=%.1f: PSNR=%.4f dB, SSIM=%.4f\n', d, psnr_ycbcr_tune(k), ssim_ycbcr_tune(k));
end

%% 任务3调参增强: HSV-V通道高斯sigma扫描
fprintf('\n--- 任务3调参(增强): HSV法 V通道高斯sigma扫描 ---\n');
hsv_sigma_vals = [0.5, 1, 1.5, 2, 3];
psnr_hsv_tune = zeros(length(hsv_sigma_vals), 1);
ssim_hsv_tune = zeros(length(hsv_sigma_vals), 1);
for k = 1:length(hsv_sigma_vals)
    s = hsv_sigma_vals(k);
    v_tmp = imgaussfilt(hsv_work(:,:,3), s);
    hsv_tmp = hsv2rgb(cat(3, hsv_work(:,:,1), hsv_work(:,:,2), v_tmp));
    psnr_hsv_tune(k) = psnr(hsv_tmp, lena_color);
    ssim_hsv_tune(k) = ssim(hsv_tmp, lena_color);
    fprintf('  sigma=%.1f: PSNR=%.4f dB, SSIM=%.4f\n', s, psnr_hsv_tune(k), ssim_hsv_tune(k));
end

%% 任务3调参增强: illumpca PCA百分比扫描
fprintf('\n--- 任务3调参(增强): illumpca PCA百分比扫描 ---\n');
pct_vals = [1, 2, 3.5, 5, 10];
psnr_illum_tune = zeros(length(pct_vals), 1);
ssim_illum_tune = zeros(length(pct_vals), 1);
for k = 1:length(pct_vals)
    p = pct_vals(k);
    illu_tmp = illumpca(lin_img, p);
    wb_tmp = chromadapt(lena_noisy, illu_tmp);
    wb_y_tmp = rgb2ycbcr(wb_tmp);
    wy = imbilatfilt(wb_y_tmp(:,:,1));
    wc = imgaussfilt(wb_y_tmp(:,:,2), 2);
    wr = imgaussfilt(wb_y_tmp(:,:,3), 2);
    illum_tmp = ycbcr2rgb(cat(3, wy, wc, wr));
    psnr_illum_tune(k) = psnr(illum_tmp, lena_color);
    ssim_illum_tune(k) = ssim(illum_tmp, lena_color);
    fprintf('  PCA_pct=%.1f: PSNR=%.4f dB, SSIM=%.4f\n', p, psnr_illum_tune(k), ssim_illum_tune(k));
end

%% 绘制增强调参曲线
figure('Name', '任务3调参增强: 颜色空间方法参数优化', 'NumberTitle', 'off', 'Position', [50, 50, 1400, 400]);
subplot(1, 3, 1);
yyaxis left; plot(dof_values, psnr_ycbcr_tune, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 8); ylabel('PSNR (dB)');
yyaxis right; plot(dof_values, ssim_ycbcr_tune, 'r-s', 'LineWidth', 1.5, 'MarkerSize', 8); ylabel('SSIM');
xlabel('DegreeOfSmoothing'); title('YCbCr法: 双边滤波强度 vs 指标');
legend({'PSNR','SSIM'},'Location','best'); grid on;
subplot(1, 3, 2);
yyaxis left; plot(hsv_sigma_vals, psnr_hsv_tune, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 8); ylabel('PSNR (dB)');
yyaxis right; plot(hsv_sigma_vals, ssim_hsv_tune, 'r-s', 'LineWidth', 1.5, 'MarkerSize', 8); ylabel('SSIM');
xlabel('高斯sigma'); title('HSV法: V通道滤波强度 vs 指标');
legend({'PSNR','SSIM'},'Location','best'); grid on;
subplot(1, 3, 3);
yyaxis left; plot(pct_vals, psnr_illum_tune, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 8); ylabel('PSNR (dB)');
yyaxis right; plot(pct_vals, ssim_illum_tune, 'r-s', 'LineWidth', 1.5, 'MarkerSize', 8); ylabel('SSIM');
xlabel('PCA百分比(%)'); title('illumpca: PCA百分比 vs 指标');
legend({'PSNR','SSIM'},'Location','best'); grid on;
sgtitle('任务3(增强): 颜色空间方法参数优化', 'FontSize', 14);
saveas(gcf, fullfile(fig_dir, 'exp7_fig7.png'));
% 保存图片
fprintf('图片已保存: exp7_fig7.png');

%% 任务3调参增强: 最优参数汇总
[~, best_y_idx] = max(psnr_ycbcr_tune);
best_dof = dof_values(best_y_idx);
[~, best_h_idx] = max(psnr_hsv_tune);
best_hsv_sigma = hsv_sigma_vals(best_h_idx);
[~, best_i_idx] = max(psnr_illum_tune);
best_pct = pct_vals(best_i_idx);
fprintf('\n★ 任务3(增强)最优参数汇总:\n');
fprintf('   方法0(RGB-medfilt2): 窗口[%d %d] → PSNR=%.4f dB, SSIM=%.4f\n', ...
    best_med_win, best_med_win, psnr_med(best_med_idx), ssim_med(best_med_idx));
fprintf('   方法A(YCbCr自适应): DegreeOfSmoothing=%.1f → PSNR=%.4f dB, SSIM=%.4f\n', ...
    best_dof, psnr_ycbcr_tune(best_y_idx), ssim_ycbcr_tune(best_y_idx));
fprintf('   方法B(HSV-V通道): sigma=%.1f → PSNR=%.4f dB, SSIM=%.4f\n', ...
    best_hsv_sigma, psnr_hsv_tune(best_h_idx), ssim_hsv_tune(best_h_idx));
fprintf('   方法C(illumpca光照感知): PCA_pct=%.1f%% → PSNR=%.4f dB, SSIM=%.4f\n', ...
    best_pct, psnr_illum_tune(best_i_idx), ssim_illum_tune(best_i_idx));
fprintf('   间接方法(Lab-L通道): sigma=%.1f → PSNR=%.4f dB, SSIM=%.4f\n', ...
    best_sigma, psnr_sig(best_sig_idx), ssim_sig(best_sig_idx));
if strcmp(cbm3d_best_space, 'opp')
    fprintf('   方法E(CBM3D协同滤波): colorspace=opp → PSNR=%.4f dB, SSIM=%.4f\n', psnr_cbm3d, ssim_cbm3d);
else
    fprintf('   方法E(CBM3D协同滤波): colorspace=yCbCr → PSNR=%.4f dB, SSIM=%.4f\n', psnr_cbm3d_ycbcr, ssim_cbm3d_ycbcr);
end
fprintf('  ──────────────────────────────────────────────\n');
fprintf('  ★ 全局遍历最优直接: %s  PSNR=%.4f, SSIM=%.4f\n', best_direct_name, best_d_psnr, direct_ssim(best_d_idx));
fprintf('  ★ 全局遍历最优间接: %s  PSNR=%.4f, SSIM=%.4f\n', best_indirect_name, best_i_psnr, best_i_ssim);

%% 任务4: 基于颜色信息的分割
fprintf('\n【任务4】基于颜色信息的黄色花朵分割\n');
fprintf('----------------------------------------\n');
% 读入yellowlily图像
yellowlily_path = 'yellowlily.jpg';
yellowlily = imread(yellowlily_path);
yellowlily_double = im2double(yellowlily);
% 转换到HSV颜色空间
yellowlily_hsv = rgb2hsv(yellowlily_double);
H = yellowlily_hsv(:,:,1);
S = yellowlily_hsv(:,:,2);
V = yellowlily_hsv(:,:,3);
% 黄色在HSV空间的范围
% H: 黄色大约在0.15-0.20之间
yellow_mask = (H > 0.10) & (H < 0.25) & (S > 0.3) & (V > 0.3);
% 形态学处理优化分割结果
yellow_mask = imopen(yellow_mask, strel('disk', 3));
yellow_mask = imclose(yellow_mask, strel('disk', 5));
yellow_mask = imfill(yellow_mask, 'holes');
% 创建分割结果图像
segmented = yellowlily_double .* repmat(yellow_mask, [1, 1, 3]);
% 计算IoU（如果有ground truth标注文件）
gt_path = 'yellowlily_gt.png';
if exist(gt_path, 'file')
    gt_mask = imread(gt_path) > 128;
    if size(gt_mask, 3) > 1
        gt_mask = rgb2gray(gt_mask) > 128;
    end
    intersection = sum(yellow_mask(:) & gt_mask(:));
    union = sum(yellow_mask(:) | gt_mask(:));
    if union > 0
        iou = intersection / union;
    else
        iou = 0;
    end
    fprintf('IoU = %.4f\n', iou);
else
    fprintf('注意: 未找到ground truth标注文件(%s)，跳过IoU计算\n', gt_path);
    fprintf('      如有标注文件，请放置后重新运行以计算IoU分割指标\n');
end
fprintf('颜色分割完成\n');
fprintf('  使用HSV颜色空间的H通道进行阈值分割\n');
fprintf('  黄色范围: H in [0.10, 0.25], S > 0.3, V > 0.3\n');
% 显示结果
figure('Name', '任务4: 黄色花朵分割', 'NumberTitle', 'off', 'Position', [50, 50, 1400, 400]);
subplot(1, 4, 1);
imshow(yellowlily);
title('原图像');
subplot(1, 4, 2);
imshow(H, []);
title('H (色调)');
subplot(1, 4, 3);
imshow(yellow_mask, []);
title('分割掩模');
subplot(1, 4, 4);
imshow(segmented, []);
title('分割结果');

%% 任务4调参: HSV阈值 + 形态学参数扫描
fprintf('\n--- 任务4调参: 参数扫描 ---\n');

%% 获取参考掩模（从文件读取或自动计算）
if exist(gt_path, 'file')
    gt_mask = imread(gt_path) > 128;
    if size(gt_mask, 3) > 1
        gt_mask = rgb2gray(gt_mask) > 128;
    end
    fprintf('  已读取GT标注文件: %s\n', gt_path);
else
    fprintf('  未找到GT标注文件，自动计算参考掩模(Lab b* + Otsu)...\n');
    % 在Lab空间b*通道上用Otsu生成参考掩模
    lab_ref = rgb2lab(yellowlily_double);
    b_star = lab_ref(:, :, 3);     % b*通道：蓝(-)~黄(+)
    % Otsu自适应阈值
    level = graythresh(b_star);
    ref_mask = b_star > level * max(b_star(:));
    % 形态学清理
    ref_mask = imopen(ref_mask, strel('disk', 3));
    ref_mask = imclose(ref_mask, strel('disk', 5));
    ref_mask = imfill(ref_mask, 'holes');
    gt_mask = ref_mask;
    fprintf('  Otsu阈值 level=%.4f, 参考掩模已生成\n', level);
end

%% 扫描H下限
fprintf('\n扫描H下限 (固定H上限=0.25, S=0.3, V=0.3):\n');
H_low_vals = 0.05:0.02:0.15;
iou_h = zeros(size(H_low_vals));
for k = 1:length(H_low_vals)
    mask = (H > H_low_vals(k)) & (H < 0.25) & (S > 0.3) & (V > 0.3);
    mask = imopen(mask, strel('disk', 3));
    mask = imclose(mask, strel('disk', 5));
    mask = imfill(mask, 'holes');
    inter = sum(mask(:) & gt_mask(:));
    uni   = sum(mask(:) | gt_mask(:));
    iou_h(k) = inter / max(uni, 1);
    fprintf('  H_low=%.2f: IoU=%.4f\n', H_low_vals(k), iou_h(k));
end
[~, best_h] = max(iou_h);
best_H_low = H_low_vals(best_h);

%% 扫描H上限
fprintf('\n扫描H上限 (固定H下限=%.2f, S=0.3, V=0.3):\n', best_H_low);
H_high_vals = 0.18:0.02:0.35;
iou_hh = zeros(size(H_high_vals));
for k = 1:length(H_high_vals)
    mask = (H > best_H_low) & (H < H_high_vals(k)) & (S > 0.3) & (V > 0.3);
    mask = imopen(mask, strel('disk', 3));
    mask = imclose(mask, strel('disk', 5));
    mask = imfill(mask, 'holes');
    inter = sum(mask(:) & gt_mask(:));
    uni   = sum(mask(:) | gt_mask(:));
    iou_hh(k) = inter / max(uni, 1);
    fprintf('  H_high=%.2f: IoU=%.4f\n', H_high_vals(k), iou_hh(k));
end
[~, best_hh] = max(iou_hh);
best_H_high = H_high_vals(best_hh);
best_H_low  = H_low_vals(best_h);
fprintf('  ★ 最优H范围: [%.2f, %.2f] (IoU=%.4f)\n', best_H_low, best_H_high, max(iou_hh));

%% 扫描S阈值
fprintf('\n扫描S阈值 (固定H=[%.2f, %.2f], V=0.3):\n', best_H_low, best_H_high);
S_vals = 0.1:0.1:0.6;
iou_s = zeros(size(S_vals));
for k = 1:length(S_vals)
    mask = (H > best_H_low) & (H < best_H_high) & (S > S_vals(k)) & (V > 0.3);
    mask = imopen(mask, strel('disk', 3));
    mask = imclose(mask, strel('disk', 5));
    mask = imfill(mask, 'holes');
    inter = sum(mask(:) & gt_mask(:));
    uni   = sum(mask(:) | gt_mask(:));
    iou_s(k) = inter / max(uni, 1);
    fprintf('  S=%.1f: IoU=%.4f\n', S_vals(k), iou_s(k));
end
[~, best_s] = max(iou_s);
best_S = S_vals(best_s);

%% 扫描V阈值
fprintf('\n扫描V阈值 (固定H=[%.2f, %.2f], S=%.1f):\n', best_H_low, best_H_high, best_S);
V_vals = 0.1:0.1:0.6;
iou_v = zeros(size(V_vals));
for k = 1:length(V_vals)
    mask = (H > best_H_low) & (H < best_H_high) & (S > best_S) & (V > V_vals(k));
    mask = imopen(mask, strel('disk', 3));
    mask = imclose(mask, strel('disk', 5));
    mask = imfill(mask, 'holes');
    inter = sum(mask(:) & gt_mask(:));
    uni   = sum(mask(:) | gt_mask(:));
    iou_v(k) = inter / max(uni, 1);
    fprintf('  V=%.1f: IoU=%.4f\n', V_vals(k), iou_v(k));
end
[~, best_v] = max(iou_v);
best_V = V_vals(best_v);

%% 扫描开运算disk半径
fprintf('\n扫描开运算disk半径 (闭运算disk=5):\n');
open_vals = [1, 2, 3, 5, 7];
iou_open = zeros(size(open_vals));
for k = 1:length(open_vals)
    mask = (H > best_H_low) & (H < best_H_high) & (S > best_S) & (V > best_V);
    mask = imopen(mask, strel('disk', open_vals(k)));
    mask = imclose(mask, strel('disk', 5));
    mask = imfill(mask, 'holes');
    inter = sum(mask(:) & gt_mask(:));
    uni   = sum(mask(:) | gt_mask(:));
    iou_open(k) = inter / max(uni, 1);
    fprintf('  open_disk=%d: IoU=%.4f\n', open_vals(k), iou_open(k));
end
[~, best_open] = max(iou_open);
best_open_r = open_vals(best_open);

%% 扫描闭运算disk半径
fprintf('\n扫描闭运算disk半径 (开运算disk=%d):\n', best_open_r);
close_vals = [1, 3, 5, 7, 10];
iou_close = zeros(size(close_vals));
for k = 1:length(close_vals)
    mask = (H > best_H_low) & (H < best_H_high) & (S > best_S) & (V > best_V);
    mask = imopen(mask, strel('disk', best_open_r));
    mask = imclose(mask, strel('disk', close_vals(k)));
    mask = imfill(mask, 'holes');
    inter = sum(mask(:) & gt_mask(:));
    uni   = sum(mask(:) | gt_mask(:));
    iou_close(k) = inter / max(uni, 1);
    fprintf('  close_disk=%d: IoU=%.4f\n', close_vals(k), iou_close(k));
end
[~, best_close] = max(iou_close);
best_close_r = close_vals(best_close);

%% 汇总最优参数
fprintf('\n★ 任务4最优参数汇总:\n');
fprintf('   H范围: [%.2f, %.2f]   S阈值: %.1f   V阈值: %.1f\n', ...
    best_H_low, best_H_high, best_S, best_V);
fprintf('   开运算disk=%d   闭运算disk=%d\n', best_open_r, best_close_r);

%% 用最优参数重新生成最终分割结果
final_mask = (H > best_H_low) & (H < best_H_high) & (S > best_S) & (V > best_V);
final_mask = imopen(final_mask, strel('disk', best_open_r));
final_mask = imclose(final_mask, strel('disk', best_close_r));
final_mask = imfill(final_mask, 'holes');
final_seg = yellowlily_double .* repmat(final_mask, [1, 1, 3]);
final_iou = sum(final_mask(:) & gt_mask(:)) / max(sum(final_mask(:) | gt_mask(:)), 1);
fprintf('   ★ 最优分割IoU: %.4f\n', final_iou);
figure('Name', '任务4调参: 最优分割结果', 'NumberTitle', 'off', 'Position', [50, 50, 1200, 400]);
subplot(1, 3, 1); imshow(yellowlily); title('原图像');
subplot(1, 3, 2); imshow(final_mask); title(sprintf('最优分割掩模 (IoU=%.4f)', final_iou));
subplot(1, 3, 3); imshow(final_seg);  title('最优分割结果');

%% HSV + Lab 双空间融合分割（方案A）
fprintf('\n--- HSV + Lab 双空间融合分割 ---\n');
% 在Lab b*通道上做Otsu自适应分割（黄色在b*有强正响应）
lab_yellowlily = rgb2lab(yellowlily_double);
b_star = lab_yellowlily(:,:,3);
b_level = graythresh(b_star);
lab_mask_raw = b_star > b_level * max(b_star(:));
% 形态学清理
lab_mask = imopen(lab_mask_raw, strel('disk', best_open_r));
lab_mask = imclose(lab_mask, strel('disk', best_close_r));
lab_mask = imfill(lab_mask, 'holes');
% IoU
lab_inter = sum(lab_mask(:) & gt_mask(:));
lab_union = sum(lab_mask(:) | gt_mask(:));
lab_iou = lab_inter / max(lab_union, 1);
fprintf('  Lab b* + Otsu: IoU=%.4f\n', lab_iou);
% 融合策略1: 交集
fusion_inter = final_mask & lab_mask;
fi_inter = sum(fusion_inter(:) & gt_mask(:));
fu_inter = sum(fusion_inter(:) | gt_mask(:));
iou_inter = fi_inter / max(fu_inter, 1);
fprintf('  HSV & Lab(交集): IoU=%.4f\n', iou_inter);
% 融合策略2: 并集
fusion_union = final_mask | lab_mask;
fi_union = sum(fusion_union(:) & gt_mask(:));
fu_union = sum(fusion_union(:) | gt_mask(:));
iou_union = fi_union / max(fu_union, 1);
fprintf('  HSV | Lab(并集): IoU=%.4f\n', iou_union);
% 选出最优融合策略
iou_all = [final_iou, lab_iou, iou_inter, iou_union];
fusion_names = {'HSV最优', 'Lab(b*)Otsu', 'HSV∩Lab', 'HSV∪Lab'};
[best_fusion_iou, best_fusion_idx] = max(iou_all);
fprintf('  ★ 最优分割: %s (IoU=%.4f)\n', fusion_names{best_fusion_idx}, best_fusion_iou);
% 用最优融合策略生成结果
switch best_fusion_idx
    case 1, best_fusion_mask = final_mask;
    case 2, best_fusion_mask = lab_mask;
    case 3, best_fusion_mask = fusion_inter;
    case 4, best_fusion_mask = fusion_union;
end
best_fusion_seg = yellowlily_double .* repmat(best_fusion_mask, [1, 1, 3]);
% 显示四种策略对比
figure('Name', '任务4: HSV+Lab双空间分割对比', 'NumberTitle', 'off', 'Position', [50, 50, 1400, 500]);
for i = 1:4
    subplot(2, 4, i);
    switch i
        case 1, seg_mask = final_mask;     seg_name = fusion_names{1};
        case 2, seg_mask = lab_mask;       seg_name = fusion_names{2};
        case 3, seg_mask = fusion_inter;   seg_name = fusion_names{3};
        case 4, seg_mask = fusion_union;   seg_name = fusion_names{4};
    end
    imshow(seg_mask); title(sprintf('%s\nIoU=%.4f', seg_name, iou_all(i)));
    subplot(2, 4, i+4);
    seg_img = yellowlily_double .* repmat(seg_mask, [1, 1, 3]);
    imshow(seg_img); title('分割结果');
end
sgtitle('HSV+Lab双空间分割策略对比', 'FontSize', 14);
saveas(gcf, fullfile(fig_dir, 'exp7_fig10.png'));
% 保存图片
fprintf('图片已保存: exp7_fig10.png');

%% 绘制调参曲线
figure('Name', '任务4调参: 参数优化曲线', 'NumberTitle', 'off', 'Position', [50, 50, 1400, 700]);
subplot(2, 3, 1);
plot(H_low_vals, iou_h, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 8); grid on;
xlabel('H下限'); ylabel('IoU'); title('H下限 vs IoU');
subplot(2, 3, 2);
plot(H_high_vals, iou_hh, 'r-s', 'LineWidth', 1.5, 'MarkerSize', 8); grid on;
xlabel('H上限'); ylabel('IoU'); title('H上限 vs IoU');
subplot(2, 3, 3);
plot(S_vals, iou_s, 'g-^', 'LineWidth', 1.5, 'MarkerSize', 8); grid on;
xlabel('S阈值'); ylabel('IoU'); title('S阈值 vs IoU');
subplot(2, 3, 4);
plot(V_vals, iou_v, 'm-d', 'LineWidth', 1.5, 'MarkerSize', 8); grid on;
xlabel('V阈值'); ylabel('IoU'); title('V阈值 vs IoU');
subplot(2, 3, 5);
bar(open_vals, iou_open, 0.5, 'FaceColor', [0.5 0.7 1]); grid on;
xlabel('开运算disk半径'); ylabel('IoU'); title('开运算半径 vs IoU');
subplot(2, 3, 6);
bar(close_vals, iou_close, 0.5, 'FaceColor', [0.6 0.9 0.6]); grid on;
xlabel('闭运算disk半径'); ylabel('IoU'); title('闭运算半径 vs IoU');
sgtitle('任务4: 分割参数优化', 'FontSize', 14);
saveas(gcf, fullfile(fig_dir, 'exp7_fig11.png'));
% 保存图片
fprintf('图片已保存: exp7_fig11.png');

%% 结果分析
fprintf('\n【分析】\n');
fprintf('1. YCbCr颜色空间: Y通道包含主要亮度信息，适合用于亮度处理\n');
fprintf('2. Lab颜色空间: L通道与人眼感知一致，a、b通道表示颜色信息\n');
fprintf('3. 彩色去噪(直接vs间接):\n');
fprintf('   ★ 直接方法遍历6种去噪，最优: %s (PSNR=%.4f, SSIM=%.4f)\n', best_direct_name, best_d_psnr, direct_ssim(best_d_idx));
fprintf('   ★ 间接方法遍历3种颜色空间×6种去噪，最优: %s (PSNR=%.4f, SSIM=%.4f)\n', best_indirect_name, best_i_psnr, best_i_ssim);
fprintf('   直接方法简单直接，但R/G/B三通道独立处理可能引入颜色失真;\n');
fprintf('   间接方法在亮度通道去噪、色度通道保持不变，颜色保真度更高;\n');
fprintf('   通常间接方法PSNR/SSIM优于直接方法，原因是人眼对亮度更敏感\n');
fprintf('4. YCbCr自适应法中Y通道双边滤波保边、Cb/Cr强去噪;\n');
fprintf('   HSV-V通道法利用颜色/强度分离; illumpca光照校正后去噪效果更稳定\n');
fprintf('5. CBM3D协同滤波: 将RGB转换到对色空间(opponent)，亮度通道做块匹配\n');
fprintf('   色度复用匹配分组信息，利用3D变换域协同滤波去噪\n');
fprintf('   作为非局部自相似性方法的代表，PSNR/SSIM均为最优或接近最优\n');
fprintf('6. 颜色分割: HSV颜色空间更适合基于颜色的分割任务;\n');
fprintf('   Lab b*通道对黄色有天然物理对应关系，Otsu自适应阈值无需手动调参;\n');
fprintf('   HSV+Lab双空间融合(交集/并集)可进一步提升分割鲁棒性\n');
fprintf('\n========================================\n');
fprintf('        实验7 完成!\n');
fprintf('========================================\n');

%% ===== 局部函数 =====
function ch = denoise_channel(ch, method)
    % 统一的单通道去噪接口
    switch method
        case 'medfilt2',   ch = medfilt2(ch, [5 5]);
        case 'gaussian',   ch = imgaussfilt(ch, 1.5);
        case 'bilateral',  ch = imbilatfilt(ch);
        case 'wiener',     ch = wiener2(ch, [5 5]);
        case 'nlm',        ch = imnlmfilt(ch);
        case 'diffusion',  ch = imdiffusefilt(ch);
        otherwise, error('未知去噪方法: %s', method);
    end
end