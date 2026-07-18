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

%% 
%% 实验5: 图像复原与重建
% 课程: 视觉与数据计算
% 重点函数: deconvwnr, deconvreg, deconvlucy, deconvblind; radon, iradon, phantom
clear all;
close all;
clc;
% 创建图片保存目录
fig_dir = fullfile(fileparts(mfilename('fullpath')), 'figs');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

%% 参数设置（集中管理，课堂演示时修改此结构体即可）
% ============================================================
% 所有可调参数集中定义，便于课堂展示时快速调整
% 参数调优: 运行 tune_parameters_v2.m 自动搜索最优参数
% ============================================================
params = struct();
% -- 任务1: 高斯模糊 + 高斯噪声退化 --
params.sigma_noise = 10;             % 噪声标准差（uint8尺度 0-255）
params.reg_np = [];                  % deconvreg NP（空=根据图像尺寸自动计算）
params.lucy_iter = 4;                % Lucy-Richardson 迭代次数
params.lucy_damp = 3*(10/255);       % DAMPAR 阻尼阈值（3×噪声std ≈ 0.1176）
% -- 任务2: 运动模糊 + 盲去卷积 --
params.blind_iter = 20;              % 盲去卷积迭代次数
params.motion_len = 10;              % 初始PSF运动长度（像素）
params.motion_theta = 150;           % 初始PSF运动角度（度，从水平逆时针）
params.blind_dampar = [];            % 盲去卷积 DAMPAR（空=根据噪声自动）
% -- 拓展分析参数 --
params.region_h = 64;                % 子区域PSF估计: 区域高度
params.region_w = 64;                % 子区域PSF估计: 区域宽度
params.n_trials = 100;               % 子区域搜索: 随机尝试次数
% 加载调优参数（若存在则覆盖默认值）
if exist('optimal_params_ex5_v2.mat', 'file')
    S = load('optimal_params_ex5_v2.mat');
    if isfield(S, 'optimal_reg_v2'),         params.reg_np       = S.optimal_reg_v2;         end
    if isfield(S, 'optimal_lucy_iter_v2'),   params.lucy_iter    = S.optimal_lucy_iter_v2;   end
    if isfield(S, 'optimal_lucy_damp_v2'),   params.lucy_damp    = S.optimal_lucy_damp_v2;   end
    if isfield(S, 'optimal_blind_iter_v2'),  params.blind_iter   = S.optimal_blind_iter_v2;  end
    if isfield(S, 'optimal_motion_len_v2'),  params.motion_len   = S.optimal_motion_len_v2;  end
    if isfield(S, 'optimal_motion_theta_v2'),params.motion_theta = S.optimal_motion_theta_v2;end
    if isfield(S, 'optimal_blind_dampar_v2'),params.blind_dampar = S.optimal_blind_dampar_v2;end
    param_src = '自动调优参数';
else
    param_src = '默认参数';
end
% 提取到局部变量（便于后续直接使用，避免 params. 前缀）
sigma_noise = params.sigma_noise;
reg_param   = params.reg_np;
lucy_iter   = params.lucy_iter;
lucy_damp   = params.lucy_damp;
blind_iter  = params.blind_iter;
motion_len  = params.motion_len;
motion_theta = params.motion_theta;
blind_dampar = params.blind_dampar;
fprintf('========================================\n');
fprintf('       实验5: 图像复原与重建 [%s]\n', param_src);
fprintf('========================================\n\n');
reg_str = 'auto'; if ~isempty(reg_param), reg_str = sprintf('%.1f',reg_param); end
fprintf('  sigma_noise=%d, reg_np=%s, lucy_iter=%d, lucy_damp=%.4f\n', ...
    sigma_noise, reg_str, lucy_iter, lucy_damp);
fprintf('  blind_iter=%d, motion_len=%d, motion_theta=%.1f', ...
    blind_iter, motion_len, motion_theta);
if ~isempty(blind_dampar)
    fprintf(', blind_dampar=%.4f\n\n', blind_dampar);
else
    fprintf('\n\n');
end
% 初始化扩展结果记录（供汇总表格使用）
extension_results = {};

%% 任务1: 高斯模糊+噪声退化图像复原
fprintf('【任务1】高斯模糊+噪声退化图像复原\n');
fprintf('----------------------------------------\n');
% 读入图像
img_path = 'lena.png';
img = imread(img_path);
if size(img, 3) == 3
    img_gray = rgb2gray(img);
else
    img_gray = img;
end
img_gray = im2double(img_gray);
% 创建高斯模糊核（5×5，偏差1.5）
PSF_gaussian = fspecial('gaussian', [5 5], 1.5);
PSF_gaussian = PSF_gaussian / sum(PSF_gaussian(:));  % 显式归一化

%% 显示高斯模糊核（用于课堂讲解"点扩散函数"概念）
figure('Name', '任务1: 高斯模糊核 PSF', 'NumberTitle', 'off', 'Position', [300, 300, 700, 300]);
subplot(1, 2, 1);
surf(PSF_gaussian); title({'高斯模糊核 PSF', '(5×5, \sigma=1.5)'});
xlabel('x'); ylabel('y'); zlabel('幅值'); colormap(gca, 'jet');
subplot(1, 2, 2);
imshow(PSF_gaussian, []); title('PSF (俯视图)'); colorbar;
% 高斯模糊
img_blurred = imfilter(img_gray, PSF_gaussian, 'conv', 'replicate');
% 添加高斯噪声
rng(0);
img_degraded = img_blurred + (sigma_noise/255) * randn(size(img_blurred));
img_degraded = max(0, min(1, img_degraded));
fprintf('退化参数:\n');
fprintf('  模糊核: 5×5 高斯，偏差=1.5\n');
fprintf('  噪声: 高斯噪声，偏差=%d\n', sigma_noise);
% 计算 deconvreg NP 参数 = 总噪声能量 ||n||² = M×N×(σ/255)²
% 若调优文件未提供 NP，则根据实际图像尺寸动态计算
if isempty(reg_param)
    reg_param = (sigma_noise/255)^2 * numel(img_gray);
    fprintf('  NP (= ||n||²) = σ² × numel = %.4f × %d = %.2f\n', ...
        (sigma_noise/255)^2, numel(img_gray), reg_param);
end
% 方法1: 维纳滤波 (deconvwnr)
% NSR = σ²_noise / σ²_signal（噪声功率 / 信号功率）
% 注: 标量 NSR 假设噪声和信号的功率谱密度在整个频带恒定，
%     对自然图像(1/f²谱)仅为近似最优，性能低于 deconvreg
NSR = (sigma_noise/255)^2 / var(img_gray(:));
img_degraded_tapered = edgetaper(img_degraded, PSF_gaussian);
img_wnr = deconvwnr(img_degraded_tapered, PSF_gaussian, NSR);
img_wnr = max(0, min(1, img_wnr));
% 方法2: 约束最小二乘滤波 (deconvreg) — L2正则化，高斯噪声最优
img_reg = deconvreg(img_degraded_tapered, PSF_gaussian, reg_param);
img_reg = max(0, min(1, img_reg));
% 方法3: Lucy-Richardson迭代 (deconvlucy) — 用 edgetaper 抑制振铃
% L-R 算法基于泊松噪声模型的极大似然推导（MLE），对高斯噪声鲁棒性弱。
% 当前参数下效果与退化图像几乎无差异（PSNR≈26.40 dB，仅提升0.06 dB），原因:
%   (1) 噪声模型不匹配: L-R 迭代公式源自泊松 MLE，而本实验为高斯噪声，
%       泊松似然在高斯假设下非最优，导致收敛路径偏离真实解
%   (2) 迭代次数不足: L-R 本质为 EM 迭代，泊松数据的收敛速度取决于噪声强度；
%       对高斯噪声，似然面形状不同，默认迭代步数（lucy_iter=4）远不足以收敛
%   (3) DAMPAR 将像素级变化控制在3σ内，但迭代不充分时噪声仍残留
%   (4) 高斯模糊+高斯噪声场景中应优先选用 Wiener 或 deconvreg，
%       L-R 更适合光子计数模式（天文、显微）的泊松退化场景
img_lucy = deconvlucy(img_degraded_tapered, PSF_gaussian, lucy_iter, lucy_damp);
img_lucy = max(0, min(1, img_lucy));
% 计算质量指标
psnr_degraded = psnr(img_degraded, img_gray);
ssim_degraded = ssim(img_degraded, img_gray);
psnr_wnr = psnr(img_wnr, img_gray);
ssim_wnr = ssim(img_wnr, img_gray);
psnr_reg = psnr(img_reg, img_gray);
ssim_reg = ssim(img_reg, img_gray);
psnr_lucy = psnr(img_lucy, img_gray);
ssim_lucy = ssim(img_lucy, img_gray);
fprintf('\n复原结果:\n');
fprintf('  退化图像: PSNR=%.4f dB, SSIM=%.4f\n', psnr_degraded, ssim_degraded);
fprintf('  维纳滤波: PSNR=%.4f dB, SSIM=%.4f\n', psnr_wnr, ssim_wnr);
fprintf('  约束最小二乘: PSNR=%.4f dB, SSIM=%.4f\n', psnr_reg, ssim_reg);
fprintf('  Lucy-Richardson: PSNR=%.4f dB, SSIM=%.4f\n', psnr_lucy, ssim_lucy);
% 显示结果
figure('Name', '任务1: 图像复原方法比较', 'NumberTitle', 'off', 'Position', [50, 50, 1200, 900]);
subplot(2, 2, 1);
imshow(img_degraded, []);
title({sprintf('退化图像'), sprintf('PSNR=%.4f, SSIM=%.4f', psnr_degraded, ssim_degraded)});
subplot(2, 2, 2);
imshow(img_wnr, []);
title({sprintf('维纳滤波'), sprintf('PSNR=%.4f, SSIM=%.4f', psnr_wnr, ssim_wnr)});
subplot(2, 2, 3);
imshow(img_reg, []);
title({sprintf('约束最小二乘'), sprintf('PSNR=%.4f, SSIM=%.4f', psnr_reg, ssim_reg)});
subplot(2, 2, 4);
imshow(img_lucy, []);
title({sprintf('Lucy-Richardson'), sprintf('PSNR=%.4f, SSIM=%.4f', psnr_lucy, ssim_lucy)});
saveas(gcf, fullfile(fig_dir, 'exp5_task1_restore.png'));
% 保存图片
fprintf('图片已保存: exp5_task1_restore.png\n');

%% 误差分析: 差异图（各方法误差空间分布对比）
% 差异图显示每个像素的 |复原结果 - 原图| 误差，颜色越亮误差越大
% 可直观看出每种方法在哪些区域失效（边缘/纹理/平坦区域）
figure('Name', '任务1: 复原误差对比', 'NumberTitle', 'off', 'Position', [50, 50, 1200, 900]);
diff_degraded = abs(img_degraded - img_gray);
diff_wnr      = abs(img_wnr - img_gray);
diff_reg      = abs(img_reg - img_gray);
diff_lucy     = abs(img_lucy - img_gray);
subplot(2, 2, 1); imshow(diff_degraded, []);
title({sprintf('退化误差'), sprintf('MAE=%.4f', mean(diff_degraded(:)))}); colorbar;
subplot(2, 2, 2); imshow(diff_wnr, []);
title({sprintf('维纳滤波误差'), sprintf('MAE=%.4f', mean(diff_wnr(:)))}); colorbar;
subplot(2, 2, 3); imshow(diff_reg, []);
title({sprintf('约束最小二乘误差'), sprintf('MAE=%.4f', mean(diff_reg(:)))}); colorbar;
subplot(2, 2, 4); imshow(diff_lucy, []);
title({sprintf('L-R误差'), sprintf('MAE=%.4f', mean(diff_lucy(:)))}); colorbar;
sgtitle('差异图: |复原结果 - 原始图像|（越暗=误差越小）');
saveas(gcf, fullfile(fig_dir, 'exp5_fig3.png'));
% 保存图片
fprintf('图片已保存: exp5_fig3.png');

%% 行截面强度对比（沿图像中间行，直观展示边缘保持 vs 噪声平滑的权衡）
row_idx = round(size(img_gray, 1) / 2);  % 取中间行
figure('Name', '任务1: 行截面强度对比', 'NumberTitle', 'off', 'Position', [100, 100, 900, 400]);
plot(img_gray(row_idx, :), 'k-', 'LineWidth', 2); hold on;
plot(img_degraded(row_idx, :), 'c--', 'LineWidth', 1);
plot(img_wnr(row_idx, :), 'b-.', 'LineWidth', 1);
plot(img_reg(row_idx, :), 'r-', 'LineWidth', 1);
plot(img_lucy(row_idx, :), 'g:', 'LineWidth', 1);
xlabel('列坐标'); ylabel('像素强度');
title(sprintf('第 %d 行截面强度对比', row_idx));
legend({'原始', '退化', '维纳滤波', '约束最小二乘', 'L-R'}, 'Location', 'best');
grid on;

%% 任务2: 运动模糊+噪声退化图像盲去卷积
fprintf('\n【任务2】运动模糊+噪声退化图像盲去卷积\n');
fprintf('----------------------------------------\n');
% 读入运动模糊+噪声退化图像
cameraman_bn_path = 'cameraman_b_n.png';
img_bn = imread(cameraman_bn_path);
if size(img_bn, 3) == 3
    img_bn = rgb2gray(img_bn);
end
img_bn = im2double(img_bn);
% 读入原始图像（用于比较）
cameraman_path = 'cameraman.png';
img_original = imread(cameraman_path);
if size(img_original, 3) == 3
    img_original = rgb2gray(img_original);
end
img_original = im2double(img_original);
% 估计退化图像中的噪声水平（用左上角平滑区域）
noise_region = img_bn(1:30, 1:30);
noise_std_est = std(noise_region(:));
fprintf('  退化图像噪声估计: σ≈%.2f (uint8: %.0f)\n', noise_std_est, noise_std_est*255);
% 若调优文件未提供 blin d_dampar，则根据实际噪声水平计算
if isempty(blind_dampar)
    
    blind_dampar = 8 * noise_std_est;  % DAMPAR=8σ: 强阻尼抑制噪声对PSF估计的干扰
end
% 盲去卷积
% 初始PSF使用运动模糊核（线状）而非均匀方核
% 注: 盲去卷积严重依赖初始PSF的形状:
%   - 均匀方核 → 算法收敛到圆形/高斯状 PSF → 复原完全失败
%   - 线状运动核 → 即使长度/角度不精确，算法也能迭代修正 → 正确收敛
% motion_len/motion_theta 可通过观测退化图像边缘拖影方向来估计
% DAMPAR = 8σ: 强阻尼抑制噪声对PSF估计的干扰
%   盲去卷积内部用Lucy-Richardson迭代估计图像，对高斯噪声鲁棒性弱，
%   高DAMPAR防止噪声在交替迭代中被放大并带偏PSF估计方向
% edgetaper: 抑制边界伪影对迭代过程的影响
INITPSF = fspecial('motion', motion_len, motion_theta);
INITPSF = INITPSF / sum(INITPSF(:));
img_bn_taper = edgetaper(img_bn, INITPSF);
[img_blind, PSF_blind] = deconvblind(img_bn_taper, INITPSF, blind_iter, blind_dampar);
img_blind = max(0, min(1, img_blind));
% 计算质量指标
psnr_bn = psnr(img_bn, img_original);
ssim_bn = ssim(img_bn, img_original);
psnr_blind = psnr(img_blind, img_original);
ssim_blind = ssim(img_blind, img_original);
fprintf('盲去卷积结果:\n');
fprintf('  退化图像: PSNR=%.4f dB, SSIM=%.4f\n', psnr_bn, ssim_bn);
fprintf('  盲去卷积: PSNR=%.4f dB, SSIM=%.4f\n', psnr_blind, ssim_blind);
% 显示结果（实验要求: 2行2列窗口）
figure('Name', '任务2: 盲去卷积复原', 'NumberTitle', 'off', 'Position', [50, 50, 1200, 900]);
subplot(2, 2, 1);
imshow(img_original, []);
title('原始图像');
subplot(2, 2, 2);
imshow(img_bn, []);
title({sprintf('退化图像'), sprintf('PSNR=%.4f, SSIM=%.4f', psnr_bn, ssim_bn)});
subplot(2, 2, 3);
imshow(img_blind, []);
title({sprintf('盲去卷积'), sprintf('PSNR=%.4f, SSIM=%.4f', psnr_blind, ssim_blind)});
subplot(2, 2, 4);
imshow(PSF_blind, []);
title('估计的点扩散函数(PSF)');
colorbar;

%% 初始PSF vs 估计PSF对比（盲去卷积核估计效果评估）
figure('Name', '任务2: PSF估计对比', 'NumberTitle', 'off', 'Position', [300, 300, 900, 300]);
subplot(1, 3, 1);
imshow(INITPSF, []); title({'初始PSF（运动模糊核）', ...
    sprintf('len=%d, theta=%.0f°', motion_len, motion_theta)}); colorbar;
subplot(1, 3, 2);
imshow(PSF_blind, []); title(sprintf('估计PSF [%d×%d]', size(PSF_blind,1), size(PSF_blind,2))); colorbar;
subplot(1, 3, 3);
surf(PSF_blind); title('估计PSF (3D视图)'); xlabel('x'); ylabel('y'); zlabel('幅值');

%% 【拓展对比】子区域PSF估计（区域选择）
% Fergus (2006) 建议人眼选择无饱和、纹理丰富的区域进行核估计，
% 减少噪声对PSF估计的干扰。这里自动寻找梯度方差最大的64×64子区域。
region_h = 64; region_w = 64;
n_trials = 100;
best_score = -inf;
best_rect = [1, 1];
rng(0);  % 固定随机种子保证可重复
for i = 1:n_trials
    r = randi([1, size(img_bn, 1) - region_h]);
    c = randi([1, size(img_bn, 2) - region_w]);
    region = img_bn(r:r+region_h-1, c:c+region_w-1);
    % 评分: 标准差/均值比（高=纹理丰富+低噪声影响）
    score = std(region(:)) / (mean(region(:)) + eps);
    if score > best_score
        best_score = score;
        best_rect = [r, c];
    end
end
% 用最佳子区域估计PSF
img_sub = img_bn(best_rect(1):best_rect(1)+region_h-1, ...
                  best_rect(2):best_rect(2)+region_w-1);
INITPSF_sub = fspecial('motion', motion_len, motion_theta);
INITPSF_sub = INITPSF_sub / sum(INITPSF_sub(:));
[~, PSF_sub] = deconvblind(edgetaper(img_sub, INITPSF_sub), ...
    INITPSF_sub, blind_iter, blind_dampar);
% 用子区域PSF在全图上做非盲复原
NP_region = (noise_std_est)^2 * numel(img_bn);
img_region = deconvreg(edgetaper(img_bn, PSF_sub), PSF_sub, NP_region);
img_region = max(0, min(1, img_region));
psnr_region = psnr(img_region, img_original);
ssim_region = ssim(img_region, img_original);
fprintf('\n【拓展对比】子区域PSF估计 (区域=[%d,%d], 尺寸=%d×%d):\n', ...
    best_rect(1), best_rect(2), region_h, region_w);
fprintf('  PSNR=%.4f dB, SSIM=%.4f (vs 全图deconvblind: %.4f dB, %.4f)\n', ...
    psnr_region, ssim_region, psnr_blind, ssim_blind);
extension_results(end+1, :) = {'子区域PSF+deconvreg', psnr_region, ssim_region};

%% 【拓展对比】混合方法: 盲估计PSF + deconvreg非盲复原
% Fergus (2006) 的流程: 先估计PSF（用VB边缘化），再用RL做非盲复原。
% 但本实验噪声为高斯型，RL（泊松MLE）非最优。改用deconvreg（L2正则化、
% 高斯噪声最优）做非盲复原，应获得更好效果。
% 注: 这也对应 Levin et al. (2009) 的建议——分离核估计与图像复原。
% 混合法正则化参数（基于噪声估计）
hybrid_NP = (noise_std_est)^2 * numel(img_bn);
img_hybrid = deconvreg(edgetaper(img_bn, PSF_blind), PSF_blind, hybrid_NP);
img_hybrid = max(0, min(1, img_hybrid));
psnr_hybrid = psnr(img_hybrid, img_original);
ssim_hybrid = ssim(img_hybrid, img_original);
fprintf('\n【拓展对比】混合方法 (deconvblind估PSF + deconvreg复原):\n');
fprintf('  PSNR=%.4f dB, SSIM=%.4f (vs 纯deconvblind: %.4f dB, %.4f)\n', ...
    psnr_hybrid, ssim_hybrid, psnr_blind, ssim_blind);
extension_results(end+1, :) = {'混合: blind+deconvreg', psnr_hybrid, ssim_hybrid};
% 对比图: deconvblind vs 混合方法
figure('Name', '拓展: 盲去卷积 vs 混合方法', 'NumberTitle', 'off', 'Position', [50, 50, 1200, 400]);
subplot(1, 3, 1);
imshow(img_blind, []);
title({sprintf('纯deconvblind'), sprintf('PSNR=%.4f, SSIM=%.4f', psnr_blind, ssim_blind)});
subplot(1, 3, 2);
imshow(img_hybrid, []);
title({sprintf('混合: deconvblind→deconvreg'), sprintf('PSNR=%.4f, SSIM=%.4f', psnr_hybrid, ssim_hybrid)});
subplot(1, 3, 3);
% 显示PSF与混合法结果的局部放大
crop_rect = [80, 100, 79, 79];  % 三脚架区域
r_start = crop_rect(2); r_end = crop_rect(2) + crop_rect(4);
c_start = crop_rect(1); c_end = crop_rect(1) + crop_rect(3);
imshow([img_bn(r_start:r_end, c_start:c_end), ...
        img_blind(r_start:r_end, c_start:c_end), ...
        img_hybrid(r_start:r_end, c_start:c_end)], []);
title('局部放大: 退化 | deconvblind | 混合');
saveas(gcf, fullfile(fig_dir, 'exp5_fig_blind_compare.png'));

%% 【拓展对比】多尺度盲去卷积 (Coarse-to-Fine)
% 参考 Fergus et al. (2006) 的多尺度策略:
%   粗尺度(1/4) → 核尺寸小→易估计 → 上采样初始化中间层
%   中间尺度(1/2) → 进一步细化 → 上采样初始化精细层
%   精细尺度(1/1) → 最终精调
% 多尺度绕过局部最优，降低对初始PSF参数的敏感性
n_scales = 3;
[img_multiscale, PSF_multiscale] = multi_scale_deconvblind(...
    img_bn, INITPSF, noise_std_est, n_scales, blind_iter, 8);
psnr_multiscale = psnr(img_multiscale, img_original);
ssim_multiscale = ssim(img_multiscale, img_original);
fprintf('\n【拓展对比】多尺度盲去卷积 (n_scales=%d):\n', n_scales);
fprintf('  PSNR=%.4f dB, SSIM=%.4f (vs 纯deconvblind: %.4f dB, %.4f)\n', ...
    psnr_multiscale, ssim_multiscale, psnr_blind, ssim_blind);
extension_results(end+1, :) = {'多尺度 blind+deconvreg', psnr_multiscale, ssim_multiscale};
% 对比图: deconvblind vs 混合 vs 多尺度
figure('Name', '拓展: 三类盲去卷积对比', 'NumberTitle', 'off', 'Position', [50, 50, 1400, 500]);
subplot(1, 4, 1);
imshow(img_blind, []);
title({sprintf('纯deconvblind (单尺度)'), sprintf('PSNR=%.4f, SSIM=%.4f', psnr_blind, ssim_blind)});
subplot(1, 4, 2);
imshow(img_hybrid, []);
title({sprintf('混合: blind+deconvreg'), sprintf('PSNR=%.4f, SSIM=%.4f', psnr_hybrid, ssim_hybrid)});
subplot(1, 4, 3);
imshow(img_multiscale, []);
title({sprintf('多尺度盲去卷积'), sprintf('PSNR=%.4f, SSIM=%.4f', psnr_multiscale, ssim_multiscale)});
subplot(1, 4, 4);
surf(PSF_multiscale);
title(sprintf('多尺度PSF [%d×%d]', size(PSF_multiscale, 1), size(PSF_multiscale, 2)));
xlabel('x'); ylabel('y'); zlabel('幅值');

%% 【拓展对比】梯度预处理（边缘增强）估计PSF
% 参考: Cho & Lee, "Fast Motion Deblurring", SIGGRAPH 2009
% 核心思想: 用边缘增强预处理模拟重尾先验的效果——增强边缘、平滑噪声，
% 使deconvblind更容易找到正确的线状PSF，而不被噪声干扰。
% 这是 Fergus 重尾先验的一种工程近似，不需修改deconvblind内部。
% 预处理: bilateral滤波去噪 + 反锐化掩模增强边缘
img_enhanced = img_bn;
filter_strength = 0.5;
for iter_pre = 1:2
    % bilateral滤波: 保留边缘的同时平滑噪声
    img_enhanced = imbilatfilt(img_enhanced, filter_strength, 3);
    % 反锐化掩模增强边缘
    img_enhanced = imsharpen(img_enhanced, 'Radius', 1.5, 'Amount', 0.6);
end
% 用增强图像估计PSF
[~, PSF_enhanced] = deconvblind(edgetaper(img_enhanced, INITPSF), ...
    INITPSF, blind_iter, blind_dampar);
% 用原始模糊图+增强PSF做非盲复原
NP_enh = (noise_std_est)^2 * numel(img_bn);
img_enh_deconv = deconvreg(edgetaper(img_bn, PSF_enhanced), PSF_enhanced, NP_enh);
img_enh_deconv = max(0, min(1, img_enh_deconv));
psnr_enh = psnr(img_enh_deconv, img_original);
ssim_enh = ssim(img_enh_deconv, img_original);
fprintf('\n【拓展对比】梯度预处理估计PSF (bilateral+sharpen):\n');
fprintf('  PSNR=%.4f dB, SSIM=%.4f (vs 纯deconvblind: %.4f dB, %.4f)\n', ...
    psnr_enh, ssim_enh, psnr_blind, ssim_blind);
extension_results(end+1, :) = {'梯度预处理PSF+deconvreg', psnr_enh, ssim_enh};
% 对比: 原始PSF估计 vs 增强PSF
figure('Name', '拓展: 梯度预处理估计PSF', 'NumberTitle', 'off', 'Position', [50, 50, 1200, 400]);
subplot(1, 3, 1);
imshow(img_enhanced, []);
title('预处理后图像 (去噪+锐化)');
subplot(1, 3, 2);
imshow(PSF_blind, []);
title(sprintf('原始PSF估计 [%d×%d]', size(PSF_blind, 1), size(PSF_blind, 2)));
colorbar;
subplot(1, 3, 3);
imshow(PSF_enhanced, []);
title(sprintf('增强后PSF估计 [%d×%d]', size(PSF_enhanced, 1), size(PSF_enhanced, 2)));
colorbar;
% Fergus et al. (2006) 发现自然图像的梯度分布具有"重尾"特性：
% 大量小梯度（平坦区域） + 少量大梯度（边缘）。
% 模糊使梯度分布尾部收窄、重尾消失——这一差异可用于区分"清晰"与"模糊"。
% MATLAB的deconvblind内部使用LR迭代（泊松MLE），不含显式重尾先验，
% 这是其在高斯噪声+模糊场景下效果受限的根本原因之一。
% 计算原始图像和退化图像的梯度幅值
[Gx_orig, Gy_orig] = gradient(img_original);
[Gx_bn, Gy_bn] = gradient(img_bn);
mag_orig = sqrt(Gx_orig.^2 + Gy_orig.^2);
mag_bn = sqrt(Gx_bn.^2 + Gy_bn.^2);
% 绘制 log-log 尺度直方图展示重尾特性
figure('Name', '拓展: 梯度重尾先验分析', 'NumberTitle', 'off', 'Position', [100, 100, 900, 400]);
subplot(1, 2, 1);
edges = logspace(-4, 0, 50);
hist_orig = histcounts(mag_orig(:), edges, 'Normalization', 'pdf');
hist_bn = histcounts(mag_bn(:), edges, 'Normalization', 'pdf');
loglog(edges(1:end-1), hist_orig, 'b-', 'LineWidth', 2); hold on;
loglog(edges(1:end-1), hist_bn, 'r--', 'LineWidth', 2);
xlabel('梯度幅值'); ylabel('概率密度');
title('梯度分布对比 (log-log)');
legend({'原始图像 (重尾)', '模糊+噪声图像 (尾部衰减)'}, 'Location', 'best');
grid on;
subplot(1, 2, 2);
imagesc([mag_orig, mag_bn]); axis image; colorbar;
title('梯度幅值: 原始(左) vs 退化(右)');
colormap(gca, 'hot');
fprintf('\n【拓展分析】梯度重尾先验:\n');
fprintf('  自然图像梯度呈重尾分布（大量小梯度 + 少量强边缘）\n');
fprintf('  模糊使梯度分布尾部收窄 → 可利用此差异区分清晰与模糊\n');
fprintf('  盲去卷积deconvblind缺乏显式重尾先验（仅用LR泊松MLE），\n');
fprintf('  是其在高噪声+模糊场景效果受限的根本性不足\n');
fprintf('  参考: Fergus et al., Removing Camera Shake from a Single Image, SIGGRAPH 2006\n\n');

%% 任务3: Radon变换与图像重建
fprintf('\n【任务3】Radon变换与图像重建\n');
fprintf('----------------------------------------\n');
% 生成大脑仿真图像
P = phantom(256);
% 定义三组投影角度
angles_18 = 0:10:170;   % 18个投影
angles_36 = 0:5:175;    % 36个投影
angles_90 = 0:2:178;    % 90个投影
fprintf('投影角度设置:\n');
fprintf('  组1: %d个投影 (0:10:170)\n', length(angles_18));
fprintf('  组2: %d个投影 (0:5:175)\n', length(angles_36));
fprintf('  组3: %d个投影 (0:2:178)\n', length(angles_90));
% 计算Radon变换（投影数据）
[R_18, xp_18] = radon(P, angles_18);
[R_36, xp_36] = radon(P, angles_36);
[R_90, xp_90] = radon(P, angles_90);
fprintf('\n投影数据大小:\n');
fprintf('  18个投影: %d x %d\n', size(R_18, 1), size(R_18, 2));
fprintf('  36个投影: %d x %d\n', size(R_36, 1), size(R_36, 2));
fprintf('  90个投影: %d x %d\n', size(R_90, 1), size(R_90, 2));
% 逆Radon变换重建
I_18 = iradon(R_18, angles_18);
I_36 = iradon(R_36, angles_36);
I_90 = iradon(R_90, angles_90);
% 调整重建图像大小与原图一致
I_18 = imresize(I_18, size(P));
I_36 = imresize(I_36, size(P));
I_90 = imresize(I_90, size(P));
% 计算重建质量
psnr_18 = psnr(I_18, P);
ssim_18 = ssim(I_18, P);
psnr_36 = psnr(I_36, P);
ssim_36 = ssim(I_36, P);
psnr_90 = psnr(I_90, P);
ssim_90 = ssim(I_90, P);
fprintf('\n重建质量:\n');
fprintf('  18个投影: PSNR=%.4f dB, SSIM=%.4f\n', psnr_18, ssim_18);
fprintf('  36个投影: PSNR=%.4f dB, SSIM=%.4f\n', psnr_36, ssim_36);
fprintf('  90个投影: PSNR=%.4f dB, SSIM=%.4f\n', psnr_90, ssim_90);
% 显示正弦图（Sinogram）
figure('Name', '任务3: 正弦图(Sinogram)', 'NumberTitle', 'off', 'Position', [50, 50, 1400, 400]);
subplot(1, 3, 1);
imagesc(angles_18, xp_18, R_18);
colormap(gray);
title({'18个投影', '正弦图'});
xlabel('角度 (度)');
ylabel('投影位置');
subplot(1, 3, 2);
imagesc(angles_36, xp_36, R_36);
colormap(gray);
title({'36个投影', '正弦图'});
xlabel('角度 (度)');
ylabel('投影位置');
subplot(1, 3, 3);
imagesc(angles_90, xp_90, R_90);
colormap(gray);
title({'90个投影', '正弦图'});
xlabel('角度 (度)');
ylabel('投影位置');
% 显示重建结果
figure('Name', '任务3: 仿真图像与重建结果', 'NumberTitle', 'off', 'Position', [50, 50, 1200, 900]);
subplot(2, 2, 1);
imshow(P, []);
title('原始仿真图像');
subplot(2, 2, 2);
imshow(I_18, []);
title({sprintf('18个投影重建'), sprintf('PSNR=%.4f, SSIM=%.4f', psnr_18, ssim_18)});
subplot(2, 2, 3);
imshow(I_36, []);
title({sprintf('36个投影重建'), sprintf('PSNR=%.4f, SSIM=%.4f', psnr_36, ssim_36)});
subplot(2, 2, 4);
imshow(I_90, []);
title({sprintf('90个投影重建'), sprintf('PSNR=%.4f, SSIM=%.4f', psnr_90, ssim_90)});
saveas(gcf, fullfile(fig_dir, 'exp5_fig_ct_recon.png'));

%% 重建质量 vs 投影数量曲线（定量展示"更多投影→更优质量"的权衡）
n_projs = [length(angles_18), length(angles_36), length(angles_90)];
psnr_vals = [psnr_18, psnr_36, psnr_90];
ssim_vals = [ssim_18, ssim_36, ssim_90];
figure('Name', '任务3: 重建质量 vs 投影数量', 'NumberTitle', 'off', 'Position', [100, 100, 700, 350]);
yyaxis left;
plot(n_projs, psnr_vals, 'b-o', 'LineWidth', 2, 'MarkerSize', 10);
ylabel('PSNR (dB)'); ylim([min(psnr_vals)-1, max(psnr_vals)+1]);
yyaxis right;
plot(n_projs, ssim_vals, 'r-s', 'LineWidth', 2, 'MarkerSize', 10);
ylabel('SSIM');
xlabel('投影数量');
title('重建质量 vs 投影数量（更多投影=更优质量，但辐射剂量更高）');
legend({'PSNR', 'SSIM'}, 'Location', 'best'); grid on;

%% 汇总指标对比（全文指标一览，便于课堂总结）
figure('Name', '实验汇总: 质量指标对比', 'NumberTitle', 'off', 'Position', [100, 100, 900, 400]);
methods_t1 = {'退化图像', '维纳滤波', '约束最小二乘', 'L-R'};
psnr_t1 = [psnr_degraded, psnr_wnr, psnr_reg, psnr_lucy];
ssim_t1 = [ssim_degraded, ssim_wnr, ssim_reg, ssim_lucy];
subplot(1, 2, 1);
bar(psnr_t1, 'FaceColor', [0.3 0.6 0.9]); hold on;
plot(psnr_t1, 'r*', 'MarkerSize', 12, 'LineWidth', 2);
set(gca, 'XTickLabel', methods_t1); ylabel('PSNR (dB)');
title('任务1: 复原方法 PSNR 对比'); grid on;
subplot(1, 2, 2);
bar(ssim_t1, 'FaceColor', [0.9 0.6 0.3]); hold on;
plot(ssim_t1, 'r*', 'MarkerSize', 12, 'LineWidth', 2);
set(gca, 'XTickLabel', methods_t1); ylabel('SSIM');
title('任务1: 复原方法 SSIM 对比'); grid on;
sgtitle('实验5 汇总指标对比', 'FontSize', 14);
saveas(gcf, fullfile(fig_dir, 'exp5_fig14.png'));
% 保存图片
fprintf('图片已保存: exp5_fig14.png');

%% 结果分析
fprintf('\n【分析】\n');
fprintf('1. 图像复原方法比较:\n');
fprintf('   - 维纳滤波: 需要知道PSF和噪声统计特性。标量NSR假设噪声与信号的\n');
fprintf('     功率谱密度比在全频带恒定，对自然图像(1/f²谱)仅为近似最优，\n');
fprintf('     性能受限于该简化假设，PSNR提升有限但SSIM有所改善\n');
fprintf('   - 约束最小二乘: 通过拉普拉斯正则化控制噪声放大，正则化参数NP\n');
fprintf('     直接对应总噪声能量||n||²，理论上更完备，效果显著优于维纳滤波\n');
fprintf('   - Lucy-Richardson: 基于泊松噪声MLE的迭代复原，对高斯噪声鲁棒性弱。\n');
fprintf('     本实验效果不佳原因：(1)泊松MLE与高斯噪声模型不匹配，\n');
fprintf('     收敛路径偏离真实解；(2)迭代次数不足（调优结果为%d次），\n', lucy_iter);
fprintf('     在高斯似然面上远未收敛；(3)更适合光子计数模式（天文、显微）的泊松退化。\n');
fprintf('   【拓展】反卷积振铃(Ringing): 强边缘附近的波动伪影，可用 edgetaper 对图像\n');
fprintf('           边缘做渐弱处理来抑制(在退化图像输入 deconv* 前调用一次即可):\n');
fprintf('           img_degraded = edgetaper(img_degraded, PSF_gaussian);\n');
fprintf('2. 盲去卷积: 在PSF未知时仍能估计并复原图像，但受多重因素制约:\n');
fprintf('   - 初始PSF形状至关重要: 运动模糊的PSF是线状的，必须用线状核初始化;\n');
fprintf('     若用均匀方核，算法收敛到圆形核，复原完全失败(PSNR反而下降)\n');
fprintf('   - 应使用 fspecial(''motion'', len, theta) 生成线状初始PSF，\n');
fprintf('     参数通过观察退化图像边缘拖影方向和长度估计\n');
fprintf('   - DAMPAR 抑制噪声干扰: 盲去卷积内部用LR迭代，对高斯噪声鲁棒性弱，\n');
fprintf('     必须设置较高DAMPAR(≈8σ)防止噪声在交替迭代中带偏PSF估计\n');
fprintf('   - 性能限制: 当噪声较强(σ≈8)时，盲去卷积复原PSNR可能低于退化图像，\n');
fprintf('     因为LR迭代在噪声和模糊的耦合下难以同时收敛\n');
fprintf('   - 混合方法: 用deconvblind估计PSF，再用deconvreg做非盲复原，\n');
fprintf('     可结合盲估计与高斯噪声最优正则化的优势\n');
fprintf('   【参考】本实验运动模糊参数约为 长度=7~10像素, 角度≈150~160°\n');
fprintf('3. Radon重建: 投影角度越多，重建质量越好，但辐射剂量也增加\n');
fprintf('   医学成像需要在图像质量和患者安全之间权衡\n');
fprintf('4. 【拓展】梯度重尾先验 (Fergus et al. 2006):\n');
fprintf('   - 自然图像的梯度分布具有重尾特性（大量小梯度+少量强边缘）\n');
fprintf('   - 模糊使梯度尾部收窄，清晰图像保持重尾，利用此差异可估计PSF\n');
fprintf('   - MATLAB deconvblind 使用LR迭代（泊松MLE+均匀先验），不含重尾先验，\n');
fprintf('     这是其在强噪声+模糊场景下效果受限的根本原因\n');
fprintf('   - 改进方向: (a)多尺度粗→精估计避免局部最优; (b)用边缘预测模拟重尾先验;\n');
fprintf('     (c)分离核估计与非盲复原，用deconvreg替代LR做最终复原\n');
fprintf('5. 【拓展对比】盲去卷积改进方法汇总:\n');
fprintf('   ┌──────────────────────────┬──────────┬──────────┐\n');
fprintf('   │ 方法                      │ PSNR(dB) │  SSIM    │\n');
fprintf('   ├──────────────────────────┼──────────┼──────────┤\n');
fprintf('   │ 退化图像 (未处理)        │ %8.4f │ %8.4f │\n', psnr_bn, ssim_bn);
fprintf('   │ 纯deconvblind (单尺度)   │ %8.4f │ %8.4f │\n', psnr_blind, ssim_blind);
for i = 1:size(extension_results, 1)
    fprintf('   │ %-24s │ %8.4f │ %8.4f │\n', ...
        extension_results{i, 1}, extension_results{i, 2}, extension_results{i, 3});
end
fprintf('   └──────────────────────────┴──────────┴──────────┘\n');
fprintf('\n========================================\n');
fprintf('        实验5 完成!\n');
fprintf('========================================\n');