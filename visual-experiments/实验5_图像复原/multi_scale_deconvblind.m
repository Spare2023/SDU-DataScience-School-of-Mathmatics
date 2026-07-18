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

function [img_restored, PSF_est] = multi_scale_deconvblind(img, init_psf, noise_std, n_scales, base_iter, base_damp_factor)
    %% 多尺度由粗到精盲去卷积 (Coarse-to-Fine Blind Deconvolution)
    % 参考: Fergus et al., "Removing camera shake from a single image", SIGGRAPH 2006
    %
    % 核心思想:
    %   粗尺度下模糊核的有效尺寸变小 → 容易估计大致形状 →
    %   上采样作为下一层初始化 → 逐层精调 → 避免局部最优
    %
    % 输入:
    %   img             - 退化图像 (double, [0,1])
    %   init_psf        - 初始PSF (运动模糊核)
    %   noise_std       - 噪声标准差估计值
    %   n_scales        - 金字塔层数 (默认3)
    %   base_iter       - 每层基础迭代次数 (默认20)
    %   base_damp_factor- DAMPAR乘数 (默认8)
    %
    % 输出:
    %   img_restored    - 最终复原图像 (deconvreg 非盲复原)
    %   PSF_est         - 多尺度估计的PSF

    % 默认参数
    if nargin < 4 || isempty(n_scales),        n_scales = 3;          end
    if nargin < 5 || isempty(base_iter),        base_iter = 20;        end
    if nargin < 6 || isempty(base_damp_factor), base_damp_factor = 8;  end

    % 确保 init_psf 归一化
    init_psf = init_psf / sum(init_psf(:));

    current_psf = init_psf;

    fprintf('  多尺度盲去卷积 (n_scales=%d):\n', n_scales);

    for scale = n_scales:-1:1
        % 下采样因子: 4, 2, 1（对n_scales=3）
        factor = 2^(scale - 1);

        % 下采样图像
        img_small = imresize(img, 1/factor);

        % 调整PSF尺寸匹配当前尺度
        % 核尺寸与图像尺寸同比例缩放
        psf_size = size(current_psf);
        new_psf_rows = max(3, ceil(psf_size(1) / factor));
        new_psf_cols = max(3, ceil(psf_size(2) / factor));
        psf_scaled = imresize(current_psf, [new_psf_rows, new_psf_cols], 'nearest');
        psf_scaled = max(psf_scaled, 0);
        psf_scaled = psf_scaled / sum(psf_scaled(:));

        % 每层参数: 粗尺度用较少迭代（收敛快）+ 较强阻尼
        iter_this = max(5, round(base_iter / factor));
        damp_this = base_damp_factor * noise_std * factor;

        % 盲去卷积
        img_taper = edgetaper(img_small, psf_scaled);
        [~, psf_est] = deconvblind(img_taper, psf_scaled, iter_this, damp_this);

        fprintf('    尺度 %d (1/%d): iter=%d, damp=%.4f, PSF尺寸=[%d,%d]\n', ...
            scale, factor, iter_this, damp_this, size(psf_est, 1), size(psf_est, 2));

        % 上采样PSF用于下一层（更精细的尺度）
        if scale > 1
            % 上采样2倍（下一尺度的factor是当前的一半）
            current_psf = imresize(psf_est, 2, 'bilinear');
            current_psf = max(current_psf, 0);
            current_psf = current_psf / sum(current_psf(:));
        else
            current_psf = psf_est;
        end
    end

    PSF_est = current_psf;

    % 最终非盲复原: 用 deconvreg（高斯噪声最优）替代 RL
    % Fergus 原论文用 RL 做非盲复原，但那是无噪声或低噪声场景。
    % 本实验 σ≈8 的高斯噪声下，deconvreg（L2正则化）更合适。
    NP = (noise_std)^2 * numel(img);
    img_restored = deconvreg(edgetaper(img, PSF_est), PSF_est, NP);
    img_restored = max(0, min(1, img_restored));

    fprintf('  多尺度完成: PSF尺寸=[%d,%d]\n', size(PSF_est, 1), size(PSF_est, 2));
end
