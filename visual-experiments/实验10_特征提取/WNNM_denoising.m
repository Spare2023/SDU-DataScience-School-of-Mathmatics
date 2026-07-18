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

function [denoised, PSNR_vec] = WNNM_denoising(noisy, clean, sigma, params)
%% WNNM: Weighted Nuclear Norm Minimization 图像去噪
%  实现 Gu et al. CVPR 2014 论文算法
%
%  输入:
%    noisy  - 噪声图像 (double, [0,1])
%    clean  - 干净图像 (用于计算中间PSNR，可选)
%    sigma  - 噪声标准差
%    params - 可选参数结构体
%
%  输出:
%    denoised  - 去噪图像
%    PSNR_vec  - 每轮迭代的 PSNR 值序列

arguments
    noisy double {mustBeNonempty}
    clean double = []
    sigma double = 30/255
    params struct = struct()
end

%% ===== 参数设置 =====
% 默认参数 (针对 σ=30/255 ≈ 0.1176 的设定)
patch_size  = 7;                % 图像块大小 7×7
step        = 4;                % 参考块步长
search_win  = 30;               % 搜索窗口半宽
K           = 70;               % 每个组中相似块数量
iter_max    = 12;               % 迭代次数
delta       = 0.1;              % 迭代正则化参数
c           = 2 * sqrt(2);      % 权重常数 √2/√2/2√2 视噪声水平调整
eps0        = 1e-8;             % 避免除零

% 覆盖用户自定义参数
if isfield(params, 'patch_size'),  patch_size  = params.patch_size;  end
if isfield(params, 'step'),        step        = params.step;        end
if isfield(params, 'search_win'),  search_win  = params.search_win;  end
if isfield(params, 'K'),           K           = params.K;           end
if isfield(params, 'iter_max'),    iter_max    = params.iter_max;    end
if isfield(params, 'delta'),       delta       = params.delta;       end
if isfield(params, 'c'),           c           = params.c;           end

has_clean = ~isempty(clean);
PSNR_vec  = zeros(iter_max, 1);

[M, N] = size(noisy);

%% ===== 初始化 =====
% 第一轮以噪声图自身作为初始估计
current = noisy;

% 预提取所有参考块位置 (网格采样)
ref_rows = 1:step:M - patch_size + 1;
ref_cols = 1:step:N - patch_size + 1;
n_ref   = length(ref_rows) * length(ref_cols);
ref_pos = zeros(n_ref, 2);
idx = 1;
for i = ref_rows
    for j = ref_cols
        ref_pos(idx, :) = [i, j];
        idx = idx + 1;
    end
end

fprintf('  WNNM 参数: patch=%d×%d, step=%d, K=%d, iter=%d\n', ...
        patch_size, patch_size, step, K, iter_max);

%% ===== 主迭代 =====
for iter = 1:iter_max
    iter_time = tic;

    % --- 迭代正则化 ---
    % y^{(k)} = x^{(k-1)} + δ(n - x^{(k-1)})
    reg = current + delta * (noisy - current);

    % --- 累积数组 ---
    accum = zeros(M, N);
    w_accum = zeros(M, N);

    % --- 遍历所有参考块 ---
    for ridx = 1:n_ref
        r = ref_pos(ridx, 1);
        c0 = ref_pos(ridx, 2);

        % 1) 在当前估计(正则化)图上提取参考块
        ref_patch = reg(r:r+patch_size-1, c0:c0+patch_size-1);
        ref_vec = ref_patch(:);

        % 2) 定义搜索窗口
        r_min = max(1,  r - search_win);
        r_max = min(M - patch_size + 1, r + search_win);
        c_min = max(1,  c0 - search_win);
        c_max = min(N - patch_size + 1, c0 + search_win);

        rows_range = r_min:r_max;
        cols_range = c_min:c_max;
        n_candidates = length(rows_range) * length(cols_range);

        % 3) 计算 SSD 寻找相似块
        ssd = zeros(n_candidates, 1);
        cand_pos = zeros(n_candidates, 2);

        cand_idx = 1;
        for ii = rows_range
            for jj = cols_range
                cand = reg(ii:ii+patch_size-1, jj:jj+patch_size-1);
                ssd(cand_idx) = sum((ref_vec - cand(:)).^2);
                cand_pos(cand_idx, :) = [ii, jj];
                cand_idx = cand_idx + 1;
            end
        end

        % 4) 选出最相似的 K 个块
        n_use = min(K, n_candidates);
        [~, order] = sort(ssd);
        sel = order(1:n_use);
        use_pos = cand_pos(sel, :);

        % 5) 从噪声图中提取对应块 → Y_j (列向量化)
        Y = zeros(patch_size^2, n_use);
        for p = 1:n_use
            ii = use_pos(p, 1);
            jj = use_pos(p, 2);
            patch_block = noisy(ii:ii+patch_size-1, jj:jj+patch_size-1);
            Y(:, p) = patch_block(:);
        end

        % 6) SVD: Y = U Σ V'
        [Uy, Sy, Vy] = svd(Y, 'econ');
        sv_y = diag(Sy);  % 噪声图奇异值

        % 7) 估计干净奇异值 σ_i(X̂) = √max(σ_i(Y)² - K·σ², 0)
        sv_x_est = sqrt(max(sv_y.^2 - n_use * sigma^2, 0));

        % 8) 计算权重 w_i = c·√K·σ² / (σ_i(X̂) + ε)
        weights = c * sqrt(n_use) * sigma^2 ./ (sv_x_est + eps0);

        % 9) 加权软阈值收缩: σ̂_i(X) = max(σ_i(Y) - w_i, 0)
        sv_x = max(sv_y - weights, 0);

        % 10) 重构组矩阵 X_j
        % 注意: 'econ' SVD 返回 Uy 的列数 = min(patch_size^2, n_use)
        r = size(Uy, 2);
        X = Uy(:, 1:r) * diag(sv_x(1:r)) * Vy(:, 1:r)';

        % 11) 累积回图像
        for p = 1:n_use
            ii = use_pos(p, 1);
            jj = use_pos(p, 2);
            patch = reshape(X(:, p), [patch_size, patch_size]);

            % 边界保护
            i_end = min(ii + patch_size - 1, M);
            j_end = min(jj + patch_size - 1, N);

            accum(ii:i_end, jj:j_end) = accum(ii:i_end, jj:j_end) + patch(1:i_end-ii+1, 1:j_end-jj+1);
            w_accum(ii:i_end, jj:j_end) = w_accum(ii:i_end, jj:j_end) + 1;
        end

    end % 参考块循环

    % 归一化 (取平均)
    current = accum ./ max(w_accum, 1);
    current = max(0, min(1, current));

    % 记录 PSNR
    if has_clean
        PSNR_vec(iter) = psnr(current, clean);
        fprintf('  迭代 %2d/%d: PSNR = %.4f dB (%.2fs)\n', ...
                iter, iter_max, PSNR_vec(iter), toc(iter_time));
    else
        fprintf('  迭代 %2d/%d: (%.2fs)\n', ...
                iter, iter_max, toc(iter_time));
    end

end % 迭代循环

denoised = current;

end
