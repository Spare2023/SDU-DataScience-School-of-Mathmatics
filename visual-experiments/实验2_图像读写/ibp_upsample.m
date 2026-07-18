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

function HR = ibp_upsample(low_res, target_size, interp_method, nIter, lambda)
%IBP_UPSAMPLE IBP超分辨率重建 (Iterative Back Projection)
%   输入:
%       low_res     - 低分辨率图像 [0,1] double
%       target_size - 目标尺寸, 如 size(I)
%       interp_method - 插值方法, 如 'bicubic'/'lanczos3'
%       nIter       - (可选) 迭代次数, 默认 60
%       lambda      - (可选) 步长, 默认 0.6
%   输出:
%       HR - 超分辨率重建结果, 与 target_size 同尺寸
%
%   算法原理:
%       以标准插值结果为初始估计, 反复执行:
%       ① 模拟退化 (下采样) → ② 低分辨率残差 → ③ 残差上采样 → ④ 修正高分辨率估计
%       使重建图的下采样版本逼近输入的低分辨率图, 从而恢复高频细节.
%
%   参考: Irani & Peleg, "Improving resolution by image registration", CVGIP 1991.

if nargin < 4
    nIter = 60;
end
if nargin < 5
    lambda = 0.6;
end

% 初始估计: 标准插值放大
HR = imresize(low_res, target_size, interp_method);

scale = 0.25;  % 下采样倍率 (对应4倍减采样)

for iter = 1:nIter
    % (a) 模拟退化: 与 low_res 相同方式下采样
    HR_down = imresize(HR, scale);

    % (b) 低分辨率残差
    residual = low_res - HR_down;

    % (c) 残差上采样回高分辨率
    res_up = imresize(residual, target_size, interp_method);

    % (d) 修正高分辨率估计
    HR = HR + lambda * res_up;

    % (e) 截断到合法像素范围 [0,1]
    HR = max(0, min(1, HR));
end

end
