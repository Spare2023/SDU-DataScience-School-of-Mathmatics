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

%% 实验1: MATLAB基础操作
% 课程: 视觉与数据计算
% 重点函数: addpath, genpath, save, load

clear all;
close all;
clc;

% 创建图片保存目录
fig_dir = fullfile(fileparts(mfilename('fullpath')), 'figs');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

fprintf('========================================\n');
fprintf('        实验1: MATLAB基础操作\n');
fprintf('========================================\n\n');

%% 任务1: 熟悉MATLAB工作环境
% 在D盘下增加搜索路径(包括子文件夹): D:\MATLAB\vision\myDemo
fprintf('【任务1】设置MATLAB搜索路径\n');
fprintf('----------------------------------------\n');

% 定义路径
path_to_add = 'D:\MATLAB\vision\myDemo';

% 检查路径是否存在，如果不存在则创建
if ~exist(path_to_add, 'dir')
    mkdir(path_to_add);
    fprintf('创建目录: %s\n', path_to_add);
end

% 添加路径（包括子文件夹）
addpath(genpath(path_to_add));
fprintf('已添加路径: %s\n', path_to_add);
fprintf('包括所有子文件夹\n\n');

%% 任务2: 熟悉矩阵运算
fprintf('【任务2】矩阵运算\n');
fprintf('----------------------------------------\n');

% (1) 生成一个5阶魔术矩阵
fprintf('(1) 5阶魔术矩阵:\n');
A_magic = magic(5);
disp(A_magic);

% 将该矩阵的第2行第3列元素赋值给变量a
a = A_magic(2, 3);
fprintf('\n变量 a (第2行第3列元素) = %d\n', a);

% 将由矩阵的第(2,3,4)行和第(1,3,5)列构成的子矩阵赋值给变量b
b = A_magic([2,3,4], [1,3,5]);
fprintf('\n子矩阵 b (第2,3,4行和第1,3,5列):\n');
disp(b);

% (2) 生成一个3阶全1矩阵c
fprintf('\n(2) 3阶全1矩阵 c:\n');
c = ones(3, 3);
disp(c);

% 计算b*c (矩阵乘法)
fprintf('\nb * c (矩阵乘法):\n');
result_mult = b * c;
disp(result_mult);
fprintf('结果大小: %d x %d\n', size(result_mult, 1), size(result_mult, 2));

% 计算b.*c (Hadamard乘积/逐元素乘法)
fprintf('\nb .* c (Hadamard乘积/逐元素乘法):\n');
% 注意: b是3x3, c也是3x3，可以直接进行逐元素乘法
result_hadamard = b .* c;
disp(result_hadamard);

%% 任务3: 创建M文件myCircle
fprintf('\n【任务3】创建圆形矩阵\n');
fprintf('----------------------------------------\n');

% 生成101行×161列的圆形矩阵
rows = 101;
cols = 161;
center_row = (rows + 1) / 2;  % 中心行
center_col = (cols + 1) / 2;  % 中心列
radius = 25;                   % 半径

% 创建坐标网格
[col_coords, row_coords] = meshgrid(1:cols, 1:rows);

% 计算每个点到圆心的距离
distances = sqrt((row_coords - center_row).^2 + (col_coords - center_col).^2);

% 创建圆形矩阵: 圆上元素取值0.2, 其余元素取值0.8
% 使用容差来判断"圆上"的点
tolerance = 0.5;
A = 0.8 * ones(rows, cols);  % 默认值为0.8
A(abs(distances - radius) <= tolerance) = 0.2;  % 圆上的点为0.2

fprintf('圆形矩阵 A 大小: %d x %d\n', rows, cols);
fprintf('圆心位置: (%.1f, %.1f)\n', center_row, center_col);
fprintf('半径: %d\n', radius);
fprintf('圆上元素值: 0.2\n');
fprintf('其余元素值: 0.8\n');

% 显示圆形矩阵
figure('Name', '圆形矩阵', 'NumberTitle', 'off');
imshow(A, []);
title('圆形矩阵 A (圆上=0.2, 其余=0.8)');
colorbar;

% 保存图片
saveas(gcf, fullfile(fig_dir, 'exp1_circle_matrix.png'));
fprintf('图片已保存: exp1_circle_matrix.png\n');

%% 任务4: 保存和载入工作空间
fprintf('\n【任务4】保存和载入工作空间\n');
fprintf('----------------------------------------\n');

% 保存变量到指定文件夹
save_path = fullfile(path_to_add, 'experiment1_data.mat');
save(save_path, 'a', 'b', 'c', 'A');
fprintf('变量 a, b, c, A 已保存到:\n%s\n', save_path);

% 保存myCircle函数文件（当前脚本）
% 注意: 实际使用中，应该将圆形矩阵生成代码保存为独立的myCircle.m函数文件
myCircle_code = sprintf([...
    'function A = myCircle(rows, cols, radius, center_val, other_val)\n' ...
    '%% myCircle 生成圆形矩阵\n' ...
    '%% 输入:\n' ...
    '%%   rows - 矩阵行数\n' ...
    '%%   cols - 矩阵列数\n' ...
    '%%   radius - 圆的半径\n' ...
    '%%   center_val - 圆上元素的值\n' ...
    '%%   other_val - 其他元素的值\n' ...
    '%% 输出:\n' ...
    '%%   A - 生成的圆形矩阵\n\n' ...
    'if nargin < 3\n' ...
    '    rows = 101; cols = 161; radius = 25;\n' ...
    '    center_val = 0.2; other_val = 0.8;\n' ...
    'elseif nargin < 5\n' ...
    '    center_val = 0.2; other_val = 0.8;\n' ...
    'end\n\n' ...
    'center_row = (rows + 1) / 2;\n' ...
    'center_col = (cols + 1) / 2;\n' ...
    '[col_coords, row_coords] = meshgrid(1:cols, 1:rows);\n' ...
    'distances = sqrt((row_coords - center_row).^2 + (col_coords - center_col).^2);\n' ...
    'tolerance = 0.5;\n' ...
    'A = other_val * ones(rows, cols);\n' ...
    'A(abs(distances - radius) <= tolerance) = center_val;\n' ...
    'end\n']);

myCircle_path = fullfile(path_to_add, 'myCircle.m');
fid = fopen(myCircle_path, 'w');
fprintf(fid, '%s', myCircle_code);
fclose(fid);
fprintf('函数文件 myCircle.m 已保存到:\n%s\n', myCircle_path);

% 清除工作空间中的变量
clear a b c A
fprintf('\n工作空间已清除\n');

% 以不同的变量名称重新载入
load(save_path);
fprintf('已从 %s 载入变量\n', save_path);

% 重命名变量
a_loaded = a; clear a;
b_loaded = b; clear b;
c_loaded = c; clear c;
A_loaded = A; clear A;

fprintf('\n变量已重命名为:\n');
fprintf('  a -> a_loaded = %d\n', a_loaded);
fprintf('  b -> b_loaded (%d x %d 矩阵)\n', size(b_loaded, 1), size(b_loaded, 2));
fprintf('  c -> c_loaded (%d x %d 矩阵)\n', size(c_loaded, 1), size(c_loaded, 2));
fprintf('  A -> A_loaded (%d x %d 矩阵)\n', size(A_loaded, 1), size(A_loaded, 2));

%% 显示结果
fprintf('\n重命名后的变量 b_loaded:\n');
disp(b_loaded);

fprintf('\n========================================\n');
fprintf('        实验1 完成!\n');
fprintf('========================================\n');

close all;
