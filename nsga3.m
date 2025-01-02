% NSGA3主程序
clc;
clear;
close all

%% 问题定义
CostFunction = @(x) MOP2(x);  % 目标函数
nVar = 1;    % 决策变量个数
VarSize = [1 nVar]; % 决策变量维度矩阵
VarMin = [6];           % 变量下界
VarMax = [20];           % 变量上界

% 目标函数个数
nObj = numel(CostFunction(unifrnd(VarMin, VarMax, VarSize)));

%% NSGA-III 参数

% 产生参考点
nDivision = 10;
Zr = GenerateReferencePoints(nObj, nDivision);

MaxIt = 1;  % 最大迭代次数

nPop = 100;  % 成员数目

pCrossover = 0.8;       % 交叉比例
nCrossover = 2*round(pCrossover*nPop/2); % 子代数目

pMutation = 0.8;       % 变异比例
nMutation = round(pMutation*nPop);  % 变异个体数目

mu = 0.5;     % 变异率0.02

sigma = 0.005*(VarMax-VarMin); % 变异步


%% 参数

params.nPop = nPop;
params.Zr = Zr;
params.nZr = size(Zr, 2);
params.zmin = [];
params.zmax = [];
params.smin = [];

%% 初始化

% disp('Staring NSGA-III ...');

empty_individual.Position = [];             % 
empty_individual.Cost = [];
empty_individual.Rank = [];
empty_individual.DominationSet = [];
empty_individual.DominatedCount = [];
empty_individual.NormalizedCost = [];
empty_individual.AssociatedRef = [];
empty_individual.DistanceToAssociatedRef = [];

pop = repmat(empty_individual, nPop, 1);
for i = 1:nPop
    pop(i).Position = unifrnd(VarMin, VarMax, VarSize);         % 初始化决策变量
    pop(i).Cost = CostFunction(pop(i).Position);                % 计算目标函数值
end

% 对成员进行排序选择
[pop, F, params] = SortAndSelectPopulation(pop, params);


%% NSGA-III Main Loop

for it = 1:MaxIt
 
    % 交叉
    popc = repmat(empty_individual, nCrossover/2, 2);
    for k = 1:nCrossover/2

        i1 = randi([1 nPop]);
        p1 = pop(i1);

        i2 = randi([1 nPop]);
        p2 = pop(i2);

        [popc(k, 1).Position, popc(k, 2).Position] = Crossover(p1.Position, p2.Position);
        popc(k, 1).Position = max(min(popc(k, 1).Position,VarMax),VarMin);
        popc(k, 2).Position = max(min(popc(k, 2).Position,VarMax),VarMin);
        
        popc(k, 1).Cost = CostFunction(popc(k, 1).Position);
        popc(k, 2).Cost = CostFunction(popc(k, 2).Position);

    end
    popc = popc(:);

    % 变异
    popm = repmat(empty_individual, nMutation, 1);
    for k = 1:nMutation

        i = randi([1 nPop]);
        p = pop(i);

        popm(k).Position = Mutate(p.Position, mu, sigma);
        popm(k).Position = max(min(popm(k).Position,VarMax),VarMin);
        
        popm(k).Cost = CostFunction(popm(k).Position);

    end

    % 合并父代成员，变异子代与交叉子代成员
    pop = [pop
           popc
           popm]; %#ok
    
    % 对成员进行排序并选择
    [pop, F, params] = SortAndSelectPopulation(pop, params);
    
    % 存F1
    F1 = pop(F{1});
    F4 = [pop
          F1
          popc
          popm];

    % 显示迭代信息
    disp(['Iteration ' num2str(it) ': Number of F1 Members = ' num2str(numel(F1))]);

    % 画出F4的图像
    figure(1);
    PlotCosts(F4);
    pause(0.01);
 
end

%% 结果


disp(['Final Iteration: Number of F1 Members = ' num2str(numel(F1))]);
disp('Optimization Terminated.');


