% NSGA3������
clc;
clear;
close all

%% ���ⶨ��
CostFunction = @(x) MOP2(x);  % Ŀ�꺯��
nVar = 1;    % ���߱�������
VarSize = [1 nVar]; % ���߱���ά�Ⱦ���
VarMin = [6];           % �����½�
VarMax = [20];           % �����Ͻ�

% Ŀ�꺯������
nObj = numel(CostFunction(unifrnd(VarMin, VarMax, VarSize)));

%% NSGA-III ����

% �����ο���
nDivision = 10;
Zr = GenerateReferencePoints(nObj, nDivision);

MaxIt = 1;  % ����������

nPop = 100;  % ��Ա��Ŀ

pCrossover = 0.8;       % �������
nCrossover = 2*round(pCrossover*nPop/2); % �Ӵ���Ŀ

pMutation = 0.8;       % �������
nMutation = round(pMutation*nPop);  % ���������Ŀ

mu = 0.5;     % ������0.02

sigma = 0.005*(VarMax-VarMin); % ���첽


%% ����

params.nPop = nPop;
params.Zr = Zr;
params.nZr = size(Zr, 2);
params.zmin = [];
params.zmax = [];
params.smin = [];

%% ��ʼ��

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
    pop(i).Position = unifrnd(VarMin, VarMax, VarSize);         % ��ʼ�����߱���
    pop(i).Cost = CostFunction(pop(i).Position);                % ����Ŀ�꺯��ֵ
end

% �Գ�Ա��������ѡ��
[pop, F, params] = SortAndSelectPopulation(pop, params);


%% NSGA-III Main Loop

for it = 1:MaxIt
 
    % ����
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

    % ����
    popm = repmat(empty_individual, nMutation, 1);
    for k = 1:nMutation

        i = randi([1 nPop]);
        p = pop(i);

        popm(k).Position = Mutate(p.Position, mu, sigma);
        popm(k).Position = max(min(popm(k).Position,VarMax),VarMin);
        
        popm(k).Cost = CostFunction(popm(k).Position);

    end

    % �ϲ�������Ա�������Ӵ��뽻���Ӵ���Ա
    pop = [pop
           popc
           popm]; %#ok
    
    % �Գ�Ա��������ѡ��
    [pop, F, params] = SortAndSelectPopulation(pop, params);
    
    % ��F1
    F1 = pop(F{1});
    F4 = [pop
          F1
          popc
          popm];

    % ��ʾ������Ϣ
    disp(['Iteration ' num2str(it) ': Number of F1 Members = ' num2str(numel(F1))]);

    % ����F4��ͼ��
    figure(1);
    PlotCosts(F4);
    pause(0.01);
 
end

%% ���


disp(['Final Iteration: Number of F1 Members = ' num2str(numel(F1))]);
disp('Optimization Terminated.');

