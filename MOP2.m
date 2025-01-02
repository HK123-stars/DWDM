%
% 目标函数
%
function z = MOP2(x)
   
  %  初始条件
 %% 水量平衡方程
% data = xlsread('水量平衡.xlsx');
P = load('P.mat');
P = struct2array(P); % 将结构体转化为数组
ETo = load('ETo.mat');
ETo = struct2array(ETo);
Ke = load('Ke.mat');
Ke = struct2array(Ke);
T = load('T.mat');
T = struct2array(T);
Z = load('Z.mat');
Z = struct2array(Z);
CR = load('CR.mat');
CR = struct2array(CR);
% P  = data(:,3); % 有效降雨量，mm
% ETO = data(:,6); % 参考作物蒸腾量，mm
% Ke = data(:,20);  %土壤蒸发系数
% T = data(:,8); % 作物蒸散发量，mm
% Z  = data(:,10); % 土壤根区变化量，mm（渗漏）
% CR = data(:,12); % 毛细管上升量，mm（地下水交换量）
%%  产量
% PAR = data(:,13);  %有效光合辐射（×10-6）
RA = load('RA.mat');
RA = struct2array(RA);
FT = load('FT.mat');
FT = struct2array(FT);
HUI = load('HUI.mat');
HUI = struct2array(HUI);
% RA = data(:,14);  %太阳辐射
% FT = data(:,15);  %温度胁迫
% HUI = data(:,16);  %日热量单位指数
WA = 50;%50; %光能-生物能转化比率
HIp = 0.48;%0.48;  %潜在收获指数
Hi = 0.45;  %经济系数
WSYF =0.5 ;  %干旱敏感因子
%%  碳排放
Cz = 1957.831;  %(作物种植产生的碳排放，kg/ha)
CH4 = 11454.576;  %CH4换算的碳排放(kg/ha)
NO2 = 1594.622;  %NO2排放（kg/ha）
SOC = 0.753;  %土壤有机碳，g/m2(40cm*1.43g/cm3*0.01316）
k = 0.014;  %土壤碳库周转速率，day
Tf = 25; %异养呼吸参考温度
hmin = -14887;  %最低水势
hmax = -100;  %最大水势
 a = 1.29;  %参数1.29
b = -1.05;  %参数-1.05
c = 1.36 ;  %参数1.36
d = -1.05;   %-1.05;  %参数
A = 1;  %面积
Td = load('Td.mat');
Td = struct2array(Td);
r = load('r.mat');
r = struct2array(r);
Ws = load('Ws.mat');
Ws = struct2array(Ws);
% Td = data(:,17);  %土壤温度
% r = data(:,18);  %根分布比例
% Ws = data(:,19);  %湿度影响系数
Cf = 0.41;  %作物合成1g干重需要吸收的碳（tCO2eq）
CSOC = 1.316;  %土壤有机碳含量（%）
CSOC0 = 1.0415;  %初始土壤有机碳含量（%）
dr = 1.36;  %容重(g/cm3)
H = 40;  %深度（cm）
t = 1;  %年份（a）
t0 = 0;  %基期年份
n = 1; % 假设我们想要10个随机x和对应的y
y_values = zeros(n, 1); % 初始化y值数组
for j = 1:n
    % 随机生成x
      %v = round(rand*(20-6)+6, 1);
      v = 15.27031451;
    WW0 = [72; zeros(131, 1)];  % 初始猜测WW0，只有第一项已知为72    第i-1天的含水量，mm（包括水层）
    WW1 = zeros(131, 1); % 第i天的含水量（包括水层）
    I = zeros(131, 1);   % 第i天的生育期灌水量
    for i = 1:51
    % 计算 W1
    W1 = fsolve(@(W1) W1 - WW0(i) - I(i) - P(i) + Ke(i) * E(ETo(i), W1) + T(i) + DP(Z(i), W1) - CR(i), 0);
    ET(i) =  Ke(i) * E(ETo(i), W1) + T(i);
    % 将 W1 放入 WW0 中
    WW0(i + 1) = W1;
    
    % 将 W1 放入 WW1 中
    WW1(i) = W1;
    % 根据WW1(i)和x更新I(i)的值
    if WW1(i) > v
        I(i+1) = 0;
    else 
        I(i+1) = 42 - WW1(i);
    end
    if P(i)>10;
        I(i-1)=0;
    end
    end
    
%     disp('计算得到的WW1:');
%     disp(WW1);
    WW1(find(WW1>72))=72;
    %% 水分胁迫
     WWW1 = 22.*(WW1>22)+WW1.*(WW1>0 & WW1<=22) +0.*(WW1<0); %  WWW1 耕作层土壤含水量，mm
%     WWW1 = double(WW1 >= 22) .* 22 + double(0 <= WW1 & WW1 < 22) .* WW1 + double(WW1 < 0) .* 0;
    WWW1(find(WWW1<6))=6 ;% 约束
    h = -420*((WWW1-2.68)/(22-2.68)).^(-1/0.3); % h 土壤吸力，mm
    FW=0.*(h<=-14887|h>0)+((h+0)/(-450+0)).*(h>-450 & h<=0)+1.*(h>-4770 & h<=-450)+((h+14887)/(-4770+14887)).*(h>-14887 & h<=-4770);
%     FW = double(h >= 0 | h <=-14887) .* 0 + double(-420 <= h & h < 0) .* (h-0)/(-420-0) +  double(-6000 <= h & h < 420) .* (h+6000)/(-420+6000)+double(-14887 <= h & h < -6000) .* (h+14887)/(-6000+14887);
      % FW 水分胁迫(可以加个含水量下限)
     FW1=0.*(WWW1<=6)+((WWW1-6)/(22-6)).*(WWW1>6 & WWW1<=22)+1.*(WWW1>22);
%     FW1 = double(WWW1 >= 22) .* 1 + double(6 <= WWW1 & WWW1 < 22) .* (WWW1-6)/(22-6) + double(WWW1 < 6) .* 0;
    %  FW 水分胁迫(可以加个含水量下限)
    %% 冠层覆盖度CC
%     data = xlsread('matlab冠层覆盖度.xlsx');
    CC0 = load('CC0.mat');
    CC0 = struct2array(CC0);
    FTC = load('FTC.mat');
    FTC = struct2array(FTC);
    CCD0 = load('CCD0.mat');
    CCD0 = struct2array(CCD0);
%     CC0 = data(:,1); % 潜在冠层覆盖度
%     FTC = data(:,2); % 冠层温度胁迫因子
%     CCD0 = data(:,10); % 冠层潜在变化量
    CGC = min(FW(1:97,1:1),FTC(1:97,1:1));         % 冠层生长胁迫因子
    CDC = 1-(min(FW(98:131,1:1),FTC(98:131,1:1)).^8);  % 冠层衰老胁迫因子
    CCD = CCD0.*[CGC;CDC]; %冠层实际变化量
    CC = cumsum(CCD,1);    %实际冠层覆盖度
    CC1 = 1.72.*CC-CC.*CC+0.3.*((CC).^3); %调整后的冠层覆盖度
    LAI =-log(1-((CC1).^(1/1.2))/1.005)/0.6;                      % LAI 实际叶面积指数
    LAI0 = load('LAI0.mat');
    LAI0 = struct2array(LAI0);
    LAI1 = load('LAI1.mat');
    LAI1 = struct2array(LAI1);
%     LAI0 = data(:,8); % 潜在叶面积指数
%     LAI1 = data(:,9); % 潜在叶面积生长过程变化量
    LAI2 = sum(LAI1.*FW);  %齐穗期实际叶面积指数
    %% 作物蒸腾量Tr
    Tr = FW1.*1.1.*CC1;  % Tr 作物蒸腾量   CC1 调整后的冠层覆盖度
%      %% 地上生物量B——水分驱动
%      data = xlsread('matlab地上生物量.xlsx');
%      FCO2  = data(:,1); % 二氧化碳影响因子
%      FTB = data(:,2); % 温度影响因子
%      B0 = 19.*FTB.*FCO2.*Tr; % 每日积累的生物量
%      B = 0.01*cumsum(B0,1); % 每日的生物量 ton/ha
     %% 地上生物量——光能驱动
     %data = xlsread('matlab干物质积累光能驱动.xlsx');
     PDT = load('PDT.mat');
     PDT = struct2array(PDT);
     I01 = load('I01.mat');
     I01 = struct2array(I01);
     I02 = load('I02.mat');
     I02 = struct2array(I02);
     I03 = load('I03.mat');
     I03 = struct2array(I03);
     P1 = load('P1.mat');
     P1 = struct2array(P1);
     P2 = load('P2.mat');
     P2 = struct2array(P2);
     P3 = load('P3.mat');
     P3 = struct2array(P3);
     K = load('K.mat');
     K = struct2array(K);
     FGWZCO2 = load('FGWZCO2.mat');
     FGWZCO2 = struct2array(FGWZCO2);
     FA = load('FA.mat');
     FA = struct2array(FA);
     FTGWZ = load('FTGWZ.mat');
     FTGWZ = struct2array(FTGWZ);
     DL = load('DL.mat');
     DL = struct2array(DL);
%      PDT  = data(:,1); % PDT 生理发育时间
%      I01  = data(:,2); % IO1 第1时刻到达冠层顶部的瞬时光合有效辐射，J*m-2*s-1
%      I02  = data(:,3); % IO2 第2时刻到达冠层顶部的瞬时光合有效辐射，J*m-2*s-1
%      I03  = data(:,4); % IO3 第3时刻到达冠层顶部的瞬时光合有效辐射，J*m-2*s-1
%      P1  = data(:,5); % P1 第1时刻冠层反射率
%      P2  = data(:,6); % P2 第2时刻冠层反射率
%      P3  = data(:,7); % P3 第3时刻冠层反射率
%      K = data(:,8); %  K  冠层消光系数
%      FGWZCO2 = data(:,9); % FGWZCO2 二氧化碳影响因子
%      FA = data(:,11); % FA 生理年龄影响函数
%      FTGWZ = data(:,12); % FTGWZ 干物质积累温度影响因子
%           DL = data(:,13); % DL 日长
     LGUSS1 = 0.0469*LAI; % 冠层第1层的叶面积指数
     LGUSS2 = 0.2308*LAI; % 冠层第2层的叶面积指数
     LGUSS3 = 0.5*LAI;    % 冠层第3层的叶面积指数
     LGUSS4 = 0.7691*LAI; % 冠层第4层的叶面积指数
     LGUSS5 = 0.9531*LAI; % 冠层第5层的叶面积指数
     I11 =K.*(1-P1).*I01.*exp(-K.*LGUSS1); % I11 冠层的第1层在第1时刻吸收的光合有效辐射，J*m-2*s-1
     I12 =K.*(1-P1).*I01.*exp(-K.*LGUSS2); % I12 冠层的第2层在第1时刻吸收的光合有效辐射，J*m-2*s-1
     I13 =K.*(1-P1).*I01.*exp(-K.*LGUSS3); % I13 冠层的第3层在第1时刻吸收的光合有效辐射，J*m-2*s-1
     I14 =K.*(1-P1).*I01.*exp(-K.*LGUSS4); % I14 冠层的第4层在第1时刻吸收的光合有效辐射，J*m-2*s-1
     I15 =K.*(1-P1).*I01.*exp(-K.*LGUSS5); % I15 冠层的第5层在第1时刻吸收的光合有效辐射，J*m-2*s-1
     I21 =K.*(1-P2).*I02.*exp(-K.*LGUSS1); % I21 冠层的第1层在第2时刻吸收的光合有效辐射，J*m-2*s-1
     I22 =K.*(1-P2).*I02.*exp(-K.*LGUSS2); % I22 冠层的第2层在第2时刻吸收的光合有效辐射，J*m-2*s-1
     I23 =K.*(1-P2).*I02.*exp(-K.*LGUSS3); % I23 冠层的第3层在第2时刻吸收的光合有效辐射，J*m-2*s-1
     I24 =K.*(1-P2).*I02.*exp(-K.*LGUSS4); % I24 冠层的第4层在第2时刻吸收的光合有效辐射，J*m-2*s-1
     I25 =K.*(1-P2).*I02.*exp(-K.*LGUSS5); % I25 冠层的第5层在第2时刻吸收的光合有效辐射，J*m-2*s-1
     I31 =K.*(1-P3).*I03.*exp(-K.*LGUSS1); % I31 冠层的第1层在第3时刻吸收的光合有效辐射，J*m-2*s-1
     I32 =K.*(1-P3).*I03.*exp(-K.*LGUSS2); % I32 冠层的第2层在第3时刻吸收的光合有效辐射，J*m-2*s-1
     I33 =K.*(1-P3).*I03.*exp(-K.*LGUSS3); % I33 冠层的第3层在第3时刻吸收的光合有效辐射，J*m-2*s-1
     I34 =K.*(1-P3).*I03.*exp(-K.*LGUSS4); % I34 冠层的第4层在第3时刻吸收的光合有效辐射，J*m-2*s-1
     I35 =K.*(1-P3).*I03.*exp(-K.*LGUSS5); % I35 冠层的第5层在第3时刻吸收的光合有效辐射，J*m-2*s-1
%     
     MAX = 54; % MAX 叶片处于最适的生理年龄、温度、营养等条件下的最大光合速率
     AMAX =MAX.*FGWZCO2.*FA.*FTGWZ.*FW1 % AMAX 饱和光强时的光合作用速率
     %单叶光合作用
     P11 = AMAX.*(1-exp(-0.54.*I11./AMAX)); % 第1时刻、冠层第1层的单叶光合作用速率
     P12 = AMAX.*(1-exp(-0.54.*I12./AMAX)); % 第1时刻、冠层第2层的单叶光合作用速率
     P13 = AMAX.*(1-exp(-0.54.*I13./AMAX)); % 第1时刻、冠层第3层的单叶光合作用速率
     P14 = AMAX.*(1-exp(-0.54.*I14./AMAX)); % 第1时刻、冠层第4层的单叶光合作用速率
     P15 = AMAX.*(1-exp(-0.54.*I15./AMAX)); % 第1时刻、冠层第5层的单叶光合作用速率
     P21 = AMAX.*(1-exp(-0.54.*I21./AMAX)); % 第2时刻、冠层第1层的单叶光合作用速率
     P22 = AMAX.*(1-exp(-0.54.*I22./AMAX)); % 第2时刻、冠层第2层的单叶光合作用速率
     P23 = AMAX.*(1-exp(-0.54.*I23./AMAX)); % 第2时刻、冠层第3层的单叶光合作用速率
     P24 = AMAX.*(1-exp(-0.54.*I24./AMAX)); % 第2时刻、冠层第4层的单叶光合作用速率
     P25 = AMAX.*(1-exp(-0.54.*I25./AMAX)); % 第2时刻、冠层第5层的单叶光合作用速率
     P31 = AMAX.*(1-exp(-0.54.*I31./AMAX)); % 第3时刻、冠层第1层的单叶光合作用速率
     P32 = AMAX.*(1-exp(-0.54.*I32./AMAX)); % 第3时刻、冠层第2层的单叶光合作用速率
     P33 = AMAX.*(1-exp(-0.54.*I33./AMAX)); % 第3时刻、冠层第3层的单叶光合作用速率
     P34 = AMAX.*(1-exp(-0.54.*I34./AMAX)); % 第3时刻、冠层第4层的单叶光合作用速率
     P35 = AMAX.*(1-exp(-0.54.*I35./AMAX)); % 第3时刻、冠层第5层的单叶光合作用速率
     %冠层光合作用
     TP1 =(P11.*0.1185+P12.*0.2393+P13.*0.2844+P14.*0.2393+P15.*0.1185).*LAI; % 第1个时刻整个冠层的瞬时光合作用速率
     TP2 =(P21.*0.1185+P22.*0.2393+P23.*0.2844+P24.*0.2393+P25.*0.1185).*LAI; % 第2个时刻整个冠层的瞬时光合作用速率
     TP3 =(P31.*0.1185+P32.*0.2393+P33.*0.2844+P34.*0.2393+P35.*0.1185).*LAI; % 第3个时刻整个冠层的瞬时光合作用速率
     DTGA = (TP1.*0.2778+TP2.*0.4444+TP3.*0.2778).*0.682.*DL; % DTGA 整个冠层每日的总光合作用速率 kg*CH2O*hm-2*d-1
     %呼吸作用
     %维持呼吸
     RMT0 = 0.0091-0.0001.*PDT; % RMT0 温度为25度时的维持呼吸速率
     Q10 = load('Q10.mat');
     Q10 = struct2array(Q10);
%      Q10  = data(:,14);      % 维持呼吸速率
          %生长呼吸
    Rg = 0.20; % 生长呼吸系数
    RG = Rg.*DTGA; % RG 生长呼吸消耗量 kg*CH2O*hm-2*d-1
     
     ABIOMASS0 = 11 % ABIOMASS0 初始地上生物量 kg/ha
     
     ABIOMASS0 = [11; zeros(130, 1)];  % 初始猜测ABIOMASS0，只有第一项已知为11
     
     ABIOMASS1 = zeros(131, 1); % 第i天的地上生物量 kg/hm-2
     
     for i = 1:131
         % 计算 A1
         A1 = fsolve(@(A1) A1 - ABIOMASS0(i) - (DTGA(i)-RMT0(i).*ABIOMASS0(i).*Q10(i)-RG(i))./0.95, 0);
         
         % 将 A1 放入 ABIOMASS0 中
                  ABIOMASS0(i + 1) = A1;
         
         % 将 A1 放入 ABIOMASS1 中
         ABIOMASS1(i) = A1;
     end
          disp('计算得到的ABIOMASS1:');
     disp(ABIOMASS1);
     ABIOMASS1 = 0.001.*ABIOMASS1; %t/hm-2
    %% 目标1——产量Y
    %潜在生长量计算
    PAR = 0.5.*RA.*(1-exp(-0.65.*LAI));
    Bp = 0.001.*WA.*PAR;
    %实际生长量计算
    Ba = Bp.*min(FW,FT);
    %产量计算
    HIa = HIp/(1+WSYF*(0.9-FW(131,1))*max(0,sin((3.14/2)*((HUI(131,1)-0.3)/0.3))));
    y_values(j)= A*HIa*sum(Ba);  %kg/ha
    y = y_values;
    vn(j) = v;
    %% 目标2——碳排放(Ws = 0.*(h<-148871.9)+r.*(log(-148871.9/h)/log(-148871.9/-420)).*(h>=-148871.9 & h<-420)+1.*(h>=-420);)
    %土壤呼吸
      Q110 = a-b.*Td+c.*WWW1+d.*(WWW1.^2); 
      Ts = Q110.^((Td-Tf)/10);
      RH = 44/12*SOC.*k.*Ts.*Ws;
 %     RH = 0.044.*exp(0.003*Td+0.001.*WWW1);
     Cs = A.*RH;
     CS =1000*sum(Cs);  %单位为t CO2eq
    %植株固碳
    Cd1 = 44/12*Cf*ABIOMASS1(131,1);  %光能驱动固碳量，单位为t CO2eq
%      Cd2 = 44/12*Cf*B(131,1);  %水分胁迫下固碳量，单位为t CO2eq
    Dw = 1./y/Hi;
    Cd3 = 44/12*Cf*Dw*1000;  %计算的Dw固碳量，单位为t CO2eq%
    %土壤固碳
    Ct = dr*H*CSOC*A;
    C0 = dr*H*CSOC0*A;
    C = 44/12*(Ct-C0)/(t-t0);  %单位为 g C
    %碳排放求和
     C1(j)= Cz+CH4+NO2+CS-Cd1-C;
%      C2(j) = Cz+CH4+NO2+CS-Cd2-C;
    C3(j) = Cz+CH4+NO2+CS-Cd3-C;
    %% 灌水量
    I1 = sum(I);
    I0(j) = 189+I1;
end
disp(CS);
disp(Cd1);
disp(Cd3);
disp('输入的v值:');
disp(vn);
disp('计算得到的y值:');
disp(y_values);
% disp(C1);
% disp(C2);
disp(C3);
disp(I0);
disp(I);
%% 脚本函数(局部函数必须放在脚本函数最后面)
% 定义E函数 % 土壤蒸发量
    function E_result = E(ETO, W1)
        Kr = min(1, max(0, (exp(0.4 * W1 / 22) - 1) / (exp(0.4) - 1)));
        E_result = Kr * ETO;
    end

% 定义L函数 % 土壤含水量的减少率
    function L_result = L(W1)
        L_result = min(1, max(0, (exp(W1 - 20) - 1) / (exp(2) - 1)));
    end

% 定义DP函数  % 渗漏量，mm
    function DP_result = DP(Z, W1)
        DP_result = 0.476 * 0.04468 * Z * L(W1);
    end

%% 计算目标函数值
f1 = 1./y;
f11 =  real(f1);
f3 =C3;
f33 = real(f3);
f2 = I0;
f22 = real(f2);    
    %% 检查约束条件
%     cons = 0;
%     % 水量约束
%     if sum(Q) > 11070000
%        cons = cons + sum(Q)-11070000;
%     end
%     %渠道供水能力约束(不太明白)
%     Q_ganqu_t = Q_t_shou(:,1);
%     cons = cons + sum(max(Q_ganqu_t-2.5*Ju,0)) + sum(max(2.5*Jd-Q_ganqu_t,0));
    
    
   
z = [f11;f22;f33];
%     
%  end
%   pop.Cost=z;

 end

