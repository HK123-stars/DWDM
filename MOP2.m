%
% Ŀ�꺯��
%
function z = MOP2(x)
   
  %  ��ʼ����
 %% ˮ��ƽ�ⷽ��
% data = xlsread('ˮ��ƽ��.xlsx');
P = load('P.mat');
P = struct2array(P); % ���ṹ��ת��Ϊ����
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
% P  = data(:,3); % ��Ч��������mm
% ETO = data(:,6); % �ο�������������mm
% Ke = data(:,20);  %��������ϵ��
% T = data(:,8); % ������ɢ������mm
% Z  = data(:,10); % ���������仯����mm����©��
% CR = data(:,12); % ëϸ����������mm������ˮ��������
%%  ����
% PAR = data(:,13);  %��Ч��Ϸ��䣨��10-6��
RA = load('RA.mat');
RA = struct2array(RA);
FT = load('FT.mat');
FT = struct2array(FT);
HUI = load('HUI.mat');
HUI = struct2array(HUI);
% RA = data(:,14);  %̫������
% FT = data(:,15);  %�¶�в��
% HUI = data(:,16);  %��������λָ��
WA = 50;%50; %����-������ת������
HIp = 0.48;%0.48;  %Ǳ���ջ�ָ��
Hi = 0.45;  %����ϵ��
WSYF =0.5 ;  %�ɺ���������
%%  ̼�ŷ�
Cz = 1957.831;  %(������ֲ������̼�ŷţ�kg/ha)
CH4 = 11454.576;  %CH4�����̼�ŷ�(kg/ha)
NO2 = 1594.622;  %NO2�ŷţ�kg/ha��
SOC = 0.753;  %�����л�̼��g/m2(40cm*1.43g/cm3*0.01316��
k = 0.014;  %����̼����ת���ʣ�day
Tf = 25; %���������ο��¶�
hmin = -14887;  %���ˮ��
hmax = -100;  %���ˮ��
 a = 1.29;  %����1.29
b = -1.05;  %����-1.05
c = 1.36 ;  %����1.36
d = -1.05;   %-1.05;  %����
A = 1;  %���
Td = load('Td.mat');
Td = struct2array(Td);
r = load('r.mat');
r = struct2array(r);
Ws = load('Ws.mat');
Ws = struct2array(Ws);
% Td = data(:,17);  %�����¶�
% r = data(:,18);  %���ֲ�����
% Ws = data(:,19);  %ʪ��Ӱ��ϵ��
Cf = 0.41;  %����ϳ�1g������Ҫ���յ�̼��tCO2eq��
CSOC = 1.316;  %�����л�̼������%��
CSOC0 = 1.0415;  %��ʼ�����л�̼������%��
dr = 1.36;  %����(g/cm3)
H = 40;  %��ȣ�cm��
t = 1;  %��ݣ�a��
t0 = 0;  %�������
n = 1; % ����������Ҫ10�����x�Ͷ�Ӧ��y
y_values = zeros(n, 1); % ��ʼ��yֵ����
for j = 1:n
    % �������x
      %v = round(rand*(20-6)+6, 1);
      v = 15.27031451;
    WW0 = [72; zeros(131, 1)];  % ��ʼ�²�WW0��ֻ�е�һ����֪Ϊ72    ��i-1��ĺ�ˮ����mm������ˮ�㣩
    WW1 = zeros(131, 1); % ��i��ĺ�ˮ��������ˮ�㣩
    I = zeros(131, 1);   % ��i��������ڹ�ˮ��
    for i = 1:51
    % ���� W1
    W1 = fsolve(@(W1) W1 - WW0(i) - I(i) - P(i) + Ke(i) * E(ETo(i), W1) + T(i) + DP(Z(i), W1) - CR(i), 0);
    ET(i) =  Ke(i) * E(ETo(i), W1) + T(i);
    % �� W1 ���� WW0 ��
    WW0(i + 1) = W1;
    
    % �� W1 ���� WW1 ��
    WW1(i) = W1;
    % ����WW1(i)��x����I(i)��ֵ
    if WW1(i) > v
        I(i+1) = 0;
    else 
        I(i+1) = 42 - WW1(i);
    end
    if P(i)>10;
        I(i-1)=0;
    end
    end
    
%     disp('����õ���WW1:');
%     disp(WW1);
    WW1(find(WW1>72))=72;
    %% ˮ��в��
     WWW1 = 22.*(WW1>22)+WW1.*(WW1>0 & WW1<=22) +0.*(WW1<0); %  WWW1 ������������ˮ����mm
%     WWW1 = double(WW1 >= 22) .* 22 + double(0 <= WW1 & WW1 < 22) .* WW1 + double(WW1 < 0) .* 0;
    WWW1(find(WWW1<6))=6 ;% Լ��
    h = -420*((WWW1-2.68)/(22-2.68)).^(-1/0.3); % h ����������mm
    FW=0.*(h<=-14887|h>0)+((h+0)/(-450+0)).*(h>-450 & h<=0)+1.*(h>-4770 & h<=-450)+((h+14887)/(-4770+14887)).*(h>-14887 & h<=-4770);
%     FW = double(h >= 0 | h <=-14887) .* 0 + double(-420 <= h & h < 0) .* (h-0)/(-420-0) +  double(-6000 <= h & h < 420) .* (h+6000)/(-420+6000)+double(-14887 <= h & h < -6000) .* (h+14887)/(-6000+14887);
      % FW ˮ��в��(���ԼӸ���ˮ������)
     FW1=0.*(WWW1<=6)+((WWW1-6)/(22-6)).*(WWW1>6 & WWW1<=22)+1.*(WWW1>22);
%     FW1 = double(WWW1 >= 22) .* 1 + double(6 <= WWW1 & WWW1 < 22) .* (WWW1-6)/(22-6) + double(WWW1 < 6) .* 0;
    %  FW ˮ��в��(���ԼӸ���ˮ������)
    %% �ڲ㸲�Ƕ�CC
%     data = xlsread('matlab�ڲ㸲�Ƕ�.xlsx');
    CC0 = load('CC0.mat');
    CC0 = struct2array(CC0);
    FTC = load('FTC.mat');
    FTC = struct2array(FTC);
    CCD0 = load('CCD0.mat');
    CCD0 = struct2array(CCD0);
%     CC0 = data(:,1); % Ǳ�ڹڲ㸲�Ƕ�
%     FTC = data(:,2); % �ڲ��¶�в������
%     CCD0 = data(:,10); % �ڲ�Ǳ�ڱ仯��
    CGC = min(FW(1:97,1:1),FTC(1:97,1:1));         % �ڲ�����в������
    CDC = 1-(min(FW(98:131,1:1),FTC(98:131,1:1)).^8);  % �ڲ�˥��в������
    CCD = CCD0.*[CGC;CDC]; %�ڲ�ʵ�ʱ仯��
    CC = cumsum(CCD,1);    %ʵ�ʹڲ㸲�Ƕ�
    CC1 = 1.72.*CC-CC.*CC+0.3.*((CC).^3); %������Ĺڲ㸲�Ƕ�
    LAI =-log(1-((CC1).^(1/1.2))/1.005)/0.6;                      % LAI ʵ��Ҷ���ָ��
    LAI0 = load('LAI0.mat');
    LAI0 = struct2array(LAI0);
    LAI1 = load('LAI1.mat');
    LAI1 = struct2array(LAI1);
%     LAI0 = data(:,8); % Ǳ��Ҷ���ָ��
%     LAI1 = data(:,9); % Ǳ��Ҷ����������̱仯��
    LAI2 = sum(LAI1.*FW);  %������ʵ��Ҷ���ָ��
    %% ����������Tr
    Tr = FW1.*1.1.*CC1;  % Tr ����������   CC1 ������Ĺڲ㸲�Ƕ�
%      %% ����������B����ˮ������
%      data = xlsread('matlab����������.xlsx');
%      FCO2  = data(:,1); % ������̼Ӱ������
%      FTB = data(:,2); % �¶�Ӱ������
%      B0 = 19.*FTB.*FCO2.*Tr; % ÿ�ջ��۵�������
%      B = 0.01*cumsum(B0,1); % ÿ�յ������� ton/ha
     %% ����������������������
     %data = xlsread('matlab�����ʻ��۹�������.xlsx');
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
%      PDT  = data(:,1); % PDT ������ʱ��
%      I01  = data(:,2); % IO1 ��1ʱ�̵���ڲ㶥����˲ʱ�����Ч���䣬J*m-2*s-1
%      I02  = data(:,3); % IO2 ��2ʱ�̵���ڲ㶥����˲ʱ�����Ч���䣬J*m-2*s-1
%      I03  = data(:,4); % IO3 ��3ʱ�̵���ڲ㶥����˲ʱ�����Ч���䣬J*m-2*s-1
%      P1  = data(:,5); % P1 ��1ʱ�̹ڲ㷴����
%      P2  = data(:,6); % P2 ��2ʱ�̹ڲ㷴����
%      P3  = data(:,7); % P3 ��3ʱ�̹ڲ㷴����
%      K = data(:,8); %  K  �ڲ�����ϵ��
%      FGWZCO2 = data(:,9); % FGWZCO2 ������̼Ӱ������
%      FA = data(:,11); % FA ��������Ӱ�캯��
%      FTGWZ = data(:,12); % FTGWZ �����ʻ����¶�Ӱ������
%           DL = data(:,13); % DL �ճ�
     LGUSS1 = 0.0469*LAI; % �ڲ��1���Ҷ���ָ��
     LGUSS2 = 0.2308*LAI; % �ڲ��2���Ҷ���ָ��
     LGUSS3 = 0.5*LAI;    % �ڲ��3���Ҷ���ָ��
     LGUSS4 = 0.7691*LAI; % �ڲ��4���Ҷ���ָ��
     LGUSS5 = 0.9531*LAI; % �ڲ��5���Ҷ���ָ��
     I11 =K.*(1-P1).*I01.*exp(-K.*LGUSS1); % I11 �ڲ�ĵ�1���ڵ�1ʱ�����յĹ����Ч���䣬J*m-2*s-1
     I12 =K.*(1-P1).*I01.*exp(-K.*LGUSS2); % I12 �ڲ�ĵ�2���ڵ�1ʱ�����յĹ����Ч���䣬J*m-2*s-1
     I13 =K.*(1-P1).*I01.*exp(-K.*LGUSS3); % I13 �ڲ�ĵ�3���ڵ�1ʱ�����յĹ����Ч���䣬J*m-2*s-1
     I14 =K.*(1-P1).*I01.*exp(-K.*LGUSS4); % I14 �ڲ�ĵ�4���ڵ�1ʱ�����յĹ����Ч���䣬J*m-2*s-1
     I15 =K.*(1-P1).*I01.*exp(-K.*LGUSS5); % I15 �ڲ�ĵ�5���ڵ�1ʱ�����յĹ����Ч���䣬J*m-2*s-1
     I21 =K.*(1-P2).*I02.*exp(-K.*LGUSS1); % I21 �ڲ�ĵ�1���ڵ�2ʱ�����յĹ����Ч���䣬J*m-2*s-1
     I22 =K.*(1-P2).*I02.*exp(-K.*LGUSS2); % I22 �ڲ�ĵ�2���ڵ�2ʱ�����յĹ����Ч���䣬J*m-2*s-1
     I23 =K.*(1-P2).*I02.*exp(-K.*LGUSS3); % I23 �ڲ�ĵ�3���ڵ�2ʱ�����յĹ����Ч���䣬J*m-2*s-1
     I24 =K.*(1-P2).*I02.*exp(-K.*LGUSS4); % I24 �ڲ�ĵ�4���ڵ�2ʱ�����յĹ����Ч���䣬J*m-2*s-1
     I25 =K.*(1-P2).*I02.*exp(-K.*LGUSS5); % I25 �ڲ�ĵ�5���ڵ�2ʱ�����յĹ����Ч���䣬J*m-2*s-1
     I31 =K.*(1-P3).*I03.*exp(-K.*LGUSS1); % I31 �ڲ�ĵ�1���ڵ�3ʱ�����յĹ����Ч���䣬J*m-2*s-1
     I32 =K.*(1-P3).*I03.*exp(-K.*LGUSS2); % I32 �ڲ�ĵ�2���ڵ�3ʱ�����յĹ����Ч���䣬J*m-2*s-1
     I33 =K.*(1-P3).*I03.*exp(-K.*LGUSS3); % I33 �ڲ�ĵ�3���ڵ�3ʱ�����յĹ����Ч���䣬J*m-2*s-1
     I34 =K.*(1-P3).*I03.*exp(-K.*LGUSS4); % I34 �ڲ�ĵ�4���ڵ�3ʱ�����յĹ����Ч���䣬J*m-2*s-1
     I35 =K.*(1-P3).*I03.*exp(-K.*LGUSS5); % I35 �ڲ�ĵ�5���ڵ�3ʱ�����յĹ����Ч���䣬J*m-2*s-1
%     
     MAX = 54; % MAX ҶƬ�������ʵ��������䡢�¶ȡ�Ӫ���������µ����������
     AMAX =MAX.*FGWZCO2.*FA.*FTGWZ.*FW1 % AMAX ���͹�ǿʱ�Ĺ����������
     %��Ҷ�������
     P11 = AMAX.*(1-exp(-0.54.*I11./AMAX)); % ��1ʱ�̡��ڲ��1��ĵ�Ҷ�����������
     P12 = AMAX.*(1-exp(-0.54.*I12./AMAX)); % ��1ʱ�̡��ڲ��2��ĵ�Ҷ�����������
     P13 = AMAX.*(1-exp(-0.54.*I13./AMAX)); % ��1ʱ�̡��ڲ��3��ĵ�Ҷ�����������
     P14 = AMAX.*(1-exp(-0.54.*I14./AMAX)); % ��1ʱ�̡��ڲ��4��ĵ�Ҷ�����������
     P15 = AMAX.*(1-exp(-0.54.*I15./AMAX)); % ��1ʱ�̡��ڲ��5��ĵ�Ҷ�����������
     P21 = AMAX.*(1-exp(-0.54.*I21./AMAX)); % ��2ʱ�̡��ڲ��1��ĵ�Ҷ�����������
     P22 = AMAX.*(1-exp(-0.54.*I22./AMAX)); % ��2ʱ�̡��ڲ��2��ĵ�Ҷ�����������
     P23 = AMAX.*(1-exp(-0.54.*I23./AMAX)); % ��2ʱ�̡��ڲ��3��ĵ�Ҷ�����������
     P24 = AMAX.*(1-exp(-0.54.*I24./AMAX)); % ��2ʱ�̡��ڲ��4��ĵ�Ҷ�����������
     P25 = AMAX.*(1-exp(-0.54.*I25./AMAX)); % ��2ʱ�̡��ڲ��5��ĵ�Ҷ�����������
     P31 = AMAX.*(1-exp(-0.54.*I31./AMAX)); % ��3ʱ�̡��ڲ��1��ĵ�Ҷ�����������
     P32 = AMAX.*(1-exp(-0.54.*I32./AMAX)); % ��3ʱ�̡��ڲ��2��ĵ�Ҷ�����������
     P33 = AMAX.*(1-exp(-0.54.*I33./AMAX)); % ��3ʱ�̡��ڲ��3��ĵ�Ҷ�����������
     P34 = AMAX.*(1-exp(-0.54.*I34./AMAX)); % ��3ʱ�̡��ڲ��4��ĵ�Ҷ�����������
     P35 = AMAX.*(1-exp(-0.54.*I35./AMAX)); % ��3ʱ�̡��ڲ��5��ĵ�Ҷ�����������
     %�ڲ�������
     TP1 =(P11.*0.1185+P12.*0.2393+P13.*0.2844+P14.*0.2393+P15.*0.1185).*LAI; % ��1��ʱ�������ڲ��˲ʱ�����������
     TP2 =(P21.*0.1185+P22.*0.2393+P23.*0.2844+P24.*0.2393+P25.*0.1185).*LAI; % ��2��ʱ�������ڲ��˲ʱ�����������
     TP3 =(P31.*0.1185+P32.*0.2393+P33.*0.2844+P34.*0.2393+P35.*0.1185).*LAI; % ��3��ʱ�������ڲ��˲ʱ�����������
     DTGA = (TP1.*0.2778+TP2.*0.4444+TP3.*0.2778).*0.682.*DL; % DTGA �����ڲ�ÿ�յ��ܹ���������� kg*CH2O*hm-2*d-1
     %��������
     %ά�ֺ���
     RMT0 = 0.0091-0.0001.*PDT; % RMT0 �¶�Ϊ25��ʱ��ά�ֺ�������
     Q10 = load('Q10.mat');
     Q10 = struct2array(Q10);
%      Q10  = data(:,14);      % ά�ֺ�������
          %��������
    Rg = 0.20; % ��������ϵ��
    RG = Rg.*DTGA; % RG �������������� kg*CH2O*hm-2*d-1
     
     ABIOMASS0 = 11 % ABIOMASS0 ��ʼ���������� kg/ha
     
     ABIOMASS0 = [11; zeros(130, 1)];  % ��ʼ�²�ABIOMASS0��ֻ�е�һ����֪Ϊ11
     
     ABIOMASS1 = zeros(131, 1); % ��i��ĵ��������� kg/hm-2
     
     for i = 1:131
         % ���� A1
         A1 = fsolve(@(A1) A1 - ABIOMASS0(i) - (DTGA(i)-RMT0(i).*ABIOMASS0(i).*Q10(i)-RG(i))./0.95, 0);
         
         % �� A1 ���� ABIOMASS0 ��
                  ABIOMASS0(i + 1) = A1;
         
         % �� A1 ���� ABIOMASS1 ��
         ABIOMASS1(i) = A1;
     end
          disp('����õ���ABIOMASS1:');
     disp(ABIOMASS1);
     ABIOMASS1 = 0.001.*ABIOMASS1; %t/hm-2
    %% Ŀ��1��������Y
    %Ǳ������������
    PAR = 0.5.*RA.*(1-exp(-0.65.*LAI));
    Bp = 0.001.*WA.*PAR;
    %ʵ������������
    Ba = Bp.*min(FW,FT);
    %��������
    HIa = HIp/(1+WSYF*(0.9-FW(131,1))*max(0,sin((3.14/2)*((HUI(131,1)-0.3)/0.3))));
    y_values(j)= A*HIa*sum(Ba);  %kg/ha
    y = y_values;
    vn(j) = v;
    %% Ŀ��2����̼�ŷ�(Ws = 0.*(h<-148871.9)+r.*(log(-148871.9/h)/log(-148871.9/-420)).*(h>=-148871.9 & h<-420)+1.*(h>=-420);)
    %��������
      Q110 = a-b.*Td+c.*WWW1+d.*(WWW1.^2); 
      Ts = Q110.^((Td-Tf)/10);
      RH = 44/12*SOC.*k.*Ts.*Ws;
 %     RH = 0.044.*exp(0.003*Td+0.001.*WWW1);
     Cs = A.*RH;
     CS =1000*sum(Cs);  %��λΪt CO2eq
    %ֲ���̼
    Cd1 = 44/12*Cf*ABIOMASS1(131,1);  %����������̼������λΪt CO2eq
%      Cd2 = 44/12*Cf*B(131,1);  %ˮ��в���¹�̼������λΪt CO2eq
    Dw = 1./y/Hi;
    Cd3 = 44/12*Cf*Dw*1000;  %�����Dw��̼������λΪt CO2eq%
    %������̼
    Ct = dr*H*CSOC*A;
    C0 = dr*H*CSOC0*A;
    C = 44/12*(Ct-C0)/(t-t0);  %��λΪ g C
    %̼�ŷ����
     C1(j)= Cz+CH4+NO2+CS-Cd1-C;
%      C2(j) = Cz+CH4+NO2+CS-Cd2-C;
    C3(j) = Cz+CH4+NO2+CS-Cd3-C;
    %% ��ˮ��
    I1 = sum(I);
    I0(j) = 189+I1;
end
disp(CS);
disp(Cd1);
disp(Cd3);
disp('�����vֵ:');
disp(vn);
disp('����õ���yֵ:');
disp(y_values);
% disp(C1);
% disp(C2);
disp(C3);
disp(I0);
disp(I);
%% �ű�����(�ֲ�����������ڽű����������)
% ����E���� % ����������
    function E_result = E(ETO, W1)
        Kr = min(1, max(0, (exp(0.4 * W1 / 22) - 1) / (exp(0.4) - 1)));
        E_result = Kr * ETO;
    end

% ����L���� % ������ˮ���ļ�����
    function L_result = L(W1)
        L_result = min(1, max(0, (exp(W1 - 20) - 1) / (exp(2) - 1)));
    end

% ����DP����  % ��©����mm
    function DP_result = DP(Z, W1)
        DP_result = 0.476 * 0.04468 * Z * L(W1);
    end

%% ����Ŀ�꺯��ֵ
f1 = 1./y;
f11 =  real(f1);
f3 =C3;
f33 = real(f3);
f2 = I0;
f22 = real(f2);    
    %% ���Լ������
%     cons = 0;
%     % ˮ��Լ��
%     if sum(Q) > 11070000
%        cons = cons + sum(Q)-11070000;
%     end
%     %������ˮ����Լ��(��̫����)
%     Q_ganqu_t = Q_t_shou(:,1);
%     cons = cons + sum(max(Q_ganqu_t-2.5*Ju,0)) + sum(max(2.5*Jd-Q_ganqu_t,0));
    
    
   
z = [f11;f22;f33];
%     
%  end
%   pop.Cost=z;

 end

