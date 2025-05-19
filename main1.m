%% Recover a single image and show it
close all; clear; clc
%% Initialization
PSNR = [];
peaksnr1_max=[];
peaksnr2_max=[];
peaksnr3_max=[];
peaksnr4_max=[];
peaksnr5_max=[];
peaksnr6_max=[];
peaksnr7_max=[];
peaksnr8_max=[];
peaksnr9_max=[];
peaksnr10_max=[];
peaksnr11_max=[];
peaksnr12_max=[];
peaksnr13_max=[];
peaksnr14_max=[];
ssim1 = [];
ssim2 = [];
ssim3 = [];
ssim4 = [];
ssim5 = [];
ssim6 = [];
ssim7 = [];
ssim8 = [];
ssim9 = [];
ssim10 = [];
ssim11 = [];
ssim12 = [];
ssim13 = [];
ssim14 = [];
t_1=[];
t_2=[];
t_3=[];
t_4=[];
t_5=[];
t_6=[];
t_7=[];
t_8=[];
t_9=[];
t_10=[];
t_11=[];
t_12=[];
t_13=[];
t_14=[];
RMSE = [];
RMSE1 = [];
RMSE2 = [];
RMSE3 = [];
RMSE4 = [];
RMSE5 = [];
RMSE6 = [];
RMSE7 = [];
RMSE8 = [];
RMSE9 = [];
RMSE10 = [];
RMSE11 = [];
RMSE12 = [];
RMSE13 = [];
RMSE14 = [];
PSNR_SVD = [];
RMSE_SVD = [];
rank1 = [];
rank2 = [];
rank3 = [];
rank4 = [];
rank5 = [];
rank6 = [];
rank7 = [];
rank8 = [];
rank9 = [];
rank10 = [];
rank11 = [];
rank12 = [];
rank13 = [];
rank14 = [];
PSNR_tot = zeros(1,12);
SSIM_tot = zeros(1,12);
RMSE_tot = zeros(1,12);
Time_tot = zeros(1,12);

%% Hyperparameters
K = 5; % Number of Monte Carlo
rak = 10;
maxiter = 50;
p = 1; % l_p norm

xi1 = 7; % t-Welsch hyperparameters
xi2 = 7;
%% Import data
image = imread('.\ZJU\7.jpg');
[width,height,z]=size(image);
if(z>1)
    image=rgb2gray(image);
end
% unit8 to double
image = mat2gray(image);
[m,n] = size(image);
M = image;
real_rank1=rank(M);

%% Robust MC methods
for kk=1:K % Monte Carlo

    % Noise
    SNR = 10;
    M_noise = imnoise(M, 'salt & pepper', 0.1);
    % figure
    % imshow(M_noise)
    M_noise = imnoise(M_noise, 'gaussian', 0, 0.0001);
    % figure
    % imshow(M_noise)

    per = 0.8;
    array_Omega = binornd( 1, per, [ m, n ] );
    M_Omega = M_noise.* array_Omega;
    % figure
    % imshow(M_Omega)

    %% AROMC-HOW  unknown rank/robust
    tic
    [X_1,Y1_RMSE1,real_RMSE,peaksnr1,U1,V1,NRE1,PMD] = RAR1MC(M, M_Omega, array_Omega, maxiter, xi1, xi2);
    toc;
    t_1=[t_1 toc];
    peaksnr1_max = [peaksnr1_max psnr(X_1, M)];
    ssim1 = [ssim1 ssim(X_1, M)];
    RMSE1 = [RMSE1 norm((M - X_1),'fro')/sqrt(m*n)];
    rank1 = [rank1 rank(X_1)];

    %% l1-ADMM  known rank/robust
    addpath(genpath('L1-ADMM'));
    mu = 5;
    tic
    [X_2, RMSE_2,T_iter] = ADMM(M, M_Omega, array_Omega, rak, maxiter, p, mu);
    toc;
    t_2= [t_2 toc];
    peaksnr2_max = [peaksnr2_max psnr(X_2, M)];
    ssim2 = [ssim2 ssim(X_2, M)];
    RMSE2 = [RMSE2 norm((M - X_2),'fro')/sqrt(m*n)];
    rank2 = [rank2 rank(X_2)];

    %% l1-reg  known rank/robust
    addpath(genpath('L1-reg'));
    tic
    [X_3,RMSE_3,~,~] = LP1(M, M_Omega, array_Omega, rak, maxiter, p);
    toc;
    t_3 = [t_3 toc];
    peaksnr3_max = [peaksnr3_max psnr(X_3, M)];
    ssim3 = [ssim3 ssim(X_3, M)];
    RMSE3 = [RMSE3 norm((M - X_3),'fro')/sqrt(m*n)];
    rank3 = [rank3 rank(X_3)];

    %% l1-MP  unknown rank/robust
    addpath(genpath('L1-MP'));
    tic
    [X_4, ~, ~] = GP(M, M_Omega, maxiter, array_Omega, p);
    toc;
    t_4 = [t_4 toc];
    peaksnr4_max = [peaksnr4_max psnr(X_4, M)];
    ssim4 = [ssim4 ssim(X_4, M)];
    RMSE4 = [RMSE4 norm((M - X_4),'fro')/sqrt(m*n)];
    rank4 = [rank4 rank(X_4)];

    %% FRR1MC  unknown rank/robust
    addpath(genpath('FRR1MC'));
    tic
    [X_5, ~,~] = GP_1(M, M_Omega, maxiter, array_Omega);
    toc;
    t_5 = [t_5 toc];
    peaksnr5_max = [peaksnr5_max psnr(X_5, M)];
    ssim5 = [ssim5 ssim(X_5, M)];
    RMSE5 = [RMSE5 norm((M - X_5),'fro')/sqrt(m*n)];
    rank5 = [rank5 rank(X_5)];

    %% RMC-Huber  known rank/robust
    addpath(genpath('RMC-Huber'));
    tic
    [X_7, RMSE_7,~] = RMC_huber(M, M_Omega, array_Omega, rak, maxiter);
    toc;
    t_7 = [t_7 toc];
    peaksnr7_max = [peaksnr7_max psnr(X_7, M)];
    ssim7 = [ssim7 ssim(X_7, M)];
    RMSE7 = [RMSE7 norm((M - X_7),'fro')/sqrt(m*n)];

    %% RMF-MM  known rank/robust
    addpath(genpath('RMF-MM'));
    U0 = randn(m,rak);
    V0 = randn(n,rak);
    para.lambda_u = 50/(m+n);
    para.lambda_v = 50/(m+n);
    para.rho = 1.5;
    tic
    [U_mm,V_mm,RMSE_8] = RMF_MM(array_Omega,M_Omega,U0,V0,para,M);
    toc
    t_8 = [t_8 toc];
    X_8 = U_mm*V_mm';
    peaksnr8_max = [peaksnr8_max psnr(X_8, M)];
    ssim8 = [ssim8 ssim(X_8, M)];
    RMSE8 = [RMSE8 norm((M - X_8),'fro')/sqrt(m*n)];

    %% Splp unknown rank/robust
    addpath(genpath('SpLp'));
    tic
    X_9 = LpRtracep(M_Omega, array_Omega, 10, 1);
    toc
    t_9 = [t_9 toc];
    peaksnr9_max = [peaksnr9_max psnr(X_9, M)];
    ssim9 = [ssim9 ssim(X_9, M)];
    RMSE9 = [RMSE9 norm((M - X_9),'fro')/sqrt(m*n)];
    
    %% HOAT
    addpath(genpath('HOAT'));
    ip = 4;
    tic
    [X_10,~,~,~]= HOAT(M_Omega, array_Omega, rak, maxiter, ip);
    toc
    t_10 = [t_10 toc];
    peaksnr10_max = [peaksnr10_max psnr(X_10, M)];
    ssim10 = [ssim10 ssim(X_10, M)];
    RMSE10 = [RMSE10 norm((M - X_10),'fro')/sqrt(m*n)];

    %% GUIG-log
    indices = find(array_Omega == 1);
    data = M_Omega(indices);

    addpath(genpath('GUIG-log'));
    p_i=[0.5,0.5];
    k=round(1.25*rak);                 
    for i=1:length(p_i)
        fprintf('p_%1d=%1d, ',i,p_i(i))
    end
    fprintf('\n')
    Inputsp = sparse(M_Omega);
    R = randn(n,k);
    tic;
    U = powerMethod(Inputsp,R,3,1e-6);
    [r,s,v] = svd(U'*Inputsp,'econ');u = U*r;
    S = diag(diag(s).^(1/2));
    X = u(:,1:k)*S(1:k,1:k);
    Y = S(1:k,1:k)*v(:,1:k)';

    clear opts
    opts.X = X;
    opts.Y = Y;

    opts.groundtruth = M;
    opts.show_progress = false;    % Whether show the progress
    opts.show_interval = 100;
    opts.eta = 0.05;
    opts.lambda = 0.01*norm(data,2);            % 1 for lp
    opts.mu = 1e-4*opts.lambda;
    opts.p_i = p_i;
    opts.p = p_i(1);
    opts.maxit = maxiter;            % Max Iteration
    opts.tol = 1e-3;
    opts.maxtime = 8e3;           % Sometimes terminating early is good for testing.
    opts.gate_upt = 1/2;          % The gate indicating which factor to be updated
    siz.m = m; siz.n = n;  siz.k = k;
    [X,Y] = Smf_lr_PALM(M_Omega,array_Omega,siz,opts);
    X_12 = X*Y;
    t_12 = [t_12 toc];
    peaksnr12_max = [peaksnr12_max psnr(X_12, M)];
    ssim12 = [ssim12 ssim(X_12, M)];
    RMSE12 = [RMSE12 norm((M - X_12),'fro')/sqrt(m*n)];

    %% HQ
    addpath(genpath('HQ-PF'))
    option.U=abs(1*rand(m,k));
    option.V=abs(1*rand(k,n));
    option.yita=2;
    option.sigmamin=0.01;
    option.maxitr=maxiter;

    tic
    option.stop_1=1e-3;
    option.stop_2=1e-5;
    X_13=HQ_PF(M_Omega,array_Omega,option);
    t_13 = [t_13 toc];
    peaksnr13_max = [peaksnr13_max psnr(X_13, M)];
    ssim13 = [ssim13 ssim(X_13, M)];
    RMSE13 = [RMSE13 norm((M - X_13),'fro')/sqrt(m*n)];

    %% NCPG
    addpath(genpath('RMC-NNSR'))
    tic
    [X_14,~,~,~,~,~,~,~,~] = NNSR_RMC(M_Omega,array_Omega,M,array_Omega,sqrt(2),sqrt(2),1.5);
    t_14 = [t_14 toc];
    peaksnr14_max = [peaksnr14_max psnr(X_14, M)];
    ssim14 = [ssim14 ssim(X_14, M)];
    RMSE14 = [RMSE14 norm((M - X_14),'fro')/sqrt(m*n)];

end

%% Data preparation

PSNR_tot = [mean(peaksnr1_max) max(peaksnr2_max) max(peaksnr3_max) mean(peaksnr4_max) max(peaksnr5_max) max(peaksnr7_max)...
    max(peaksnr8_max) max(peaksnr9_max) max(peaksnr10_max) max(peaksnr12_max) max(peaksnr13_max) max(peaksnr14_max)];
Time_tot  = [mean(t_1) mean(t_2) mean(t_3) mean(t_4) mean(t_5) mean(t_7) mean(t_8) mean(t_9) mean(t_10) mean(t_12) mean(t_13) mean(t_14)];
RMSE_tot  = [mean(RMSE1) mean(RMSE2) mean(RMSE3) mean(RMSE4) mean(RMSE5) mean(RMSE7)...
    mean(RMSE8) mean(RMSE9) mean(RMSE10) mean(RMSE12) mean(RMSE13) mean(RMSE14)];
Rank_tot = [mean(rank1) mean(rank2) mean(rank3) mean(rank4) mean(rank5) mean(rank7)...
    mean(rank8) mean(rank9) mean(rank10) mean(rank12) mean(rank13) mean(rank14)];
peaksnr1_max = [peaksnr1_max mean(peaksnr1_max)];
RMSE1 = [RMSE1 mean(RMSE1)];
peaksnr2_max = [peaksnr2_max mean(peaksnr2_max)];
RMSE2 = [RMSE2 mean(RMSE2)];
peaksnr3_max = [peaksnr3_max mean(peaksnr3_max)];
RMSE3 = [RMSE3 mean(RMSE3)];
peaksnr4_max = [peaksnr4_max mean(peaksnr4_max)];
RMSE4 = [RMSE4 mean(RMSE4)];
peaksnr5_max = [peaksnr5_max mean(peaksnr5_max)];
RMSE5 = [RMSE5 mean(RMSE5)];
peaksnr6_max = [peaksnr6_max mean(peaksnr6_max)];
RMSE6 = [RMSE6 mean(RMSE6)];
peaksnr7_max = [peaksnr7_max mean(peaksnr7_max)];
RMSE7 = [RMSE7 mean(RMSE7)];
peaksnr8_max = [peaksnr8_max mean(peaksnr8_max)];
RMSE8 = [RMSE8 mean(RMSE8)];
peaksnr9_max = [peaksnr9_max mean(peaksnr9_max)];
RMSE9 = [RMSE9 mean(RMSE9)];
peaksnr10_max = [peaksnr10_max mean(peaksnr10_max)];
RMSE10 = [RMSE10 mean(RMSE10)];
peaksnr12_max = [peaksnr12_max mean(peaksnr12_max)];
RMSE12 = [RMSE12 mean(RMSE12)];
peaksnr13_max = [peaksnr13_max mean(peaksnr13_max)];
RMSE13 = [RMSE13 mean(RMSE13)];
peaksnr14_max = [peaksnr14_max mean(peaksnr14_max)];
RMSE14 = [RMSE14 mean(RMSE14)];

%%
% time = datestr(now, 'yyyy-mm-dd HH-MM-SS');
% filename = sprintf('Image Gaussian and outlier %s.mat',time);
% save( fullfile('D:\Downloads\AROMC-robust\my code\data', filename) )

%% Show images
figure
% imshow(M)
imshow(M,'border','tight','initialmagnification','fit');
set (gcf,'Position',[100,100,n,m]);
axis normal;

figure
% imshow(M_Omega)
imshow(M_Omega,'border','tight','initialmagnification','fit');
set (gcf,'Position',[100,100,n,m]);
axis normal;
figure;
% imshow(X_1)
imshow(X_1,'border','tight','initialmagnification','fit');
set (gcf,'Position',[100,100,n,m]);
axis normal;
title('AROMC-HOW','Interpreter','latex')
figure;
% imshow(X_2)
imshow(X_2,'border','tight','initialmagnification','fit');
set (gcf,'Position',[100,100,n,m]);
axis normal;
title('$\ell_1$-ADMM','Interpreter','latex')
figure;
% imshow(X_3)
imshow(X_3,'border','tight','initialmagnification','fit');
set (gcf,'Position',[100,100,n,m]);
axis normal;
title('$\ell_1$-reg','Interpreter','latex')
figure;
% imshow(X_4)
imshow(X_4,'border','tight','initialmagnification','fit');
set (gcf,'Position',[100,100,n,m]);
axis normal;
title('$\ell_1$-MP','Interpreter','latex')
figure;
% imshow(X_5)
imshow(X_5,'border','tight','initialmagnification','fit');
set (gcf,'Position',[100,100,n,m]);
axis normal;
title('FRR1MC','Interpreter','latex')

figure;
% imshow(X_7)
imshow(X_7,'border','tight','initialmagnification','fit');
set (gcf,'Position',[100,100,n,m]);
axis normal;
title('RMC-Huber','Interpreter','latex')

figure;
% imshow(X_8)
imshow(X_8,'border','tight','initialmagnification','fit');
set (gcf,'Position',[100,100,n,m]);
axis normal;
title('RMF-MM','Interpreter','latex')

figure;
% imshow(X_9)
imshow(X_9,'border','tight','initialmagnification','fit');
set (gcf,'Position',[100,100,n,m]);
axis normal;
title('Sp$\ell_p$','Interpreter','latex')

figure;
% imshow(X_10)
imshow(X_10,'border','tight','initialmagnification','fit');
set (gcf,'Position',[100,100,n,m]);
axis normal;
title('GUIG$_{log}$-$\ell_1$','Interpreter','latex')

figure;
% imshow(X_12)
imshow(X_12,'border','tight','initialmagnification','fit');
set (gcf,'Position',[100,100,n,m]);
axis normal;
title('NCPG-$\ell_1$','Interpreter','latex')

figure;
% imshow(X_13)
imshow(X_13,'border','tight','initialmagnification','fit');
set (gcf,'Position',[100,100,n,m]);
axis normal;
title('CFN-RTC','Interpreter','latex')

figure;
% imshow(X_14)
imshow(X_14,'border','tight','initialmagnification','fit');
set (gcf,'Position',[100,100,n,m]);
axis normal;
title('HQ-TCTF','Interpreter','latex')

%% Local Magnify
addpath(genpath('plot_kits'));
start_y = 243;
start_x = 169;
rect_size = 30;
m_factor = 3;
Mm_noise = magnify_image(M_noise,start_y, start_x,rect_size,m_factor);
Mm = magnify_image(M,start_y, start_x,rect_size,m_factor);
Xm_1 = magnify_image(X_1,start_y, start_x,rect_size,m_factor);
Xm_2 = magnify_image(X_2,start_y, start_x,rect_size,m_factor);
Xm_3 = magnify_image(X_3,start_y, start_x,rect_size,m_factor);
Xm_4 = magnify_image(X_4,start_y, start_x,rect_size,m_factor);
Xm_5 = magnify_image(X_5,start_y, start_x,rect_size,m_factor);
% Xm_6 = magnify_image(X_6,start_y, start_x,rect_size,m_factor);
Xm_7 = magnify_image(X_7,start_y, start_x,rect_size,m_factor);
Xm_8 = magnify_image(X_8,start_y, start_x,rect_size,m_factor);
Xm_9 = magnify_image(X_9,start_y, start_x,rect_size,m_factor);
Xm_10 = magnify_image(X_10,start_y, start_x,rect_size,m_factor);
Xm_12 = magnify_image(X_12,start_y, start_x,rect_size,m_factor);
Xm_13 = magnify_image(X_13,start_y, start_x,rect_size,m_factor);
Xm_14 = magnify_image(X_14,start_y, start_x,rect_size,m_factor);


%% Draw Maginified Fig

figure
% imshow(M)
imshow(Mm,'border','tight','initialmagnification','fit');
set (gcf,'Position',[100,100,n,m]);
axis normal;

figure
% imshow(M_Omega)
imshow(M_Omega,'border','tight','initialmagnification','fit');
set (gcf,'Position',[100,100,n,m]);
axis normal;
figure;
% imshow(X_1)
imshow(Xm_1,'border','tight','initialmagnification','fit');
set (gcf,'Position',[100,100,n,m]);
axis normal;
title('AROMC-HOW','Interpreter','latex')
figure;
% imshow(X_2)
imshow(Xm_2,'border','tight','initialmagnification','fit');
set (gcf,'Position',[100,100,n,m]);
axis normal;
title('$\ell_1$-ADMM','Interpreter','latex')
figure;
% imshow(X_3)
imshow(Xm_3,'border','tight','initialmagnification','fit');
set (gcf,'Position',[100,100,n,m]);
axis normal;
title('$\ell_1$-reg','Interpreter','latex')
figure;
% imshow(X_4)
imshow(Xm_4,'border','tight','initialmagnification','fit');
set (gcf,'Position',[100,100,n,m]);
axis normal;
title('$\ell_1$-MP','Interpreter','latex')
figure;
% imshow(X_5)
imshow(Xm_5,'border','tight','initialmagnification','fit');
set (gcf,'Position',[100,100,n,m]);
axis normal;
title('FRR1MC','Interpreter','latex')

figure;
% imshow(X_7)
imshow(Xm_7,'border','tight','initialmagnification','fit');
set (gcf,'Position',[100,100,n,m]);
axis normal;
title('RMC-Huber','Interpreter','latex')

figure;
% imshow(X_8)
imshow(Xm_8,'border','tight','initialmagnification','fit');
set (gcf,'Position',[100,100,n,m]);
axis normal;
title('RMF-MM','Interpreter','latex')

figure;
% imshow(X_9)
imshow(Xm_9,'border','tight','initialmagnification','fit');
set (gcf,'Position',[100,100,n,m]);
axis normal;
title('Sp$\ell_p$','Interpreter','latex')

figure;
% imshow(X_10)
imshow(Xm_10,'border','tight','initialmagnification','fit');
set (gcf,'Position',[100,100,n,m]);
axis normal;
title('GUIG$_{log}$-$\ell_1$','Interpreter','latex')

figure;
% imshow(X_12)
imshow(Xm_12,'border','tight','initialmagnification','fit');
set (gcf,'Position',[100,100,n,m]);
axis normal;
title('NCPG-$\ell_1$','Interpreter','latex')

figure;
% imshow(X_13)
imshow(Xm_13,'border','tight','initialmagnification','fit');
set (gcf,'Position',[100,100,n,m]);
axis normal;
title('CFN-RTC','Interpreter','latex')

figure;
% imshow(X_14)
imshow(Xm_14,'border','tight','initialmagnification','fit');
set (gcf,'Position',[100,100,n,m]);
axis normal;
title('HQ-TCTF','Interpreter','latex')

%% tight_subplot
figure
% ha = self_subplot(2,4,[.0001 .001],[0.05 0.05],[.0001 .0001]);
ha = self_subplot(2,7,[.08 .03],[.1 .01],[.01 .01]);

axes(ha(1));
imshow(M)
pos=axis;%????????????[xmin xmax ymin ymax]
xlabel('$\it{\bf{\Theta}}$','Interpreter','latex','position',[(pos(2)+pos(1))/2 1.05*pos(4)])

axes(ha(2));
imshow(M_Omega)
% pos=axis;%????????????[xmin xmax ymin ymax]
xlabel('$\bf{\Theta}_\Omega$','position',[(pos(2)+pos(1))/2 1.05*pos(4)],'Interpreter','latex')

axes(ha(3));
imshow(X_1)
pos=axis;%????????????[xmin xmax ymin ymax]
xlabel({'AROMC-HOW';...
    ['PSNR = ', num2str(peaksnr1_max(1)),' SSIM = ', num2str(ssim1(1))]},'position',[(pos(2)+pos(1))/2 1.05*pos(4)],'Interpreter','latex')

axes(ha(4));
imshow(X_2)
pos=axis;%????????????[xmin xmax ymin ymax]
xlabel({'$\ell_1$-ADMM';...
    ['PSNR = ', num2str(peaksnr2_max(1)),' SSIM = ', num2str(ssim2(1))]},'position',[(pos(2)+pos(1))/2 1.05*pos(4)],'Interpreter','latex')

axes(ha(5));
imshow(X_3)
pos=axis;%????????????[xmin xmax ymin ymax]
xlabel({'$\ell_1$-reg';...
    ['PSNR = ', num2str(peaksnr3_max(1)),' SSIM = ', num2str(ssim3(1))]},'position',[(pos(2)+pos(1))/2 1.05*pos(4)],'Interpreter','latex')

axes(ha(6));
imshow(X_4)
pos=axis;%????????????[xmin xmax ymin ymax]
xlabel({'$\ell_1$-MP';...
    ['PSNR = ', num2str(peaksnr4_max(1)),' SSIM = ', num2str(ssim4(1))]},'position',[(pos(2)+pos(1))/2 1.05*pos(4)],'Interpreter','latex')

axes(ha(7));
imshow(X_5)
% pos=axis;%????????????[xmin xmax ymin ymax]
xlabel({'FRR1MC';...
    ['PSNR = ', num2str(peaksnr5_max(1)),' SSIM = ', num2str(ssim5(1))]},'position',[(pos(2)+pos(1))/2 1.05*pos(4)],'Interpreter','latex')

axes(ha(8));
imshow(X_7)
% pos=axis;%????????????[xmin xmax ymin ymax]
xlabel({'RMC-Huber';...
    ['PSNR = ', num2str(peaksnr7_max(1)),' SSIM = ', num2str(ssim7(1))]},'position',[(pos(2)+pos(1))/2 1.05*pos(4)],'Interpreter','latex')

axes(ha(9));
imshow(X_8)
% pos=axis;%????????????[xmin xmax ymin ymax]
xlabel({'RMF-MM';...
    ['PSNR = ', num2str(peaksnr8_max(1)),' SSIM = ', num2str(ssim8(1))]},'position',[(pos(2)+pos(1))/2 1.05*pos(4)],'Interpreter','latex')

axes(ha(10));
imshow(X_9)
% pos=axis;%????????????[xmin xmax ymin ymax]
xlabel({'Sp$\ell_p$';...
    ['PSNR = ', num2str(peaksnr9_max(1)),' SSIM = ', num2str(ssim9(1))]},'position',[(pos(2)+pos(1))/2 1.05*pos(4)],'Interpreter','latex')

axes(ha(11));
imshow(X_10)
% pos=axis;%????????????[xmin xmax ymin ymax]
xlabel({'GUIG$_{log}$-$\ell_1$';...
    ['PSNR = ', num2str(peaksnr10_max(1)),' SSIM = ', num2str(ssim10(1))]},'position',[(pos(2)+pos(1))/2 1.05*pos(4)],'Interpreter','latex')

axes(ha(12));
imshow(X_12)
% pos=axis;%????????????[xmin xmax ymin ymax]
xlabel({'NCPG-$\ell_1$';...
    ['PSNR = ', num2str(peaksnr12_max(1)),' SSIM = ', num2str(ssim12(1))]},'position',[(pos(2)+pos(1))/2 1.05*pos(4)],'Interpreter','latex')

axes(ha(13));
imshow(X_13)
% pos=axis;%????????????[xmin xmax ymin ymax]
xlabel({'CFN-RTC';...
    ['PSNR = ', num2str(peaksnr13_max(1)),' SSIM = ', num2str(ssim13(1))]},'position',[(pos(2)+pos(1))/2 1.05*pos(4)],'Interpreter','latex')

axes(ha(14));
imshow(X_14)
% pos=axis;%????????????[xmin xmax ymin ymax]
xlabel({'HQ-TCTF';...
    ['PSNR = ', num2str(peaksnr14_max(1)),' SSIM = ', num2str(ssim14(1))]},'position',[(pos(2)+pos(1))/2 1.05*pos(4)],'Interpreter','latex')
%%
fig = gcf;
fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];