%% Recover multiple images and calculate metrices
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%          Hao Nan Sheng, Zhi-Yong Wang, Hing Cheung So           %
%    Robust Rank-One Matrix Completion via Explicit Regularizer   %
% IEEE Transactions on Neural Networks and Learning Systems, 2025 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
PSNR_tot = zeros(8,12);
SSIM_tot = zeros(8,12);
Time_tot = zeros(8,12);
RMSE_tot = zeros(8,12);
Rank_tot = zeros(8,12);
%% Hyperparameters
K = 5; % Monte Carlo
rak = 10;
maxiter = 50;
p = 1; % l_p norm
xi1 = 5; % t-Welsch hyperparameters
xi2 = 5;
%% Import data
imgPath = 'path/to/your/images/';
imgDir  = dir([imgPath '*.jpg']);
for i = 1:length(imgDir)
    image = imread([imgPath imgDir(i).name]);
    [width,height,z]=size(image);
    if(z>1)
        image=rgb2gray(image);
    end
    % unit8 to double
    image = mat2gray(image);
    [m,n] = size(image);
    M = image;

    %% Robust MC methods
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
    for kk=1:K % Monte Carlo
        disp(['i=',num2str(i)]);
        disp(['kk=',num2str(kk)]);

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
        [X_1,Y1_RMSE1,real_RMSE,peaksnr1,U1,V1,NRE1] = RAR1MC(M, M_Omega, array_Omega, maxiter, xi1, xi2);
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
        % k = m;
        for ii=1:length(p_i)
            fprintf('p_%1d=%1d, ',ii,p_i(ii))
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
    PSNR_tot(i,:) = [mean(peaksnr1_max) max(peaksnr2_max) max(peaksnr3_max) mean(peaksnr4_max) max(peaksnr5_max) max(peaksnr7_max)...
        max(peaksnr8_max) max(peaksnr9_max) max(peaksnr10_max) max(peaksnr12_max) max(peaksnr13_max) max(peaksnr14_max)];
    Time_tot(i,:)  = [mean(t_1) mean(t_2) mean(t_3) mean(t_4) mean(t_5) mean(t_7) mean(t_8) mean(t_9) mean(t_10) mean(t_12) mean(t_13) mean(t_14)];
    RMSE_tot(i,:)  = [mean(RMSE1) mean(RMSE2) mean(RMSE3) mean(RMSE4) mean(RMSE5) mean(RMSE7)...
        mean(RMSE8) mean(RMSE9) mean(RMSE10) mean(RMSE12) mean(RMSE13) mean(RMSE14)];
    Rank_tot(i,:) = [mean(rank1) mean(rank2) mean(rank3) mean(rank4) mean(rank5) mean(rank7)...
        mean(rank8) mean(rank9) mean(rank10) mean(rank12) mean(rank13) mean(rank14)];
end
%% Plot
PSNR_total = [PSNR_tot; mean(PSNR_tot,1)].';
SSIM_total = [SSIM_tot; mean(SSIM_tot,1)].';
Time_total = [Time_tot; mean(Time_tot,1)].';
RMSE_total = [RMSE_tot; mean(RMSE_tot,1)].';

PSNR_total=round(PSNR_total * 1e4) / 1e4;
Time_total=round(Time_total * 1e4) / 1e4;
RMSE_total=round(RMSE_total * 1e4) / 1e4;
SSIM_total=round(SSIM_total * 1e4) / 1e4;

%% Save workspace
time = datestr(now, 'yyyy-mm-dd HH-MM-SS');
filename = sprintf('8 Images Gaussian and outlier %s.mat',time);
save( fullfile('path\to\your\data', filename) )
