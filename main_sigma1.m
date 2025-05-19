%% Recover synthetic data with different Gaussian noise variance SNR
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

%% Hyperparameters
K = 10; % Monte Carlo
rak = 10;
maxiter = 50;
p = 1; % l_p norm
xi1 = 5; % t-Welsch hyperparameters
xi2 = 5;
%% Robust MC methods
SNR = 0:5:10;

[PSNR1,PSNR2,PSNR3,PSNR4,PSNR5,PSNR6,PSNR7,PSNR8,PSNR9,PSNR10,PSNR11,PSNR12,PSNR13,PSNR14] = deal(zeros(length(SNR),K));
[SSIM1,SSIM2,SSIM3,SSIM4,SSIM5,SSIM6,SSIM7,SSIM8,SSIM9,SSIM10,SSIM11,SSIM12,SSIM13,SSIM14] = deal(zeros(length(SNR),K));
[RMSE1,RMSE2,RMSE3,RMSE4,RMSE5,RMSE6,RMSE7,RMSE8,RMSE9,RMSE10,RMSE11,RMSE12,RMSE13,RMSE14] = deal(zeros(length(SNR),K));
for ii = 1:length(SNR)
    [peaksnr1_max,peaksnr2_max,peaksnr3_max,peaksnr4_max,peaksnr5_max,peaksnr6_max,peaksnr7_max,...
        peaksnr8_max,peaksnr9_max,peaksnr10_max,peaksnr11_max,peaksnr12_max,peaksnr13_max,peaksnr14_max] = deal([]);
    [ssim1,ssim2,ssim3,ssim4,ssim5,ssim6,ssim7,ssim8,ssim9,ssim10,ssim11,ssim12,ssim13,ssim14] = deal([]);
    [rmse1,rmse2,rmse3,rmse4,rmse5,rmse6,rmse7,rmse8,rmse9,rmse10,rmse11,rmse12,rmse13,rmse14] = deal([]);
    for kk = 1:K % Monte Carlo
        disp(['ii=',num2str(ii)]);
        disp(['kk=',num2str(kk)]);
        %%%%%%%%%%%%%% synthetic data %%%%%%%%%%%%%
        
        rak1 = 10;
        m = 300;
        n = 200;
        M_1 = randn(m,n); % N(0.1)
        [U,~,V] = svd(M_1);
        a = ones(rak1,1);
        a(2:end) = 1.5;
        b = cumprod(a);
        b = flipud(b);
        S = diag(b);
        M = U(:,1:rak1)*S*V(:,1:rak1).';
        [m,n] = size(M);

        % Noise
        sigma1 = (norm(M,'fro'))^2/(m*n*10^(SNR(ii)/10));
        M_noise = M + sqrt(sigma1) * randn(m,n);

        per = 0.5; % Observation persentage
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
        rmse1 = [rmse1 norm((M - X_1),'fro')/sqrt(m*n)];
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
        rmse2 = [rmse2 norm((M - X_2),'fro')/sqrt(m*n)];
        rank2 = [rank2 rank(X_2)];

        %% l1-reg  known rank/robust
        addpath(genpath('L1-reg'));
        tic
        [X_3,RMSE_3,~,~] = LP1(M, M_Omega, array_Omega, rak, maxiter, p);
        toc;
        t_3 = [t_3 toc];
        peaksnr3_max = [peaksnr3_max psnr(X_3, M)];
        ssim3 = [ssim3 ssim(X_3, M)];
        rmse3 = [rmse3 norm((M - X_3),'fro')/sqrt(m*n)];
        rank3 = [rank3 rank(X_3)];

        %% l1-MP  unknown rank/robust
        addpath(genpath('L1-MP'));
        tic
        [X_4, ~, ~] = GP(M, M_Omega, maxiter, array_Omega, p);
        toc;
        t_4 = [t_4 toc];
        peaksnr4_max = [peaksnr4_max psnr(X_4, M)];
        ssim4 = [ssim4 ssim(X_4, M)];
        rmse4 = [rmse4 norm((M - X_4),'fro')/sqrt(m*n)];
        rank4 = [rank4 rank(X_4)];

        %% FRR1MC  unknown rank/robust
        addpath(genpath('FRR1MC'));
        tic
        [X_5, ~,~] = GP_1(M, M_Omega, maxiter, array_Omega);
        toc;
        t_5 = [t_5 toc];
        peaksnr5_max = [peaksnr5_max psnr(X_5, M)];
        ssim5 = [ssim5 ssim(X_5, M)];
        rmse5 = [rmse5 norm((M - X_5),'fro')/sqrt(m*n)];
        rank5 = [rank5 rank(X_5)];

        %% RMC-Huber  known rank/robust
        addpath(genpath('RMC-Huber'));
        tic
        [X_7, RMSE_7,~] = RMC_huber(M, M_Omega, array_Omega, rak, maxiter);
        toc;
        t_7 = [t_7 toc];
        peaksnr7_max = [peaksnr7_max psnr(X_7, M)];
        ssim7 = [ssim7 ssim(X_7, M)];
        rmse7 = [rmse7 norm((M - X_7),'fro')/sqrt(m*n)];

        %% RMF-MM  known rank/robust
        addpath(genpath('RMF-MM'));
        U0 = randn(m,rak);
        V0 = randn(n,rak);
        para.lambda_u = 20/(m+n);
        para.lambda_v = 20/(m+n);
        para.rho =1.5;
        tic
        [U_mm,V_mm,RMSE_8] = RMF_MM(array_Omega,M_Omega,U0,V0,para,M);
        toc
        t_8 = [t_8 toc];
        X_8 = U_mm*V_mm';
        peaksnr8_max = [peaksnr8_max psnr(X_8, M)];
        ssim8 = [ssim8 ssim(X_8, M)];
        rmse8 = [rmse8 norm((M - X_8),'fro')/sqrt(m*n)];

        %% Splp unknown rank/robust
        addpath(genpath('SpLp'));
        tic
        X_9 = LpRtracep(M_Omega, array_Omega, rak, p);
        toc
        t_9 = [t_9 toc];
        peaksnr9_max = [peaksnr9_max psnr(X_9, M)];
        rmse9 = [rmse9 norm((M - X_9),'fro')/sqrt(m*n)];
        ssim9 = [ssim9 ssim(X_9, M)];
        % Rank_9 = [Rank_2 rank(X_2)];

        %% HOAT
        addpath(genpath('HOAT'));
        ip = 3;
        tic
        [X_10,~,~,~]= HOAT(M_Omega, array_Omega, rak, maxiter, ip);
        toc
        t_10 = [t_10 toc];
        peaksnr10_max = [peaksnr10_max psnr(X_10, M)];
        ssim10 = [ssim10 ssim(X_10, M)];
        rmse10 = [rmse10 norm((M - X_10),'fro')/sqrt(m*n)];

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
        rmse12 = [rmse12 norm((M - X_12),'fro')/sqrt(m*n)];

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
        rmse13 = [rmse13 norm((M - X_13),'fro')/sqrt(m*n)];

        %% NCPG
        addpath(genpath('RMC-NNSR'))
        tic
        [X_14,~,~,~,~,~,~,~,~] = NNSR_RMC(M_Omega,array_Omega,M,array_Omega,sqrt(2),sqrt(2),1.5);
        t_14 = [t_14 toc];
        peaksnr14_max = [peaksnr14_max psnr(X_14, M)];
        ssim14 = [ssim14 ssim(X_14, M)];
        rmse14 = [rmse14 norm((M - X_14),'fro')/sqrt(m*n)];

        %% Data preparation
        [peaksnr6_max,peaksnr11_max] = deal(peaksnr12_max);
        [ssim6,ssim11] = deal(ssim12);
        [rmse6,rmse11] = deal(rmse12);
    end
    [PSNR1(ii,:),PSNR2(ii,:),PSNR3(ii,:),PSNR4(ii,:),PSNR5(ii,:),PSNR6(ii,:),PSNR7(ii,:),...
        PSNR8(ii,:),PSNR9(ii,:),PSNR10(ii,:),PSNR11(ii,:),PSNR12(ii,:),PSNR13(ii,:),PSNR14(ii,:)] = ...
        deal(peaksnr1_max,peaksnr2_max,peaksnr3_max,peaksnr4_max,peaksnr5_max,peaksnr6_max,peaksnr7_max,...
        peaksnr8_max,peaksnr9_max,peaksnr10_max,peaksnr11_max,peaksnr12_max,peaksnr13_max,peaksnr14_max);
    [SSIM1(ii,:),SSIM2(ii,:),SSIM3(ii,:),SSIM4(ii,:),SSIM5(ii,:),SSIM6(ii,:),SSIM7(ii,:),...
        SSIM8(ii,:),SSIM9(ii,:),SSIM10(ii,:),SSIM11(ii,:),SSIM12(ii,:),SSIM13(ii,:),SSIM14(ii,:)] = ...
        deal(ssim1,ssim2,ssim3,ssim4,ssim5,ssim6,ssim7,ssim8,ssim9,ssim10,ssim11,ssim12,ssim13,ssim14);
    [RMSE1(ii,:),RMSE2(ii,:),RMSE3(ii,:),RMSE4(ii,:),RMSE5(ii,:),RMSE6(ii,:),RMSE7(ii,:),...
        RMSE8(ii,:),RMSE9(ii,:),RMSE10(ii,:),RMSE11(ii,:),RMSE12(ii,:),RMSE13(ii,:),RMSE14(ii,:)] = ...
        deal(rmse1,rmse2,rmse3,rmse4,rmse5,rmse6,rmse7,rmse8,rmse9,rmse10,rmse11,rmse12,rmse13,rmse14);
end
result_PSNR = [mean(PSNR1,2),mean(PSNR2,2),mean(PSNR3,2),mean(PSNR4,2),mean(PSNR5,2),mean(PSNR6,2),mean(PSNR7,2),...
    mean(PSNR8,2),mean(PSNR9,2),mean(PSNR10,2),mean(PSNR11,2),mean(PSNR12,2),mean(PSNR13,2),mean(PSNR14,2)];
result_SSIM = [mean(SSIM1,2),mean(SSIM2,2),mean(SSIM3,2),mean(SSIM4,2),mean(SSIM5,2),mean(SSIM6,2),mean(SSIM7,2),...
    mean(SSIM8,2),mean(SSIM9,2),mean(SSIM10,2),mean(SSIM11,2),mean(SSIM12,2),mean(SSIM13,2),mean(SSIM14,2)];
result_RMSE = [mean(RMSE1,2),mean(RMSE2,2),mean(RMSE3,2),mean(RMSE4,2),mean(RMSE5,2),mean(RMSE6,2),mean(RMSE7,2),...
    mean(RMSE8,2),mean(RMSE9,2),mean(RMSE10,2),mean(RMSE11,2),mean(RMSE12,2),mean(RMSE13,2),mean(RMSE14,2)];

%% Plot
figure
plot(SNR,result_PSNR(:,1),'*-','Linewidth',1.5,'Color',"#FF00FF")
hold on
plot(SNR,result_PSNR(:,2),'o-','Linewidth',1.5,'Color',"#D95319")
plot(SNR,result_PSNR(:,3),'s-','Linewidth',1.5,'Color',"#EDB120")
plot(SNR,result_PSNR(:,4),'^-','Linewidth',1.5,'Color',"#7E2F8E")
plot(SNR,result_PSNR(:,5),'d-','Linewidth',1.5,'Color',"#77AC30")
plot(SNR,result_PSNR(:,7),'+-','Linewidth',1.5,'Color',"#A2142F")
plot(SNR,result_PSNR(:,8),'x-','Linewidth',1.5,'Color',"#0072BD")
plot(SNR,result_PSNR(:,9),'v-','Linewidth',1.5,'Color',"#4DBEEE")
plot(SNR,result_PSNR(:,12),'hexagram-','Linewidth',1.5,'Color',[0.75,0.75,0])
plot(SNR,result_PSNR(:,14),'_-','Linewidth',1.5,'Color',[0.25,0.25,0.25])
plot(SNR,result_PSNR(:,10),'.-','Linewidth',1.5,'Color',[0.75,0,0.75],'MarkerSize',8)
plot(SNR,result_PSNR(:,13),'pentagram-','Linewidth',1.5,'Color',[0.75,0.75,0.75])
xlabel('SNR(dB)','Interpreter','latex')
ylabel('PSNR (dB)','Interpreter','latex')
% title('PSNR versus $O_p$','Interpreter','latex')
legend('RAR1MC','$\ell_1$-ADMM','$\ell_1$-reg','$\ell_1$-MP','FRR1MC','RMC-Huber','RMF-MM',...
    'Sp$\ell_p$','GUIG$_{log}$-$\ell_1$','NCPG-$\ell_1$','CFN-RTC','HQ-TCTF','Interpreter','latex','Location','best')
set(gca,'FontSize',12,'FontName','Times');
grid on
fig = gcf;
fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];

figure
plot(SNR,result_SSIM(:,1),'*-','Linewidth',1.5,'Color',"#FF00FF")
hold on
plot(SNR,result_SSIM(:,2),'o-','Linewidth',1.5,'Color',"#D95319")
plot(SNR,result_SSIM(:,3),'s-','Linewidth',1.5,'Color',"#EDB120")
plot(SNR,result_SSIM(:,4),'^-','Linewidth',1.5,'Color',"#7E2F8E")
plot(SNR,result_SSIM(:,5),'d-','Linewidth',1.5,'Color',"#77AC30")
plot(SNR,result_SSIM(:,7),'+-','Linewidth',1.5,'Color',"#A2142F")
plot(SNR,result_SSIM(:,8),'x-','Linewidth',1.5,'Color',"#0072BD")
plot(SNR,result_SSIM(:,9),'v-','Linewidth',1.5,'Color',"#4DBEEE")
plot(SNR,result_SSIM(:,12),'hexagram-','Linewidth',1.5,'Color',[0.75,0.75,0])
plot(SNR,result_SSIM(:,14),'_-','Linewidth',1.5,'Color',[0.25,0.25,0.25])
plot(SNR,result_SSIM(:,10),'.-','Linewidth',1.5,'Color',[0.75,0,0.75],'MarkerSize',8)
plot(SNR,result_SSIM(:,13),'pentagram-','Linewidth',1.5,'Color',[0.75,0.75,0.75])
xlabel('SNR(dB)','Interpreter','latex')
ylabel('SSIM','Interpreter','latex')
% title('PSNR versus $O_p$','Interpreter','latex')
legend('RAR1MC','$\ell_1$-ADMM','$\ell_1$-reg','$\ell_1$-MP','FRR1MC','RMC-Huber','RMF-MM',...
    'Sp$\ell_p$','GUIG$_{log}$-$\ell_1$','NCPG-$\ell_1$','CFN-RTC','HQ-TCTF','Interpreter','latex','Location','best')
set(gca,'FontSize',12,'FontName','Times');
grid on
fig = gcf;
fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];

figure
plot(SNR,result_RMSE(:,1),'*-','Linewidth',1.5,'Color',"#FF00FF")
hold on
plot(SNR,result_RMSE(:,2),'o-','Linewidth',1.5,'Color',"#D95319")
plot(SNR,result_RMSE(:,3),'s-','Linewidth',1.5,'Color',"#EDB120")
plot(SNR,result_RMSE(:,4),'^-','Linewidth',1.5,'Color',"#7E2F8E")
plot(SNR,result_RMSE(:,5),'d-','Linewidth',1.5,'Color',"#77AC30")
plot(SNR,result_RMSE(:,7),'+-','Linewidth',1.5,'Color',"#A2142F") % RMC-HUBER
plot(SNR,result_RMSE(:,8),'x-','Linewidth',1.5,'Color',"#0072BD") % RMF-MM
plot(SNR,result_RMSE(:,9),'v-','Linewidth',1.5,'Color',"#4DBEEE") % Splp
plot(SNR,result_RMSE(:,12),'hexagram-','Linewidth',1.5,'Color',[0.75,0.75,0]) % GUIG-log
plot(SNR,result_RMSE(:,14),'_-','Linewidth',1.5,'Color',[0.25,0.25,0.25]) % NCPG
plot(SNR,result_RMSE(:,10),'.-','Linewidth',1.5,'Color',[0.75,0,0.75],'MarkerSize',8) % HOAT
plot(SNR,result_RMSE(:,13),'pentagram-','Linewidth',1.5,'Color',[0.75,0.75,0.75]) % HQ
xlabel('SNR(dB)','Interpreter','latex')
ylabel('RMSE','Interpreter','latex')
% title('PSNR versus $O_p$','Interpreter','latex')
legend('RAR1MC','$\ell_1$-ADMM','$\ell_1$-reg','$\ell_1$-MP','FRR1MC','RMC-Huber','RMF-MM',...
    'Sp$\ell_p$','GUIG$_{log}$-$\ell_1$','NCPG-$\ell_1$','CFN-RTC','HQ-TCTF','Interpreter','latex','Location','best')
set(gca,'FontSize',12,'FontName','Times');
grid on
%% 
fig = gcf;
fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];
%% Save workspace
time = datestr(now, 'yyyy-mm-dd HH-MM-SS');
filename = sprintf('synthetic data Gaussian only SNR %s.mat',time);
save( fullfile('path\to\your\data', filename) )