function [M_hat,Y,S_hat,RE_M, RE_S,RE_X,RE_E, RE_inf,iter] = NNSR_RMC(X,Omega,M,S,IP_1,IP_2,xi,tol, maxIter)

% X - m x n matrix of observations/data (required input)
%
% Omega - observation data indicator (binary matrix)
%
% M - The true matrix
%
% IP_1, IP_2 and xi are hyperparameters
%
% tol - tolerance for stopping criterion.
%     - DEFAULT 1e-7 if omitted or -1.
%
% maxIter - maximum number of iterations
%         - DEFAULT 1000, if omitted or -1.
%
% addpath PROPACK;
% Sep. 2024
%
% This matlab code implements Robust matrix completion based on double non-convex regularizers associated with the Welsch function in
% "Robust low-rank matrix completion via sparsity-inducing regularizer." Wang, Z.Y., So, H.C. and Zoubir, A.M., Signal Processing, p.109666, 2024.
%
% The program is written based on the code provided in
%
% "Weighted nuclear norm minimization and its applications in low level
%  vision". S. Gu, Q. Xie, D. Meng, W. Zuo, X. Feng, L. Zhang.

% "The Augmented Lagrange Multiplier Method for Exact Recovery of Corrupted Low-Rank Matrices".
%    Z. Lin, M. Chen, L. Wu. arXiv:1009.5055, 2010
%

if nargin < 8
    tol = 1e-7;
elseif tol == -1
    tol = 1e-7;
end

if nargin < 9
    maxIter = 1000;
elseif maxIter == -1
    maxIter = 1000;
end

[m n] = size(X);
In_Omega = 1 - Omega;
P_O=@(x,LAMB,sigm) 0.*(abs(x)<=LAMB) + (abs(x)-abs(x).*exp((LAMB^2-x.^2)/sigm^2.*(abs(x)>LAMB))).*sign(x).*(abs(x)>LAMB);
P_1=@(x,LAMB) 0.*(abs(x)<=LAMB)+(abs(x)-LAMB).*sign(x).*(abs(x)>LAMB);

% initialize
Y = X;
norm_two = lansvd(Y, 1, 'L');
norm_inf = norm( Y(:), inf) ;
dual_norm = max(norm_two, norm_inf);
Y = Y / dual_norm;

M_hat = zeros( m, n);
S_hat = zeros( m, n);
mu = 1/norm_two; % this one can be tuned
mu_bar = mu * 1e7;
rho = 3;          % 1.05 this one can be tuned 高斯3 
d_norm = norm(X, 'fro');

iter = 0;
total_svd = 0;
converged = false;
stopCriterion = 1;
sv = 10;
lambda = 0.3 / (sqrt(m));
RE_M = [ ];
RE_S = [ ];
RE_E = [ ];
RE_X = [ ];
RE_inf = [];
A1 = X;
S1 = X;
E1 = X;
E_hat = M_hat.*In_Omega;
M_Omega = M.*Omega;
M_InOmega = M.*In_Omega;
S_Omega = S.*Omega;
while ~converged
    iter = iter + 1;
    temp_T = (X - M_hat + (1/mu)*Y).*Omega;
    LAMB_1 = lambda/mu;
    S_hat = P_1(temp_T,LAMB_1); % L1

    if choosvd(n, sv) == 1
        [U S_1 V] = lansvd(X - S_hat + (1/mu)*Y - E_hat, sv, 'L');
    else
        [U S_1 V] = svd(X - S_hat + (1/mu)*Y - E_hat, 'econ');
    end

    diagS = diag(S_1);
    LAMB_2 = 1/mu;
    tempDiagS = ST23(diagS,LAMB_2); % L2/3
    svp = length(tempDiagS);
    M_hat = U(:,1:svp)*diag(tempDiagS)*V(:,1:svp)';

    E_hat = ((1/mu)*Y-M_hat).*In_Omega;

    if svp < sv
        sv = min(svp + 1, n);
    else
        sv = min(svp + round(0.05*n), n);
    end

    total_svd = total_svd + 1;

    Z = X - M_hat - S_hat -E_hat;

    Y = Y + mu*Z;
    mu = min(mu*rho, mu_bar);

    %% stop Criterion
    stopCriterion = norm(Z, 'fro') / d_norm;
    if stopCriterion < tol
        converged = true;
    end

    if ~converged && iter >= maxIter
        converged = 1 ;
    end

    RE_M = [RE_M min(1,norm((M_hat-A1),'fro')/norm(A1,'fro'))];
    RE_S = [RE_S min(1,norm(S_hat-S1,'fro')/norm(S_Omega,'fro'))];
    RE_E = [RE_E min(1,norm(E_hat-E1,'fro')/norm(M_InOmega,'fro'))];
    RE_X = [RE_X min(1,stopCriterion)];

    RE_inf = [RE_inf max([norm(M_hat-A1,'inf') norm(S_hat-S1,'inf') norm(E_hat-E1,'inf')])];

    A1 = M_hat;
    S1 = S_hat;
    E1 = E_hat;
end
end

function w = ST23(temp_v, lambda)
% lambda = 1.5*(lambda^4/2)^(1/3);
temp_p = 2*(3*lambda^3)^0.25/3;

temp_w = abs(temp_v) > temp_p;
temp_w = sign(temp_v).*temp_w;
pp = acosh((27*temp_v.^2)/16*lambda^(-3/2));
pp = 2/sqrt(3)*lambda^(0.25)*(cosh(pp/3).^(0.5));
w  = temp_w.*((pp + sqrt(2*abs(temp_v)./pp - pp.^2))/2).^3;
end

function w = ST12(temp_v, lambda)
lambda = sqrt(64*lambda^3/54);
temp_c = 54^(1/3)*(lambda)^(2/3)/4;
temp_w = abs(temp_v) > temp_c; 
%temp_H = temp_w.*(abs(temp_v/3).^(-3/2)); 
temp_H = acos((lambda/8)*(temp_w.*(abs(temp_v/3).^(-3/2))));
temp_H = temp_w.*(1 + cos((2/3)*pi - (2/3)*temp_H));
w = (2/3)*temp_v.*temp_H;
end