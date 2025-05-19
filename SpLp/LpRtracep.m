% min_X  ||X.*A-D.*A||_p^p + r*||X||_Sp^p
% 0<p<1
% Optimization with ALM

function [X, obj] = LpRtracep(D, A, r, p)
% D: m*n data matrix
% A: m*n matrix, Aij=1 if Dij is observed, otherwise Aij=0
% r: parameter
% p: the p value of the Lp norm and Sp norm
% X: m*n recovered data matrix

% Ref:
% Feiping Nie, Hua Wang, Xiao Cai, Heng Huang, Chris H. Q. Ding
% Robust Matrix Completion via Joint Schatten p-Norm and lp-Norm Minimization. 
% ICDM 2012: 566-574



ITER = 50;

obj = zeros(ITER,1);
[m, n] = size(D);
transpose = 0;
if m<n
    transpose = 1;
    D = D';
    A = A';
    [m, n] = size(D);
end;
X = D.*A;
%X=zeros(m,n);
e = abs((X-D).*A); e=e(:); ex = svd(X,0);
obj(1) = sum(e.^p) + r*sum(ex.^p);
idx  = find(A==1);
idx1 = find(A~=1);

Sigma = zeros(m,n);
Lamda = zeros(m,n);
E     = zeros(m,n);
mu    = 0.01;
rho   = 1.2;   % 1<rho<1.5, smaller rho gives better solution but slower speed

for iter = 1:ITER
    inmu = 1/mu;
    
    M = X + inmu*Sigma;
    [U, S, V] = svd(M,0);
    s = diag(S);
    lambda = 2*r*inmu;
    for i = 1:length(s)
        s1(i) = findrootp0(s(i), lambda, p);
    end;
    Z = U*diag(s1)*V';
    
    M2 = X-D-inmu*Lamda;
    lambda2 = 2*inmu;
    for i = 1:length(idx)
        E(idx(i)) = findrootp(M2(idx(i)), lambda2, p);
    end;
      
    M3 = Z - inmu*Sigma;
    M4 = E + D+inmu*Lamda;
    X(idx1) = M3(idx1);
    X(idx) = (M3(idx) + M4(idx))/2;
    
    Sigma = Sigma + mu*(X-Z);
    Lamda = Lamda + mu*(E-X+D);
    mu = min(10^20, mu*rho);
    
    e = abs((X-D).*A); e=e(:);
    ex = svd(X,0);
    obj(iter+1) = sum(e.^p) + r*sum(ex.^p);
end;

if transpose == 1
    X = X';
end;







% min_{sigma}  sigma^2 - 2*alpha*sigma + lambda*|sigma|^p
% or min_{sigma}  (sigma-alpha)^2 + lambda*|sigma|^p
function sigma = findrootp(alpha, lambda, p)

a = (lambda*p*(1-p)/2)^(1/(2-p))+eps;
b = 2*a-2*alpha+lambda*p*a^(p-1);
if b < 0
    sigma = 2*a;
    for i = 1:10
        f = 2*sigma-2*alpha+lambda*p*sigma^(p-1);
        g = 2+lambda*p*(p-1)*sigma^(p-2);
        sigma = sigma-f/g;
    end;
    sigma1 = sigma;
    ob1 = sigma^2-2*sigma*alpha+lambda*abs(sigma)^p;
else
    sigma1 = 1;ob1=inf;
end;

a = -a;
b = 2*a-2*alpha+lambda*p*abs(a)^(p-1);
if b > 0
    sigma = 2*a;
    for i = 1:10
        f = 2*sigma-2*alpha-lambda*p*abs(sigma)^(p-1);
        g = 2+lambda*p*(p-1)*abs(sigma)^(p-2);
        sigma = sigma-f/g;
    end;
    sigma2 = sigma;
    ob2 = sigma^2-2*sigma*alpha+lambda*abs(sigma)^p;
else
    sigma2 = 1;ob2=inf;
end;
sigma_can = [0,sigma1,sigma2];
[temp,idx] = min([0,ob1,ob2]);
sigma = sigma_can(idx);
%sig=sigma-0.1:.01:sigma+0.1;ob = sig.^2-2*sig*alpha+lambda*abs(sig).^p; [t,i]=min(ob);sg=sig(i);aa = abs(sigma-sg);if aa > 0.011   disp('ft'); end;
1;


% min_{sigma>=0}  sigma^2 - 2*alpha*sigma + lambda*|sigma|^p
function sigma = findrootp0(alpha, lambda, p)

a = (lambda*p*(1-p)/2)^(1/(2-p))+eps;
b = 2*a-2*alpha+lambda*p*a^(p-1);
if b < 0
    sigma = 2*a;
    for i = 1:10
        f = 2*sigma-2*alpha+lambda*p*sigma^(p-1);
        g = 2+lambda*p*(p-1)*sigma^(p-2);
        sigma = sigma-f/g;
    end;
    ob = sigma^2-2*sigma*alpha+lambda*sigma^p;
    if ob > 0
        sigma = 0;
    end;
else
    sigma = 0;
end;
%sig=0:.01:1.1*sigma;ob = sig.^2-2*sig*alpha+lambda*abs(sig).^p; [t,i]=min(ob);sg=sig(i);aa = abs(sigma-sg);if aa > 0.011   disp('ft'); end;
1;