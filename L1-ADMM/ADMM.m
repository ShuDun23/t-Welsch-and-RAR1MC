function [Out_X, RMSE,T_iter] = ADMM(M, M_Omega, Omega_array, rak, maxiter, p, mu)
% l1-admm
E = zeros(size(M_Omega));
Gamma = zeros(size(M_Omega));
iter = 1;
RMSE = [];
[r,c] = size(M_Omega);
sum_Omega = sum(sum(Omega_array));
T_iter = [];
T_iter = [T_iter 0];
RMSE = [RMSE 1];
Out_X_1 = 0;
U_1 = 0;
V_1 = 0;
while(iter <= maxiter)
%     tic
    [U, V] = LP2(E.*Omega_array - Gamma.*Omega_array./mu+ M_Omega,Omega_array, rak, 5);
    
    Y = (U*V).*Omega_array + Gamma.*Omega_array./mu - M_Omega;
    
    Omega = find(Omega_array == 1);
    
    y = Y(Omega);
    T = U*V;
    t = T(Omega);
    lambda = Gamma(Omega);
    e = calE(y, mu, p);
    lambda = lambda + mu*(t - e - M_Omega(Omega));
    Gamma(Omega) = lambda;
    E(Omega) = e;
    Out_X = U*V;
%     RMSE(1, iter) = norm((M-U*V).*Omega_array,'fro')/sqrt(sum_Omega);
    RMSE = [RMSE norm(M-Out_X,'fro')/sqrt(r*c)];
    
    rel_1 = norm(Out_X-Out_X_1,'fro')^2/norm(Out_X,'fro')^2;
%     rel_2 = norm(U_1-U,'fro')^2/norm(U,'fro')^2
%     rel_3 = norm(V_1-V,'fro')^2/norm(V,'fro')^2
    Out_X_1 = Out_X;
%     U_1 = U;
%     V_1 = V;
%     if max([rel_1,rel_2,rel_3])<1*10^(-4)
    if rel_1 < 1*10^(-4)
        break
    end
    
    
%     toc 
%     T_iter = [T_iter toc + T_iter(end)];
%         if norm(M - U*V,'fro')^2/(r*c) < 0.1
%             break
%         end
    iter = iter + 1;
    
end
end



function [U, V] = LP2(M_Omega,Omega_array,rak,maxiter)
[r,c] = size(M_Omega);
U = randn(r,rak);
V = randn(rak,c);
for iter = 1 : maxiter
    for j = 1:c
        row = find(Omega_array(:,j) == 1);
        U_I =  U(row,:);
        b_I = M_Omega(row,j);
        V(:,j) = pinv(U_I)* b_I;
    end
    for i = 1 : r
        col = find(Omega_array(i,:) == 1);
        V_I = V(:,col);
        b_I = M_Omega(i,col);
        U(i,:) = b_I * pinv(V_I);
    end
end
end
function e = calE(y, mu, p)
g_diff = @(e_i,y_i) e_i - y_i +p/mu*e_i^(p-1);
g_diff2 = @(e_i,y_i) e_i - y_i -p/mu*(-e_i)^(p-1);
g = @(e_i,y_i) 0.5*(e_i - y_i)^2 + 1/mu*abs(e_i)^p;
if p ==1
    y_sign = y;
    y_sign( y_sign<0 )=-1;
    y_sign( y_sign>0 )=1;
    y_hat = abs(y) - 1/mu*ones(size(y));
    y_hat(y_hat<0)=0;
    e = y_sign.*y_hat;
elseif p<1
    e = zeros(size(y));
    beta_a = 2/mu*(1-p)^(1/(2-p));
    h_a = beta_a + p/mu*beta_a^(p-1);
    for i = 1 : size(y,1)
        if abs(y(i)) <= h_a
            e(i,1) = 0;
        else
            beta = beta_a;
            for j = 1 : 20
                beta = abs(y(i)) -1/mu*beta^(p-1);
            end
            beta_star = beta;
            e(i) = sign(y(i))*beta_star;
        end
    end
else
    e = zeros(size(y));
    for i = 1 : size(y,1)
        if y(i) >= 0
            a = 0;
            b = y(i);
            c = (a + b)/2;
            middle = g_diff(c,y(i));
            inner_iter = 1;
            while abs(middle) > 1e-8
                if (middle > 0)
                    b = c;
                else
                    a = c;
                end
                c = (a+b)/2;
                middle = g_diff(c,y(i));
                inner_iter = inner_iter + 1;
                if inner_iter > 50
                    break
                end
            end
            if g(c,y(i)) < g(0,y(i))
                e(i,1) = c;
            else
                e(i,1) = 0;
            end
        else
            a = y(i);
            b = 0;
            c = (a + b)/2;
            middle = g_diff2(c,y(i));
            inner_iter = 1;
            while abs(middle) > 1e-8
                if (middle > 0)
                    b = c;
                else
                    a = c;
                end
                c = (a+b)/2;
                middle = g_diff2(c,y(i));
                inner_iter = inner_iter + 1;
                if inner_iter > 50
                    break
                end
            end
            if g(c,y(i)) < g(0,y(i))
                e(i) = c;
            else
                e(i) = 0;
            end
        end
    end
end
end