function [Y, RMSE,PSNR_M] = GP(M, M_Omega, maxiteration, array_Omega, p)
% l1-MP
[m ,n] = size(M_Omega);
X_j = zeros(m, n);
u_j = randn(m, 1);
v_j = randn(n, 1);
v_j = sum(M_Omega,1)';
RMSE = [];
in = 0;
PSNR_M = [];
sum_Omega = sum(sum(array_Omega));
for iter = 1 : maxiteration
    R_j = M_Omega - X_j.*array_Omega; 
    %% calculate u_j
    in2 = 0;
    while (in2 < 2)
%         a = norm(R_j - u_j * v_j.','fro')^2/norm(R_j,'fro')^2;
        a = lp_norm(R_j - u_j * v_j.',p)^(p)/lp_norm(R_j,p)^(p);
        for i = 1 : m
            g_l = R_j(i,:).';
            index = array_Omega(i,:).';
%             u_j(i,1) = Newton_BLR(v_j, g_l,p,100,index);
            u_j(i,1) = IRLS(v_j, g_l, p, 10, index);
        end
        
        %% calculate v_j
        for i = 1 : n
            h_q = R_j(:,i);
            index = array_Omega(:,i);
%             v_j(i,1) = Newton_BLR(u_j, h_q, p,100,index);
            v_j(i,1) = IRLS(u_j, h_q, p, 10, index);
        end
%         b = norm(R_j - u_j * v_j.','fro')^2/norm(R_j,'fro')^2;
        b = lp_norm(R_j - u_j * v_j.',p)^(p)/lp_norm(R_j,p)^(p);
        if a-b < 0.005
            in2 = in2 + 1;
        end
    end
    X_j = X_j + u_j * v_j.';
    PSNR_M =[PSNR_M psnr(X_j, M)];
    %% judgement
%     MSE = [MSE, norm(M-X_j,'fro')^2/norm(M,'fro')^2];
    % RMSE = [RMSE, norm((M-X_j).*array_Omega,'fro')/sqrt(sum_Omega)];
    RMSE = [RMSE, norm((M-X_j),'fro')/sqrt(m*n)];
    Y = X_j;
    if iter~=1
        step_MSE = RMSE(iter-1) - RMSE(iter);
        if step_MSE < 1e-4 %0.0005
            in = in  + 1;
        end
        if in > 1
            break;
        end
    end

end

end