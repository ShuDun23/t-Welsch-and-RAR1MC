function [Out_X, RMSE,MSE] = RMC_huber(M, M_Omega, Omega_array, rak, maxiter)
% RMC-huber
[r,c] = size(M_Omega);
U = randn(r,rak);
V = randn(rak,c);
RMSE = [];
MSE = [];
sum_Omega = sum(sum(Omega_array));
for iter = 1 : maxiter
    clear row col;
    for j = 1:c
        row = find(Omega_array(:,j) == 1);
        U_I =  U(row,:);
        b_I = M_Omega(row,j);
        V(:,j) = Huber_function_UV( b_I,U_I ,1.345, 0.7102);
        clear U_I b_I;
    end
    clear row col;
    for i = 1:r
        col = find(Omega_array(i,:) == 1);
        V_I = V(:,col);
        b_I = M_Omega(i,col);
        U(i,:) = Huber_function_UV( b_I.',V_I.' ,1.345, 0.7102);
        clear V_I b_I;
    end
    X = U*V;
    MSE = [MSE norm(M-X,'fro')^2/(r*c)];
    RMSE= [RMSE norm(M-X,'fro')/sqrt(r*c)];
%     RMSE= [RMSE norm((M-X).*Omega_array,'fro')/sqrt(sum_Omega)];

    if iter~=1
        if RMSE(iter-1) < RMSE(iter)
            break;
        end
    end
    if iter~=1
        step_MSE = RMSE(iter-1) - RMSE(iter);
        if step_MSE < 1e-4 %0.000001
            break;
        end
    end

end
Out_X = X;
end