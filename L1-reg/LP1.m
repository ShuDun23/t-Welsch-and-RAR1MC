function [Out_X,Y4_RMSE,U,V] = LP1(M, M_Omega, Omega_array, rak, maxiter, p)
% l1-reg
[r,c] = size(M_Omega);
U = randn(r,rak);
V = randn(rak,c);
% RMSE = zeros(1, maxiter);
t = 25;
sum_Omega = sum(sum(Omega_array));
Y4_RMSE(1:maxiter)=zeros;
for iter = 1 : maxiter
    clear row col;
    for j = 1:c
        row = find(Omega_array(:,j) == 1);
        U_I =  U(row,:);
        b_I = M_Omega(row,j);
        
        [~,W_col] = size(b_I');
        if iter == 1
            W = eye(W_col);
        else
            ksi = U_I * V(:,j) - b_I;
            clear W;
            for W_index = 1 : W_col
                W(W_index,W_index) = 1/((abs(ksi(W_index,1)))^2 + 0.0001)^((1-p/2)/2);
            end
        end
        for inner_iter = 1 : t
            V(:,j) = pinv(U_I.' * W.' * W * U_I)* U_I.' * W.' * W * b_I;
            ksi = U_I * V(:,j) - b_I;
            clear W;
            for W_index = 1 : W_col
                W(W_index,W_index) = 1/((abs(ksi(W_index,1)))^2 + 0.0001)^((1-p/2)/2);
            end
        end
        clear U_I b_I;
    end
    clear row col;
    for i = 1:r
        col = find(Omega_array(i,:) == 1);
        V_I = V(:,col);
        b_I = M_Omega(i,col);
        
        [~,W_col] = size(b_I);
        ksi = U(i,:) * V_I - b_I;
        clear W;
        for W_index = 1 : W_col
            W(W_index,W_index) = 1/((abs(ksi(1,W_index)))^2 + 0.0001)^((1-p/2)/2);
        end
        
        for inner_iter = 1 : t
            U(i,:) = b_I * W * W.'* V_I.' * pinv(V_I * W * W.'* V_I.');
            ksi = U(i,:) * V_I - b_I;
            clear W;
            for W_index = 1 : W_col
                W(W_index,W_index) = 1/((abs(ksi(1,W_index)))^2 + 0.0001)^((1-p/2)/2);
            end
        end
        clear V_I b_I;
    end
    X = U*V;
    Y4_RMSE(iter) = norm((M-X).*Omega_array,'fro')/sqrt(sum_Omega);
    RMSE(1, iter) = norm(M-X,'fro')/nnz(Omega_array);
    if iter~=1
        if RMSE(iter-1) < RMSE(iter)
            break;
        end
    end
end
Out_X = X;
end