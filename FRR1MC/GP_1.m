function [Y, RMSE,PSNR_M] = GP_1(M, M_Omega, maxiteration, array_Omega)
% FRR1MC
[m ,n] = size(M_Omega);
X_j = zeros(m, n);
% u_j = randn(m, 1);
RMSE = [];
in = 0;
PSNR_M = [];
E =  zeros(m,n);
scale_max =[];
ieta_1 = [];
sum_Omega = sum(sum(array_Omega));
v_j = randn(n, 1);
    
for iter = 1 : maxiteration % outer iteration
    R_j = M_Omega - X_j.*array_Omega; 

    %% calculate u_j
    in2 = 0;    
    
    R_j_1 = R_j - E;
    b = 0;
    while (in2 < 1) % inner iteration
        a = b;
        for i = 1 : m
            g_l = R_j_1(i,:).';
            col = find(array_Omega(i,:) == 1);
            v_m = v_j(col);
            u_j(i,1) = v_m'*g_l(col)/(v_m'*v_m);
        end
        %% calculate v_j
        for j = 1 : n
            h_q = R_j_1(:,j);
            row = find(array_Omega(:,j) == 1);
            u_m = u_j(row);
            v_j(j,1) = u_m'*h_q(row)/(u_m'*u_m);
        end
        
        T = R_j - u_j * v_j'.*array_Omega;
        
      
%         t_m_n = T(find(T));
%         scale = 3*(1.06*min(1.4815*median(abs(t_m_n - median(t_m_n))),std(t_m_n)));
%         me = median(scale);
%         E = T - T.*exp(-(T-me).^2./(2*scale^2));


        scale = [];
        for j = 1:n
            t_m = T(:,j);
            t_m_n = t_m(t_m~=0);
            scale =  [scale 3*(1*min(1.4815*median(abs(t_m_n - median(t_m_n))),std(t_m_n)))];
%             scale =  [scale 5*std(t_m_n)];
        end
        
        sigma = ones(m,n)*diag(2*scale.^2);
        E = T - T.*exp(-T.*T./sigma);

        R_j_1 = R_j - E;
        
        b = u_j * v_j';
        if norm(a-b,'fro')^2/norm(a+0.00001,'fro')^2 < 0.00005
            in2 = in2 + 1;
        end
    end % the end of inner iteration
    scale_max =[scale_max max(scale)];
    X_j = X_j + u_j * v_j.';
    v_j = randn(n, 1);
    PSNR_M =[PSNR_M psnr(X_j, M)];
    %% judgement
    % RMSE = [RMSE, norm((M-X_j).*array_Omega,'fro')/sqrt(sum_Omega)];
    RMSE = [RMSE, norm((M-X_j),'fro')/sqrt(m*n)];
    Y = X_j;
    if iter~=1
        step_MSE = RMSE(iter-1) - RMSE(iter);
        if step_MSE < 1e-4 %0.00001
            in = in  + 1;
        end
        if in > 1
            break;
        end
    end

end % the end of outer iteration

end