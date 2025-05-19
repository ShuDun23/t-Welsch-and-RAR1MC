function [Out_X, Y1_RMSE,Y2_RMSE,peaksnr,U1,V1,NRE,PMD] = RAR1MC(M, M_Omega, array_Omega, maxiter, xi1, xi2)
%%%%%%%%%%%%%%%% INPUTS %%%%%%%%%%%%%%%%
% M - true data
% M_Omega - observed data with missing entries
% array_Omega - observed index, which is a matrix, '1' means observed entries
% M_Omega = M.*array_Omega;
% maxiter - maximum iteration number
% xi1, xi2 - two positive constants satisfying xi1 >= xi2/sqrt(2);
%%%%%%%%%%%%%%%% OUTPUTS %%%%%%%%%%%%%%%%

[m,n] = size(M_Omega);
Y1_RMSE = [];
Y2_RMSE = [];
NRE(1:maxiter)=zeros;
X = zeros(m, n);
peaksnr = [];
U = randn(m,1);
V = randn(1,n);
W = ones(m,n);
M_2 = M_Omega;
U1 = [];

% xi1 >= xi2/sqrt(2);
c = 2;
sigma = 1;

% tolaerance
delta = 1e-5;
xi = 0.001;

PMD = [];
RE = [];
SPMD = [];
for iter = 1 : maxiter % outer iteration
    M_1 = M_2;
    M_Omega = M_1;
    rak = iter;
    in2 = 0;

    while (in2 <2)     % inner iteration
        a = norm((M_Omega - U * V).*array_Omega,'fro')^(2)/norm(M_Omega.*array_Omega,'fro')^(2); % MSE

        if iter>1
            M_1= M_Omega - U1*V(1:iter-1,:); % Res
        end

        for j = 1:n
            row = find(array_Omega(:,j) == 1);
            U_I = U(row,:);
            b_I = M_Omega(row,j);
            W_I = diag(W(row,j));
            V(:,j) = pinv(U_I'*W_I*U_I)*(U_I'*W_I*b_I);
        end

        for i = 1:m
            U_bf = U(i,end);
            col = find(array_Omega(i,:) == 1);
            V_I = V(iter,col);
            b_I = M_1(i,col);
            W_I = diag(W(i,col));
            U(i,iter) = b_I*W_I*V_I'/(V_I*W_I*V_I');
        end

        D = M_Omega - U*V.*array_Omega;
        D_m_n = D(find(D));
        d = iqr(abs(D_m_n))/1.349;
        c = min([c xi1*d]);
        sigma = min([sigma xi2*d]);

        W=ones(m,n);
        ind=find(abs(D)>c);
        W(ind)=exp((c^2-abs(D(ind)).^2) / (sigma^2));

        b = norm((M_Omega - U * V).*array_Omega,'fro')^2/norm(M_Omega.*array_Omega,'fro')^2; % MSE
        if a-b < delta
            in2 = in2 + 1;
        end
    end % inner iteration ends

    U1 = U;
    V1 = V;
    U = [U1 randn(m,1)];
    % V = randn(iter+1,c);
    V = [V1;randn(1,n)];

    X = U1*V1;
    Y = X;
    iter
    Rank_X=rank(X)
    peaksnr = [peaksnr psnr(Y, M)];
    Y1_RMSE = [Y1_RMSE norm((M-X),'fro')/sqrt(m*n)];
    Y2_RMSE = [Y2_RMSE norm((M_Omega-X).*array_Omega,'fro')/sqrt(m*n)];
    NRE(iter) = (norm((M_Omega).*array_Omega,'fro'))^2/(norm((M_Omega-X).*array_Omega,'fro'))^2;

    if iter>1
        RE = [RE norm(M-X,'fro')/sqrt(m*n)];
        if iter>2
            if RE(end-1) - RE(end)<1e-4
                break
            end
        end
    end

end % outer iteration ends
Out_X = X;
end


