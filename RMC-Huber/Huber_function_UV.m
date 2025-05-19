function [beta error] = Huber_function_UV(a,b,c, alpha)
% y is a column
% X is a matrix
% initial_guess is a column

y = a;
X = b;
[~,sn]=size(X);
% alpha = 0.7102;
maxiter = 1500;

initial_guess = randn(sn,1);
eps = 1e-6; % for termination condition
counter = 0;
X_plus = pinv(X);

beta = initial_guess;

% initail scale
r = y-X*beta;
scale = 1.4815*median(abs(r-median(r)));
error = [];

while counter < maxiter
    % update residuals
    r = y-X*beta;
    % update pseudo residuals
    r_1=r/scale;
%     Psi=[];
%     for k=1:length(r_1)
%         Psi = [Psi;Huber_psi(c,r_1(k))];
%     end
%     r_pseu = Psi*scale;
    f=@(x) x.*(abs(x)<=c)+c*sign(x).*(abs(x)>c);
    r_pseu = f(r_1)*scale;
    % update scale
    scale=sqrt(r_pseu'*r_pseu/(length(y)*2*alpha));
    scale = 1.4815*median(abs(r-median(r)));
    % reupdate pseudo residuals
    r_2=r_pseu/scale;
    
    r_pseu_1=f(r_2)*scale;
    % compute the regression update
    delta = X_plus*r_pseu_1;
    % update the regression vector beta
    beta=beta+delta;
    % compute the convergence criterion
    crit=norm(delta)/norm(beta);
    % check the condition of termination
    error = [error crit];
    
%     if beta>1
%         beta=1;
%         break;
%     elseif beta<0
%         beta=0.01;
%         break;
%     end
%         
    if (crit < eps)
        break;
    end
    counter = counter + 1;
end

end