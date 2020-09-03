function [yhat,varY] = gpr_predict(x,X,Y,theta,Method,indices)

%-- Checking input
[n,N] = size(X);
[~,Nt] = size(x);

switch nargin
    case 3
        % Default GPR parameters
        theta(1) = 1e-2;
        theta(2) = 1;
        theta(3:n+3) = 1e-2*eye(n);
        
        % Default GPR covariance calculation method
        Method = 'full';
        indices = true(1,N);
        
    case 4
        % Default GPR covariance calculation method
        Method = 'full';
        indices = true(1,N);
        
end

%-- Parameters of the GPR
scale = theta(3:end);                                                       % Length-scale
sigmaF2 = theta(2);                                                         % Kernel variance
sigmaU2 = theta(1);                                                         % Noise variance
m = sum(indices);

%-- Initializing computation matrices
yhat = zeros(1,Nt);
varY = zeros(1,Nt);

switch Method
    case 'full'             % Full covariance matrix
        
        %-- Constructing the kernel of the training set
        K = sigmaF2*sqexp_kern( X, X, scale );                              % Kernel - no-noise
        Ky = K + sigmaU2*eye(N);                                            % Kernel plus noise
        
        %-- Cholesky decomposition of the kernel matrix
        L = chol(Ky,'lower');
        alpha = L'\(L\Y(:));
        
        for i=1:Nt
            %-- Construct the kernel on the evaluation set
            K_ast = sigmaF2*sqexp_kern( x(:,i), X, scale );
            k_ast = sigmaF2*sqexp_kern( x(:,i), x(:,i), scale );
            
            %-- Calculate the predictions and predictive variance
            yhat(i) = K_ast * alpha;
            V = L\K_ast';
            varY(i) = diag( k_ast - V'*V );
        end
        
    case {'SoR','PP'}              % Subset of Regressors and Projected Process
        
        %-- Constructing the kernel of the training set
        Xm = X(:,indices);                                                  % Inducing inputs
        X = [Xm X(:,~indices)];                                             % Remaining inputs
        Y = [Y(indices) Y(~indices)];
        Knm = sigmaF2*sqexp_kern( X, Xm, scale );                           % Kernel - no-noise
                
        % Calculate the covariance inverse approximation
        V = chol(Knm(1:m,:),'lower');                                       % Cholesky factor of the inducing input covariance
        A = [Knm; sqrt(sigmaU2)*V'];
        [Q,R] = qr(A);
        alpha = R \ Q(1:N,:)'*Y(:);
        Rinv = pinv(R);
        
        for i=1:Nt
            
            %-- Constructing the kernel in the evaluation set
            Kmast = sigmaF2*sqexp_kern( x(:,i), Xm, scale );
            
            %-- Calculate the predictions and predictive variance
            yhat(i) = Kmast * alpha;
            Beta = Kmast * Rinv;
            varY(i) = sigmaU2*diag( Beta*Beta' );
            
            if strcmp(Method,'PP')
                k_ast = sigmaF2*sqexp_kern( x(:,i), x(:,i), scale );
                Gamma = Kmast / V;
                S = diag( k_ast - Gamma*Gamma' );
                varY(i) = varY(i) + S;
            end
            
        end
        
    case 'SoD'      % Subset of Datapoints
        
        % Calculate the covariance inverse approximation
        Xm = X(:,indices);
        Kmm = sigmaF2*sqexp_kern( Xm, Xm, scale );                          % Kernel - no-noise
                
        Ky = Kmm + sigmaU2*eye(m);                                          % Kernel plus noise
        L = chol(Ky,'lower');
        alpha = L'\(L\Y(indices)');
        
        parfor i=1:Nt
            
            fprintf('Running case No. %5d of %5d\n',i,Nt)
            
            %-- Construct the kernel in the evaluation set
            K_ast = sigmaF2*sqexp_kern( x(:,i), Xm, scale );
            k_ast = sigmaF2*sqexp_kern( x(:,i), x(:,i), scale );
            
            %-- Calculate the predictions and predictive variance
            yhat(i) = K_ast * alpha;
            V = L\K_ast';
            varY(i) = diag( k_ast - V'*V );
            
        end
        
end
