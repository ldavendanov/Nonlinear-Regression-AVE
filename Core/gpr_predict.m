function [yhat,varY] = gpr_predict(x,X,Y,theta,Method,indices)

%-- Checking input
[n,N] = size(X);
M = size(x,2);

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
        
    case 5
        indices = SparseResampling(X,50);
        
end

%-- Parameters of the GPR
scale = theta(3:end);                                                       % Length-scale
sigmaF2 = theta(2);                                                         % Kernel variance
sigmaU2 = theta(1);                                                         % Noise variance
m = sum(indices);

switch Method
    case 'full'             % Full covariance matrix
        
        %-- Constructing the kernel of the training set
        K = sigmaF2*sqexp_kern( X, X, scale );                              % Kernel - no-noise
        Ky = K + sigmaU2*eye(N);                                            % Kernel plus noise
        
        %-- Cholesky decomposition of the kernel matrix
        L = chol(Ky,'lower');
        alpha = L'\(L\Y(:));
        
        %-- Calculate the predictions and predictive variance
        yhat = zeros(1,M);
        varY = zeros(1,M);
        for j=1:M
            K_ast = sigmaF2*sqexp_kern( x(:,j), X, scale );
            k_ast = sigmaF2*sqexp_kern( x(:,j), x(:,j), scale );
            
            yhat(j) = K_ast * alpha;
            if nargout == 2
                V = L\K_ast';
                varY(j) = k_ast - V'*V;
            end
        end
        
    case {'SoR','PP'}              % Subset of Regressors and Projected Process
        
        %-- Constructing the kernel of the training set
        Xm = X(:,indices);                                                  % Inducing inputs
        X = [Xm X(:,~indices)];                                             % Remaining inputs
        Y = [Y(indices) Y(~indices)];
        Kmn = sigmaF2*sqexp_kern( Xm, X, scale );                           % Kernel - no-noise
        Kmast = sigmaF2*sqexp_kern( Xm, x, scale );
        
        % Calculate the covariance inverse approximation
        [Lmm,~] = chol( sigmaU2*Kmn(:,1:m) + Kmn*Kmn', 'lower' );
        alpha = Lmm'\(Lmm\Kmn)*Y(:);
        
        %-- Calculate the predictions and predictive variance
        yhat = Kmast' * alpha;
        C = Lmm\Kmast;
        varY = sigmaU2*diag(C'*C);
        
        if strcmp(Method,'PP')
            for i=1:size(x,2)
                k_ast = sigmaF2*sqexp_kern( x(:,i), x(:,i), scale );
                S = k_ast - Kmast(:,i)'*(Kmn(:,1:m)\Kmast(:,i));
                varY(i) = varY(i) + S;
            end
        end
        
    case 'SoD'      % Subset of Datapoints
        
        % Calculate the covariance inverse approximation
        Kmm = sigmaF2*sqexp_kern( X(:,indices), X(:,indices), scale );        % Kernel - no-noise
        K_ast = sigmaF2*sqexp_kern( x, X(:,indices), scale );
        k_ast = sigmaF2*sqexp_kern( x, x, scale );
        
        Ky = Kmm + sigmaU2*eye(m);                                          % Kernel plus noise
        L = chol(Ky,'lower');
        alpha = L'\(L\Y(indices)');
        
        %-- Calculate the predictions and predictive variance
        yhat = K_ast * alpha;
        V = L\K_ast';
        varY = diag( k_ast - V'*V );

end