function [lnL,dL_dTh] = gpr_likelihood(X,Y,theta,Method,indices)

%-- Checking input
[n,N] = size(X);

switch nargin
    case 2
        % Default GPR parameters
        theta(1) = 1e-2;
        theta(2) = 1;
        theta(3:n+3) = 1e-2*eye(n);
        
        % Default GPR covariance calculation method
        Method = 'full';
        indices = true(1,N);
        
    case 3
        % Default GPR covariance calculation method
        Method = 'full';
        indices = true(1,N);
        
end

%-- Parameters of the GPR
scale = theta(3:end);                                                       % Length-scale
sigmaF2 = theta(2);                                                         % Kernel variance
sigmaU2 = theta(1);                                                         % Noise variance
m = sum(indices);

switch Method
    case 'full'     % Full covariance matrix
        
        %-- Constructing the kernel of the training set
        K = sigmaF2*sqexp_kern( X, X, scale );                              % Kernel - no-noise
        Ky = K + sigmaU2*eye(N);                                            % Kernel plus noise
        
        %-- Cholesky decomposition of the kernel matrix
        L = chol(Ky,'lower');
        alpha = L'\(L\Y(:));
        
        %-- Calculate the negative log marginal likelihood
        lnL = 0.5*( Y(:)'*alpha ) + trace( log(L) );
        
    case {'SoR','PP'}      % Subset of Regressors and Projected process
        
        %-- Constructing the kernel of the training set
        Xm = X(:,indices);                                                  % Inducing inputs
        X = [Xm X(:,~indices)];                                             % All inputs - sorted
        Ym = Y(indices);
        Y = [Ym Y(~indices)];                                               % All outputs - sorted
        Kmn = sigmaF2*sqexp_kern( Xm, X, scale );                           % Kernel: inducing inputs - all inputs
        
        % Calculate the covariance inverse approximation
        [Lmm,flag] = chol( Kmn(:,1:m), 'lower' );
        if ~flag
            C = Lmm\Kmn;
            A = chol( C*C' + sigmaU2*eye(m), 'lower' )\C;
            B = ( eye(N) - A'*A )/sigmaU2;
            alpha = B*Y(:);
            
            %-- Calculate the negative log marginal likelihood
            lnL = 0.5*( Y(:)'*alpha ) + 0.5*log( det(eye(m)+(C*C'/sigmaU2)) ) + N*0.5*log(sigmaU2);
        else
            lnL = Inf;
        end
        
        
    case {'PPvar'}      % Subset of Regressors and Projected process
        
        %-- Sorting inputs
        Xm = X(:,indices);                                                  % Inducing inputs
        X = [Xm X(:,~indices)];                                             % All inputs - sorted
        Ym = Y(indices);
        Y = [Ym Y(~indices)];                                               % All outputs - sorted
        
        %-- Constructing the kernel of the training set
        Kmn = sigmaF2*sqexp_kern( Xm, X, scale );                           % Kernel: inducing inputs - all inputs
        Knn = sigmaF2*sqexp_kern( X, X, scale );                            % Kernel: all inputs
        
        %-- Calculate the covariance approximation
        [U,S,V] = svd(Kmn(:,1:m));
        sigma = diag(S);
        ind = sigma >= 1e-20*max(sigma);
        Z = diag(1./sigma(ind));
        R = V(:,ind)*Z*U(:,ind)'*Kmn;
        K = Kmn'*R;
        
        %-- Calculate the covariance inverse approximation
        A = sigmaU2*Kmn(:,1:m) + Kmn*Kmn';        
        [U,S,V] = svd(A);
        sigma = diag(S);
        ind = sigma >= 1e-20*max(sigma);
        Z = diag(1./sigma(ind));
        B = ( eye(N) - Kmn'*V(:,ind)*Z*U(:,ind)'*Kmn )/sigmaU2;
        alpha = B*Y(:);
        
        %-- Calculate the marginal likelihood
        lnL = 0.5*( Y(:)'*alpha ) - 0.5*log(det(B)) + 0.5*trace( Knn - K )/sigmaU2;
        
    case 'SoD'      % Subset of Datapoints
        
        % Calculate the covariance inverse approximation
        Kmm = sigmaF2*sqexp_kern( X(:,indices), X(:,indices), scale );        % Kernel - no-noise
        Ky = Kmm + sigmaU2*eye(m);                                          % Kernel plus noise
        L = chol(Ky,'lower');
        alpha = L'\(L\Y(indices)');
        
        %-- Calculate the marginal likelihood
        lnL = 0.5*( Y(indices)*alpha ) + trace( log(L) );

end
    
if nargout == 2
   
    switch Method
        case 'full'
            %-- Matrix of partial derivatives of the kernel wrt the hyperparameters
            D = zeros(N,N,n+2);
            D(:,:,1) = eye(N);
            D(:,:,2) = K/sigmaF2;
            for i=1:n
                D(:,:,i+2) = -0.5*( X(i,:) - X(i,:)' ).^2.*K;
            end
            
            dL_dTh = zeros(n+2,1);
            for i=1:n+2
                M1 = (alpha*alpha')*D(:,:,i);
                M2 = L'\(L\D(:,:,i));
                dL_dTh(i) = -0.5*trace( M1 - M2 );
            end
            
        case {'SoR','PP'}

            [U,S,V] = svd(Kmn(:,1:m));
            sigma = diag(S);
            ind = sigma >= 1e-15*max(sigma);
            Z = diag(1./sigma(ind));
            R = V(:,ind)*Z*U(:,ind)'*Kmn;
            K = Kmn'*R;
            
            %-- Matrix of partial derivatives of the kernel wrt the hyperparameters
            D = zeros(N,N,n+2);
            D(:,:,1) = eye(N);
            D(:,:,2) = K/sigmaF2;
            for i=1:n
                Delta = -0.5*( X(i,:) - Xm(i,:)' ).^2.*Kmn;
                B1 = R;
                C1 = Delta'*B1;
                C2 = -B1'*Delta(:,1:m)*B1;
                D(:,:,i+2) = C1 + C2 + C1';
            end
            
            dL_dTh = zeros(n+2,1);
            for i=1:n+2
                M1 = (alpha*alpha')*D(:,:,i);
                M2 = B*D(:,:,i);
                dL_dTh(i) = -0.5*trace( M1 - M2 );
            end
            
        case 'SoD'
            
            %-- Matrix of partial derivatives of the kernel wrt the hyperparameters
            D = zeros(m,m,n+2);
            D(:,:,1) = eye(m);
            D(:,:,2) = Kmm/sigmaF2;
            for i=1:n
                D(:,:,i+2) = -0.5*( X(i,indices) - X(i,indices)' ).^2.*Kmm;
            end
            
            dL_dTh = zeros(n+2,1);
            for i=1:n+2
                M1 = (alpha*alpha')*D(:,:,i);
                M2 = L'\(L\D(:,:,i));
                dL_dTh(i) = -0.5*trace( M1 - M2 );
            end
             
    end
    
end