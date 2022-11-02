function RegressionModel = NonLinRegression(x,y,basis_props,method)

if nargin < 4
    method = 'ols';
end

n = size(y,1);
m = size(x,2);

% Construct regression basis
p_max = basis_props.max_order;
p = prod(p_max);
basis_type = basis_props.type;
if ~isfield(basis_props,'basis_indices')

    ba = true(p,1);
    
else
    ba = basis_props.basis_indices;
end

if m==1
    Phi = basis(x,p_max,basis_type);
else
    Phi = tensorbasis(x,p_max,basis_type);
end
Phi = Phi(:,ba);
p = size(Phi,2);

% Calculate regression coefficients
switch method
    case 'ols'
        beta = Phi \ y;                                                             % OLS parameter estimates
    case 'qr'
        [Q,R] = qr(Phi,'econ');
        beta = R\(Q'*y);
    case 'pc'
        [U,S,V] = svd(Phi,'econ');
        D = V*diag(1./diag(S));
        beta = D*U'*y;
end

% Calculate error figures
y_hat = Phi*beta;                                                           % Response predictions
err = y - y_hat;                                                            % Prediction error (residuals)
sigmaW2 = var(err);                                                         % Residual variance estimate

% Performance criteria
criteria.MSE = (1/n)*sum(err.^2);                                           % Mean squared error
criteria.RSS = sum(err.^2);                                                 % Residual Sum of Squares
TSS = sum(y.^2);                                                            % Total Sum of Squares
criteria.R2 = 1 - criteria.RSS./TSS;                                        % Coefficient of determination (R-squared)
criteria.F_stat = ( (TSS-criteria.RSS)/p ) / ( criteria.RSS/(n-p-1) );      % F-statistic
criteria.BIC = (1/n)*(criteria.RSS+log(n)*p*sigmaW2);                       % Bayesian information criterion
criteria.adjR2 = 1 - (criteria.RSS/(n-p-1))/(TSS/(n-1));                    % Adjusted R-squared

% Calculate LOO error
switch method
    case 'ols'
        H = Phi*pinv(Phi);
    case 'qr'
        H = Q*Q';
    case 'pc'
        H = U*U';
end
h = diag(H);
criteria.MSEloo = (1/n)*sum( (err./(1-h)).^2 );                             % Leave-One-Out MSE

% Calculate statistics on the parameter estimates
switch method
    case 'ols'
        Lambda = Phi'*Phi*(1/sigmaW2);                                              % Precision matrix of parameter estimates
        SigmaBeta = pinv(Lambda);                                                   % Covariance matrix of parameter estimates
    case 'qr'
        Lambda = R'*R*(1/sigmaW2);
        SigmaBeta = pinv(Lambda);                                                   % Covariance matrix of parameter estimates
    case 'pc'
        Lambda = sigmaW2*(V*S)*(S*V');
        SigmaBeta = sigmaW2*(D*D');
end

seBeta = sqrt(diag(SigmaBeta));                                             % Standard error of parameter estimates
t_stat = abs(beta./seBeta);                                                 % t-statistic of each parameter estimate

% Packing the output
RegressionModel.beta = beta;
RegressionModel.Parameters.ParameterVector = beta;
RegressionModel.sigmaW2 = sigmaW2;
RegressionModel.Parameters.CovarianceMat = SigmaBeta;
RegressionModel.Parameters.PrecisionMat = Lambda;
RegressionModel.Parameters.StandardError = seBeta;
RegressionModel.Parameters.t_statistic = t_stat;
RegressionModel.Performance = criteria;

RegressionModel.BasisProperties = basis_props;
RegressionModel.Method = method;
