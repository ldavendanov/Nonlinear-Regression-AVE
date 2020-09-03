function [Yhat,Parameters,criteria] = BayesianNLMR_posterior(X,Y,basis_opt)
%--------------------------------------------------------------------------
% This function calculates the parameter vector and error covariance matrix
% of a Non-Linear Multivariate Regression via Maximum Likelihood for data
% consisting of inputs "x" and outputs "y". Basis properties are summarized
% in "basis_opt". 
% 
% Created by : Luis David Avendano-Valencia - April 2020
%
%--------------------------------------------------------------------------

%- Matrix sizes
[m,N] = size(X);

%- Calculating the representation basis
if m>1
    f = tensorbasis(X,basis_opt);
else
    f = basis(X,basis_opt.order,basis_opt.type);
end

%- Calculate parameter estimates and error covariance matrix
WN = Y*f'/(f*f');
Yhat = WN*f;
DeltaY = Y - Yhat;
SigmaE = cov(DeltaY');

%- Packing the output
Parameters.W = WN;
Parameters.SigmaE = SigmaE;

%- Performance criteria
criteria.rss_sss = diag(DeltaY*DeltaY')./diag(Y*Y');
criteria.CN = cond(f*f');
H = f'*((f*f')\f);
Err_loo = DeltaY ./ repmat(( 1- diag(H) )',size(Y,1),1);
criteria.rss_loo = diag( Err_loo*Err_loo' ) ./ diag( Y*Y' );