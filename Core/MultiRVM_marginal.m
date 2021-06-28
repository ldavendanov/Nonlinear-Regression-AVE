function [lnL,criteria] = MultiRVM_marginal(X,Y,basis_opt,HyperPar)
%--------------------------------------------------------------------------
% This function calcualtes the marginal likelihood of the Bayesian
% Non-Linear Multivariate Regression for data consisting of inputs "x" and
% outputs "y". Basis properties are summarized in "basis_opt", while prior
% hyperparameters are summarized in the structure "HyperPar".
% 
% Created by : Luis David Avendano-Valencia - April 2020
%
%--------------------------------------------------------------------------

%- Matrix sizes
n = size(Y,1);
[~,N] = size(X);

%- Extract hyperparameters
Lambda = HyperPar.Lambda;
V = HyperPar.V;
nu = HyperPar.nu;
l = HyperPar.L;

%- Calculating the representation basis
f = [ones(1,N); sqexp_kern( basis_opt.Xi, X, l )];

%- Prediction error
DeltaY = Y;

%- Covariance matrices
SigmaE = V/(nu-n-1);
LambdaN = Lambda + f*f';
B = chol(LambdaN,'lower')\f;
A = eye(N) - B'*B;
D = svd(A);

%- Calculate the marginal likelihood
Dm = DeltaY*A*DeltaY';
lnL = -0.5*trace(SigmaE\Dm) + (n/2)*sum( log( D(D>1e-25*max(D)) ) ) - (N/2)*log(det(SigmaE));
criteria.rss = trace(DeltaY*DeltaY');
criteria.rss_sss = criteria.rss / trace( Y*Y' );

%- Calculate the partial derivatives
if nargout == 3
    dlnL = zeros(n+p,1);
    S = zeros(N);
    for i=1:n
        dlnL(i) = 0.5*( Dm(i,i)/SigmaE(i,i)^2 - N/SigmaE(i,i) );
        S = S + ( DeltaY(i,:)'*DeltaY(i,:) )/ SigmaE(i,i);
    end
    dlnL_dA = 0.5*( A*S*A - n*A );
    for i=1:p
        dlnL(i+n) = -(Lambda(i,i)^2)\f(i,:)*dlnL_dA*f(i,:)';
    end
end