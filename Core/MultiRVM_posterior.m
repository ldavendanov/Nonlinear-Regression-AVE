function [Yh_post,PosteriorPar,criteria] = MultiRVM_posterior(X,Y,basis_opt,HyperPar)
%--------------------------------------------------------------------------
% This function performs the Bayesian Non-Linear Multivariate Regression
% for data consisting of inputs "x" and outputs "y". Basis properties are
% summarized in "basis_opt", while prior hyperparameters are summarized in
% the structure "HyperPar".
% 
% Created by : Luis David Avendano-Valencia - April 2020
%
%--------------------------------------------------------------------------

%- Matrix sizes
[~,N] = size(X);

%- Extract hyperparameters
Lambda = HyperPar.Lambda;
V = HyperPar.V;
nu = HyperPar.nu;
l = HyperPar.L;

%- Calculating the representation basis
f = [ones(1,N); sqexp_kern( basis_opt.Xi, X, l )];

%- Calculate the parameters of the posterior
LambdaN = Lambda + f*f';
L = chol(LambdaN,'lower');
WN = ( ( Y*f' ) / L' ) / L;
nuN = nu + N;
Yh_post = WN*f;
DeltaY = Y - Yh_post;
DeltaW = WN;
VN = V + DeltaW*Lambda*DeltaW' + DeltaY*DeltaY';

%- Packing the output
PosteriorPar.W = WN;
PosteriorPar.Lambda = LambdaN;
PosteriorPar.nu = nuN;
PosteriorPar.V = VN;
PosteriorPar.L = l;

%- Performance criteria
criteria.rss_sss = diag(DeltaY*DeltaY')./diag(Y*Y');
criteria.CN = cond(LambdaN);
B = L\f;
H = B'*B;
Err_loo = DeltaY ./ repmat(( 1- diag(H) )',size(Y,1),1);
criteria.rss_loo = diag( Err_loo*Err_loo' ) ./ diag( Y*Y' );