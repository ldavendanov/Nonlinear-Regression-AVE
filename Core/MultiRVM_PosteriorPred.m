function [Yh_post,Sigma_pred] = MultiRVM_PosteriorPred(X,basis_opt,PosteriorPar)
%--------------------------------------------------------------------------
% This function calculates the posterior predictive distribution of the
% function estimates provided by the Bayesian NLMR characterized by the
% basis described in 'basis_opt' and the posterior hyperparameter estimates
% 'PosteriorPar', for inputs given in 'X'.
%
% Created by : Luis David Avendaño-Valencia - January 2021
%--------------------------------------------------------------------------

[~,N] = size(X);

%- Calculating the representation basis
f = [ones(1,N); sqexp_kern( basis_opt.Xi, X, PosteriorPar.L )];

%- Calculate the predictive mean (posterior mean)
W = PosteriorPar.W;
Yh_post = W*f;

%- Calculate the predictive covariance
if nargout == 2
    n = size(W,1);
    SigmaE = PosteriorPar.V/( PosteriorPar.nu - n - 1 );
    Sigma_pred = zeros(n,n,N);
    L = chol(PosteriorPar.Lambda,'lower');
    D = L\f;
    for i=1:N
        Sigma_pred(:,:,i) = ( 1 + D(:,i)'*D(:,i) )*SigmaE;
    end
    if n==1
        Sigma_pred = squeeze(Sigma_pred);
    end
end