function [Yh_post,Sigma_pred] = BayesianNLMR_PosteriorPred(X,basis_opt,PosteriorPar)
%--------------------------------------------------------------------------
% This function calculates the posterior predictive distribution of the
% function estimates provided by the Bayesian NLMR characterized by the
% basis described in 'basis_opt' and the posterior hyperparameter estimates
% 'PosteriorPar', for inputs given in 'X'.
%
% Created by : Luis David Avendaño-Valencia - April 2020
%--------------------------------------------------------------------------

[m,N] = size(X);

%- Calculating the representation basis
if m>1
    f = tensorbasis(X,basis_opt);
else
    f = basis(X,basis_opt.order,basis_opt.type);
end

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
        Sigma_pred = squeeze(Sigma_pred)';
    end
end