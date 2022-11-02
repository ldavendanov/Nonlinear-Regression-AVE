function [y_hat,se_y] = NL_predict(x,RegModel)

m = size(x,2);

beta = RegModel.beta;
sigmaW2 = RegModel.sigmaW2;
SigmaBeta = RegModel.Parameters.CovarianceMat;
basis_props = RegModel.BasisProperties;

% p_max = basis_props.max_order;
% basis_type = basis_props.type;
% if ~isfield(basis_props,'basis_indices')
%     ba = 1:p_max;
% else
%     ba = basis_props.basis_indices;
% end
% Phi = basis(x,p_max,basis_type,ba);

% Construct regression basis
p_max = basis_props.max_order;
basis_type = basis_props.type;
if ~isfield(basis_props,'basis_indices')
    ba = true(prod(p_max),1);
else
    ba = basis_props.basis_indices;
end

if m==1
    Phi = basis(x,p_max,basis_type);
else
    Phi = tensorbasis(x,p_max,basis_type);
end
Phi = Phi(:,ba);

y_hat = Phi*beta;
delta2 = diag( Phi*SigmaBeta*Phi' );
sigmaY2 = sigmaW2*( delta2+1 );
se_y = sqrt(sigmaY2);
