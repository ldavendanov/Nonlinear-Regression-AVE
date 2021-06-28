function [HyperPar,lnL] = OptimizeMultiRVM( X, Y, basis_opt, CovStructure )

if nargin < 4
    CovStructure = 'diag';
end

m = size(X,1);
n = size(Y,1);
p = basis_opt.order;

problem.objective = @(theta) logMarginal( X, Y, basis_opt, theta, CovStructure );

switch CovStructure
    case 'diag'
        
        problem.x0 = [1e0*ones(n,1); 1e-2*ones(p+1,1); ones(m,1)];
        problem.lb = 1e-20*ones(n+p+m+1,1);
        problem.ub = 1e20*ones(n+p+m+1,1);
        problem.solver = 'fmincon';
        problem.options = optimoptions('fmincon','PlotFcn','optimplotfval',...
            'Display','iter','SpecifyObjectiveGradient',false,'CheckGradients',false,...
            'UseParallel',true);
        
        [theta,lnL] = fmincon(problem);
        lnL = -lnL;
        
        HyperPar.Wo = zeros(n,p+1);
        HyperPar.V = diag(theta(1:n));
        HyperPar.Lambda = diag(theta(n+1:n+p+1));
        HyperPar.nu = n+2;
        HyperPar.L = theta(n+p+2:end);
    
    case 'scalar'
        
        problem.x0 = [1e0; 1e-2; 1e0*ones(m,1)];
        problem.lb = 1e-20*ones(2+m,1);
        problem.ub = 1e20*ones(2+m,1);
        problem.solver = 'fmincon';
        problem.options = optimoptions('fmincon','PlotFcn','optimplotfval',...
            'Display','iter','SpecifyObjectiveGradient',false,...
            'CheckGradients',false,'UseParallel',true);
        
        [theta,lnL] = fmincon(problem);
        lnL = -lnL;
        
        HyperPar.Wo = zeros(n,p+1);
        HyperPar.V = theta(1)*eye(n);
        HyperPar.Lambda = theta(2)*eye(p+1);
        HyperPar.L = theta(3:end);
        HyperPar.nu = n+2;
        
end

function lnL = logMarginal( X, Y, basis_opt, theta, CovStructure )

% m = size(X,1);
n = size(Y,1);
p = basis_opt.order;

switch CovStructure
    case 'diag'
        HyperPar.Wo = zeros(n,p+1);
        HyperPar.V = diag(theta(1:n));
        HyperPar.Lambda = diag(theta(n+1:n+p+1));
        HyperPar.L = theta(n+p+2:end);
        HyperPar.nu = n+2;
        
    case 'scalar'
        HyperPar.Wo = zeros(n,p+1);
        HyperPar.V = theta(1)*eye(n);
        HyperPar.Lambda = theta(2)*eye(p+1);
        HyperPar.L = theta(3:end);
        HyperPar.nu = n+2;
end

lnL = MultiRVM_marginal(X,Y,basis_opt,HyperPar);
lnL = -lnL;
% dlnL = -dlnL;

% if strcmp(CovStructure,'scalar')
%     d(1) = sum(dlnL(1:n));
%     d(2) = sum(dlnL(n+1:end));
%     dlnL = d;
% end