function [HyperPar,lnL] = OptimizeBayesianNLMR( X, Y, basis_opt, CovStructure )

if nargin < 4
    CovStructure = 'diag';
end

n = size(Y,1);
p = prod(basis_opt.order);

problem.objective = @(theta) logMarginal( X, Y, basis_opt, theta, CovStructure );

switch CovStructure
    case 'diag'
        
        problem.x0 = [1e0*ones(n,1); 1e-2*ones(p,1)];
        problem.lb = 1e-20*ones(n+p,1);
        problem.ub = 1e20*ones(n+p,1);
        problem.solver = 'fmincon';
        problem.options = optimoptions('fmincon',...
            'Display','iter','SpecifyObjectiveGradient',true,'CheckGradients',false,...
            'UseParallel',true);
%         problem.options = optimoptions('fmincon','PlotFcn','optimplotfval',...
%             'Display','iter','SpecifyObjectiveGradient',true,'CheckGradients',false,...
%             'UseParallel',true);
        
        [theta,lnL] = fmincon(problem);
        lnL = -lnL;
        
        HyperPar.Wo = zeros(n,p);
        HyperPar.V = diag(theta(1:n));
        HyperPar.Lambda = diag(theta(n+1:end));
        HyperPar.nu = n+2;
    
    case 'scalar'
        
        problem.x0 = [1e0; 1e-2];
        problem.lb = 1e-20*ones(2,1);
        problem.ub = 1e20*ones(2,1);
        problem.solver = 'fmincon';
        problem.options = optimoptions('fmincon','PlotFcn','optimplotfval',...
            'Display','iter','SpecifyObjectiveGradient',true,'CheckGradients',false);
        
        [theta,lnL] = fmincon(problem);
        lnL = -lnL;
        
        HyperPar.Wo = zeros(n,p);
        HyperPar.V = theta(1)*eye(n);
        HyperPar.Lambda = theta(2)*eye(p);
        HyperPar.nu = n+2;
        
end

function [lnL,dlnL] = logMarginal( X, Y, basis_opt, theta, CovStructure )

n = size(Y,1);
p = prod( basis_opt.order );

switch CovStructure
    case 'diag'
        HyperPar.Wo = zeros(n,p);
        HyperPar.V = diag(theta(1:n));
        HyperPar.Lambda = diag(theta(n+1:end));
        HyperPar.nu = n+2;
        
    case 'scalar'
        HyperPar.Wo = zeros(n,p);
        HyperPar.V = theta(1)*eye(n);
        HyperPar.Lambda = theta(2)*eye(p);
        HyperPar.nu = n+2;
end

[lnL,~,dlnL] = BayesianNLMR_marginal(X,Y,basis_opt,HyperPar);
lnL = -lnL;
dlnL = -dlnL;

if strcmp(CovStructure,'scalar')
    d(1) = sum(dlnL(1:n));
    d(2) = sum(dlnL(n+1:end));
    dlnL = d;
end