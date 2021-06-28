function [hyperpar,lnL] = optimize_gpr( X, Y, theta0, Method, indices )

switch nargin
    case 3
        Method = 'full';
        indices = [];
    case 4
        indices = SparseResampling(X,10);
end

n = size(X,1);
problem.objective = @(theta) gpr_likelihood( X, Y, theta, Method, indices );
problem.solver = 'fmincon';
problem.lb = 1e-16*ones(n+2,1);

if nargin >= 3
    problem.x0 = theta0;
else
    problem.x0 = 1e-3*ones(1,n+2);
end

problem.options = optimoptions('fmincon');
problem.options.Display = 'iter';
problem.options.PlotFcns = @optimplotfval;
problem.options.UseParallel = true;
if strcmp(Method,'full')
    problem.options.SpecifyObjectiveGradient = true;
end

[hyperpar,lnL] = fmincon(problem);