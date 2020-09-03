clear
close all
clc

addpath('..\Core\')

%% Producing simulations of the sinc function

omega = 2*pi;
sigmaW2 = 1e-3;

n = 1;
N = 200;
X = 2*rand(1,N)-1;
X = sort(X);
f = sin( omega*X )./(omega*X) + 0.1*atan(omega*X);
Y = f + sqrt(sigmaW2)*randn(1,N);

Ntrain = N;

%% Estimation based on the Bayesian NLMR method

%- Fixing the basis options
basis_opt.order = 20;
basis_opt.type = 'h';
p = basis_opt.order;

%- Optimize hyperparameter values
[HyperPar,lnL] = OptimizeBayesianNLMR( X, Y, basis_opt );

%- Calculate the parameters of the posterior
[~,PosteriorPar,criteria] = BayesianNLMR_posterior(X,Y,basis_opt,HyperPar);

%- Calculate the posterior predictive distribution
[yh_pred,sigma_pred] = BayesianNLMR_PosteriorPred(X,basis_opt,PosteriorPar);

%% Plotting results
close all
clc

clr = lines(2);

XX = [X flip(X)];
YY = [yh_pred+3*sqrt(sigma_pred') flip(yh_pred-3*sqrt(sigma_pred'))];

figure('Position',[100 100 900 480])
p0 = fill(XX,YY,brighten(clr(1,:),0.95),'LineStyle','none');
hold on
p1 = plot(X,yh_pred,'Color',clr(1,:),'LineWidth',2);
p2 = plot(X,f,'--k','LineWidth',1.5);
p3 = plot(X,Y,'.','Color',clr(2,:));
grid on
legend([p2 p3 p1 p0],{'Actual','Noisy data','Mean Bayesian NLMR','95% CI Bayesian NLMR'})
set(gca,'FontName','Times New Roman','FontSize',12)
xlabel('$\xi$','Interpreter','latex')
ylabel('$f(\xi)$','Interpreter','latex')

figure('Position',[1000 100 600 480])
subplot(311)
bar(log10(abs(PosteriorPar.W)))
grid on
set(gca,'FontName','Times New Roman','FontSize',12)
ylabel('$\log_{10} w_i$','Interpreter','latex')

subplot(312)
bar( log10(diag(PosteriorPar.Lambda)) )
grid on
set(gca,'FontName','Times New Roman','FontSize',12)
ylabel('$\log_{10}| \lambda_i |$','Interpreter','latex')

subplot(313)
bar( log10( (PosteriorPar.W).^2 .* diag(PosteriorPar.Lambda)' ) )
grid on
set(gca,'FontName','Times New Roman','FontSize',12)
ylabel('$\log_{10} \lambda_i w_i^2$','Interpreter','latex')
xlabel('Basis index')