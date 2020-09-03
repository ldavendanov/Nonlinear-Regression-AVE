clear
close all
clc

%% Producing simulations of the sinc function

n = 2;
N = 8e2;
x = 2*rand(2,N) - 1;
[f,y] = sinc2D( x );

Ntrain = N;
X = x(:,1:Ntrain);
Y = y(1:Ntrain);

%% Estimation based on the Bayesian NLMR method

%- Fixing the basis options
basis_opt.order = [10 10];
basis_opt.type = 'll';

%- Optimize hyperparameter values
[HyperPar,lnL] = OptimizeBayesianNLMR( X, Y, basis_opt );

%- Calculate the parameters of the posterior
[Yh_post,PosteriorPar,criteria] = BayesianNLMR_posterior(X,Y,basis_opt,HyperPar);

%% Plot results
close all
clc

[X1,X2] = ndgrid( linspace(-1,1,40) );
x_test = [X1(:) X2(:)]';

f = sinc2D( x_test );
f = reshape(f,40,40);
y_hat = BayesianNLMR_PosteriorPred( x_test, basis_opt, PosteriorPar );
y_hat = reshape(y_hat,40,40);

figure('Position',[100 100 1200 480])
subplot(121)
surf(X1,X2,f)
xlabel('$\xi_1$','Interpreter','latex')
ylabel('$\xi_2$','Interpreter','latex')
zlabel('$f(\xi_1,\xi_2)$','Interpreter','latex')
set(gca,'FontName','Times New Roman','FontSize',12)

subplot(122)
surf(X1,X2,y_hat)
xlabel('$\xi_1$','Interpreter','latex')
ylabel('$\xi_2$','Interpreter','latex')
zlabel('$\hat{f}(\xi_1,\xi_2)$','Interpreter','latex')
set(gca,'FontName','Times New Roman','FontSize',12)

%% Parameter analysis
close all
clc

figure('Position',[100 100 600 480])
subplot(311)
plot( log10( abs( PosteriorPar.W ) ), '.', 'MarkerSize', 10 )
grid on
ylabel('$\log_{10} |w_i|$','Interpreter','latex')
set(gca,'FontName','Times New Roman','FontSize',12)

subplot(312)
plot( log10( diag( PosteriorPar.Lambda ) ), '.', 'MarkerSize', 10 )
grid on
ylabel('$\log_{10} \lambda_i$','Interpreter','latex')
set(gca,'FontName','Times New Roman','FontSize',12)

subplot(313)
plot( log10( PosteriorPar.W.^2.*diag( PosteriorPar.Lambda )' ), '.', 'MarkerSize', 10 )
grid on
ylabel('$\log_{10} \lambda_i w_i^2$','Interpreter','latex')
set(gca,'FontName','Times New Roman','FontSize',12)