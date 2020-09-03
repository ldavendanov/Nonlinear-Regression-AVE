clear
close all
clc

addpath('..\Core\')

%% Producing simulations of the sinc function

omega = 8;
sigmaW2 = 1e-4;

n = 1;
N = 2e2;
x = 2*pi*rand(1,N) - pi;
f = sin( omega*x )./(omega*x) + 0.1*atan(omega*x);
y = f + sqrt(sigmaW2)*randn(1,N);

Ntrain = N/2;
X = x(1:Ntrain);
Y = y(1:Ntrain);

figure
plot(x/pi,y,'.')

%%
close all
clc

m = 20;
ind = UniformSpaceSampling(X,m);                                            % Random sample from the full data set
indices = false(1,Ntrain);
indices(ind) = true;

plot(X/pi,Y,'.')
hold on
plot(X(indices)/pi,0,'x')

%% Calculating the GPR with sparse covariance approximations
close all
clc

%-- Optimizing based on the full covariance matrix
theta0 = ones(n+2,1);
tic
[hyperP1,lnL1] = optimize_gpr( X, Y, theta0 );
toc

%%-- Optimizing based on the Subset of Regressors approach
Method1 = 'SoR';

tic
[hyperP2,lnL2] = optimize_gpr( X, Y, theta0, Method1, indices );
toc

%%-- Optimizing based on the Subset of Regressors approach
Method2 = 'SoD';
tic
[hyperP3,lnL3] = optimize_gpr( X, Y, theta0, Method2, indices );
toc

%% Evaluate the obtained models
close all
clc

clr = lines(3);

[X_test,indx] = sort(x);
Y_test = y(indx);

method_name = {'Full covariance';
               'Subset of Regressors';
               'Subset of Data';
               'Projected Process';};

%-- Predictions from the full covariance GPR
[yh{1},varY{1}] = gpr_predict( X_test, X, Y, hyperP1 );

%-- Predictions from the Subset of Regressors method
[yh{2},varY{2}] = gpr_predict( X_test, X, Y, hyperP2, Method1, indices );

%-- Predictions from the Subset of Data method
[yh{3},varY{3}] = gpr_predict( X_test, X, Y, hyperP3, Method2, indices );

%-- Predictions from the Projected Process method
Method3 = 'PP';
[yh{4},varY{4}] = gpr_predict( X_test, X, Y, hyperP2, Method3, indices );


figure('Position',[100 100 1200 900])
for k=1:4
    
    YY = [yh{k} + 3*sqrt(varY{k}); flip(yh{k} - 3*sqrt(varY{k}))];
    XX = [X_test flip(X_test)]/pi;
    
    subplot(2,2,k)
    
    fill(XX,YY,brighten(clr(2,:),0.8),'LineStyle','none')
    hold on
    plot(X_test/pi,f(indx),'LineWidth',1.5,'Color',zeros(1,3))
    hold on
    plot(X_test/pi,Y_test,'.','MarkerSize',10,'Color',0.75*ones(1,3))
    plot(X_test/pi,yh{k},'.','MarkerSize',10,'Color',clr(2,:))
    grid on
    
    title(method_name{k})
    
    if k>1
        plot(X(indices)/pi,-1,'xk','MarkerSize',10)
    end
    
    ylim([-1 1.5])
    
end