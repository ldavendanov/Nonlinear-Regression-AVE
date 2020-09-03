clear
close all
clc

%% Producing simulations of the sinc function

omega = 2*pi*3;
sigmaW2 = 1e-4;

n = 1;
N = 250;
X = 2*rand(1,N)-1;
X = sort(X);
f = sin( omega*X )./(omega*X) + 0.1*atan(omega*X);
Y = f + sqrt(sigmaW2)*randn(1,N);

Ntrain = N;

%% Optimize the GPR with full covariance matrix

%-- Optimizing based on the full covariance matrix
theta0 = [sigmaW2 1 10];
[hyperP1,lnL1] = optimize_gpr( X, Y, theta0 );


%% Optimize the indices of inducing points for sparse GPR approximation
close all
clc

%-- Initial indices - set as random
m = 10;
[~,ind] = datasample(X,m,2,'Replace',false);                                % Random sample from the full data set
indices = false(1,Ntrain);
indices(ind) = true;

Niter = 100;

subplot(311)
plot([0 Niter],lnL1*[1 1],'k')
hold on

subplot(312)
bar(indices)

theta0 = ones(n+2,1);
hyperP = theta0;

for i=1:Niter
    
    %-- Optimizing based on the Subset of Regressors approach
    [hyperP,lnL] = optimize_gpr( X, Y, hyperP, 'PP', indices );
    
    
%     lnL_new = zeros(Ntrain,1);
%     indx = 1:Ntrain;
%     indx = indx(~indices);
%     for j=1:sum(~indices)
%         ind = indices;
%         ind(indx(j)) = true;
%         lnL_new(indx(j)) = gpr_likelihood(X,Y,hyperP,'PP',ind);
%     end
%     
%     yhat = gpr_predict(X,X,Y,hyperP,'PP',indices);
%     
%     %-- Calculating the geometric distance from the current inducing points
%     d = (X(indices) - X').^2;
%     
%     J = -lnL_new + log(prod(d,2));
%     
%     [~,ind] = max(J);
%     indices(ind) = true;
    

    %-- Calculating the predictive variance on the remaining data
    [yhat,varY] = gpr_predict(X,X,Y,hyperP,'PP',indices);
    varY(indices) = 0;
    
    %-- Calculating the geometric distance from the current inducing points
    d = (X(indices) - X').^2;
    
    J = log(varY) + log(prod(d,2));
    
    [~,ind] = max(J);
    indices(ind) = true;
    
    subplot(311)
    plot(i,lnL,'.b')
    hold on
    xlim([0 Niter])
    
    subplot(312)
    bar(indices)
    
    subplot(313)
%     plot(J)
	plot(X,f(1:Ntrain),'k','LineWidth',2)
    hold on
    plot(X,yhat,'.')
    hold off    
    
    drawnow
    
end

