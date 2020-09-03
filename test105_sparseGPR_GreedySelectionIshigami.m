clear
close all
clc

%% Producing a set of simulations based on the Ishigami function ----------

% Parameters of the Ishigami function
a = 0.7;
b = 0.1;

% Measurement noise
sigmaW2 = 1e-1;

% Monte-Carlo simulation of the Ishigami function
N = 4e2;                                                                    % Number of samples
x = 2*pi*rand(3,N) - pi;                                                    % Simulating inputs
f = sin( x(1,:) ) + a*sin( x(2,:) ).^2 + b*x(3,:).^4.*sin( x(1,:) );        % Ishigami function
y = f + sqrt(sigmaW2)*randn(1,N);

%% Optimize the GPR with full covariance matrix

%-- Optimizing based on the full covariance matrix
theta0 = [sigmaW2 1 10 10 10];
[hyperP1,lnL1] = optimize_gpr( x, y, theta0 );


%% Optimize the indices of inducing points for sparse GPR approximation
close all
clc

%-- Initial indices - set as random
m = 40;
[~,ind] = datasample(x,m,2,'Replace',false);                                % Random sample from the full data set
indices = false(1,N);
indices(ind) = true;

Niter = 100;

subplot(311)
plot([0 Niter],lnL1*[1 1],'k')
hold on

subplot(312)
bar(indices)

n = size(x,1);
hyperP = ones(n+2,1);

for i=1:Niter
    
    %-- Optimizing based on the Subset of Regressors approach
    [hyperP,lnL] = optimize_gpr( x, y, hyperP, 'PP', indices );
    
    %-- Calculating the predictive variance on the remaining data
    [yhat,varY] = gpr_predict(x,x,y,hyperP,'PP',indices);
    varY(indices) = 0;
    
    %-- Calculating the geometric distance from the current inducing points
    d = zeros(sum(indices),N);
    xm = x(:,indices);
    for j=1:sum(indices)
        err = xm(:,j) - x; 
        d = diag(err'*err);
    end
    
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
    plot(J)
	
    drawnow
    
end

%% Performance evaluation
close all
clc

Nsim = 20^2;
[x1,x2] = ndgrid(linspace(-pi,pi,sqrt(Nsim)));
x_sim = [x1(:) zeros(Nsim,1) x2(:)]';

%-- Prediction with the full model
yh{1} = gpr_predict(x_sim,x,y,hyperP1);

%-- Prediction with the sparse model
yh{2} = gpr_predict(x_sim,x,y,hyperP,'PP',indices);

figure('Position',[100 100 1200 450])
for i=1:2
    subplot(1,2,i)
    surf(x1,x2,reshape(yh{i},sqrt(Nsim),sqrt(Nsim)))
    hold on
    plot3(x(1,:),x(3,:),y,'.r')
end
