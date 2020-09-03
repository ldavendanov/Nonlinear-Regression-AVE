clear
close all
clc

addpath('..\Core\')

%% Producing simulations of the sinc function

omega = 2*pi;
sigmaW2 = 1e-4;

n = 2;
N = 4e2;
x = 2*rand(2,N) - 1;
L = 0.5*[2 0; 0 1];
r = diag(x'*L*x);
f = sin( omega*r )./(omega*r) + 0.1*atan( omega*r );
y = f' + sqrt(sigmaW2)*randn(1,N);

Ntrain = N;
X = x(:,1:Ntrain);
Y = y(1:Ntrain);

figure
subplot(211)
plot(x(1,:),y,'.')

subplot(212)
plot(x(2,:),y,'.')

%% Extracting a sub-sample for the sparse GPR method
close all
clc

m = 50;
ind = UniformSpaceSampling(X,m);                                            % Random sample from the full data set
indices = false(1,Ntrain);
indices(ind) = true;

plot(X(1,:),X(2,:),'.','Color',0.75*ones(3,1))
hold on
plot(X(1,ind),X(2,ind),'X')

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

Ngrid = 20;
[x1,x2] = ndgrid(linspace(-1,1,Ngrid));
x_test = [x1(:)'; x2(:)'];
r = diag(x_test'*L*x_test);
f_test = sin( omega*r )./(omega*r) + 0.1*atan( omega*r );
y_test = f_test' + sqrt(sigmaW2)*randn(1,Ngrid^2);

%-- Predictions from the full covariance GPR
tic
[yh{1},varY{1}] = gpr_predict( x_test, X, Y, hyperP1 );
toc

%-- Predictions from the Subset of Regressors method
tic
[yh{2},varY{2}] = gpr_predict( x_test, X, Y, hyperP2, Method1, indices );
toc

%-- Predictions from the Subset of Data method
tic
[yh{3},varY{3}] = gpr_predict( x_test, X, Y, hyperP3, Method2, indices );
toc

%-- Predictions from the Projected Process method
Method3 = 'PP';
tic
[yh{4},varY{4}] = gpr_predict( x_test, X, Y, hyperP2, Method3, indices );
toc

%% Plotting the results
close all
clc


figure('Position',[100 100 1200 900])

subplot(2,3,1)
surf(x1,x2,reshape(f_test,Ngrid,Ngrid))
view(30,45)
for i=1:4
    subplot(2,3,i+1)
    surf(x1,x2,reshape(yh{i},Ngrid,Ngrid))
    view(30,45)
end