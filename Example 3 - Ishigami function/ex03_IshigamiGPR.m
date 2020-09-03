clear
close all
clc

addpath('..\Core\')

%% Producing a set of simulations based on the Ishigami function ----------

% Parameters of the Ishigami function
a = 0.7;
b = 0.1;

% Measurement noise
sigmaW2 = 1e-3;

% Monte-Carlo simulation of the Ishigami function
N = 4e2;                                                                    % Number of samples
n = 3;                                                                      % Dimensionality of input space
x = 2*pi*rand(n,N) - pi;                                                    % Simulating inputs
f = sin( x(1,:) ) + a*sin( x(2,:) ).^2 + b*x(3,:).^4.*sin( x(1,:) );        % Ishigami function
y = f + sqrt(sigmaW2)*randn(1,N);

%% Extracting a sub-sample for the sparse GPR method
close all
clc

m = 360;
ind = UniformSpaceSampling(x,m);                                            % Random sample from the full data set
indices = false(1,N);
indices(ind) = true;

plot(x(1,:),x(2,:),'.','Color',0.75*ones(3,1))
hold on
plot(x(1,ind),x(2,ind),'X')

%% Calculating the GPR with sparse covariance approximations
close all
clc

%-- Optimizing based on the full covariance matrix
theta0 = ones(n+2,1);
tic
[hyperP1,lnL1] = optimize_gpr( x, y, theta0 );
toc

%%-- Optimizing based on the Subset of Regressors approach
Method1 = 'SoR';
tic
[hyperP2,lnL2] = optimize_gpr( x, y, theta0, Method1, indices );
toc

%%-- Optimizing based on the Subset of Regressors approach
Method2 = 'SoD';
tic
[hyperP3,lnL3] = optimize_gpr( x, y, theta0, Method2, indices );
toc

%% Evaluate the obtained models
close all
clc

Ngrid = 20;
[x1,x2,x3] = ndgrid(linspace(-pi,pi,Ngrid));
x_test = [x1(:)'; x2(:)'; x3(:)'];
f_test = sin( x1(:) ) + a*sin( x2(:) ).^2 + b*x3(:).^4.*sin( x1(:) );       % Ishigami function

%-- Predictions from the full covariance GPR
tic
[yh{1},varY{1}] = gpr_predict( x_test, x, y, hyperP1 );
toc

%-- Predictions from the Subset of Regressors method
tic
[yh{2},varY{2}] = gpr_predict( x_test, x, y, hyperP2, Method1, indices );
toc

%-- Predictions from the Subset of Data method
tic
[yh{3},varY{3}] = gpr_predict( x_test, x, y, hyperP3, Method2, indices );
toc

%% Plotting the results
close all
clc

figure('Position',[100 100 1200 900])

Ftest = reshape(f_test,Ngrid,Ngrid,Ngrid);

k = 10;

subplot(2,2,1)
surf(squeeze(x2(k,:,:)),squeeze(x3(k,:,:)),squeeze(Ftest(k,:,:)))
view(30,45)
for i=1:3
    
    Yh = reshape(yh{i},Ngrid,Ngrid,Ngrid);
    
    subplot(2,2,i+1)
    surf(squeeze(x2(k,:,:)),squeeze(x3(k,:,:)),squeeze(Yh(k,:,:)))
    view(30,45)
end

figure
subplot(2,2,1)
surf(squeeze(x1(:,:,k)),squeeze(x2(:,:,k)),squeeze(Ftest(:,:,k)))
view(30,45)
for i=1:3
    
    Yh = reshape(yh{i},Ngrid,Ngrid,Ngrid);
    
    subplot(2,2,i+1)
    surf(squeeze(x1(:,:,k)),squeeze(x2(:,:,k)),squeeze(Yh(:,:,k)))
    view(30,45)
end