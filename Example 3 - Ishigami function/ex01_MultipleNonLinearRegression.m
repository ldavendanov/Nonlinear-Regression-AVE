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
x = 2*pi*rand(N,n) - pi;                                                    % Simulating inputs
f = sin( x(:,1) ) + a*sin( x(:,2) ).^2 + b*x(:,3).^4.*sin( x(:,1) );        % Ishigami function
y = f + sqrt(sigmaW2)*randn(N,1);

c = cvpartition(N,'HoldOut',0.5);           % Split training and validation using hold-out method

% Training samples
X = x(c.training,:);
Y = y(c.training);

%% Evaluating the best structure for non-linear regression
close all
clc

p_max = 5;
MSE = zeros(p_max,2);
R2 = zeros(p_max,2);
F_stat = zeros(p_max,1);
BIC = zeros(p_max,1);

for i=1:p_max
    basis_props.max_order = i*[1 1 1];
    basis_props.basis_indices = true(prod(basis_props.max_order),1);
    basis_props.type = 'hhh';
    RegModel = NonLinRegression(X,Y,basis_props,'qr');
    MSE(i,:) = [RegModel.Performance.MSE RegModel.Performance.MSEloo];
    R2(i,:) = [RegModel.Performance.R2 RegModel.Performance.adjR2];
    F_stat(i) = RegModel.Performance.F_stat;
    BIC(i) = RegModel.Performance.BIC;
end

figure('Position',[100 100 900 600])
subplot(221)
semilogy( 1:p_max, MSE,'-o')
xlabel('Basis order')
ylabel('MSE')
legend('Training','LOO')
grid on

subplot(222)
plot(1:p_max,R2,'-o')
xlabel('Basis order')
ylabel('Coefficient of determination')
legend('R^2','Adjusted R^2')
grid on

subplot(223)
semilogy(1:p_max,F_stat,'-o')
xlabel('Basis order')
ylabel('F-statistic')
grid on

subplot(224)
plot(1:p_max,BIC,'-o')
xlabel('Basis order')
ylabel('BIC')
grid on

%% Predict based on optimal model
close all
clc

% x_ast = linspace(-1,1,50)';
% [X1_ast,X2_ast,X3_ast] = ndgrid(x_ast);
% 
% X_ast = [X1_ast(:) X2_ast(:) X3_ast(:)];

pa = 5;
basis_props.max_order = pa*[1 1 1];
basis_props.basis_indices = true(prod(basis_props.max_order),1);
basis_props.type = 'hhh';
RegModel = NonLinRegression(X,Y,basis_props,'qr');
[y_hat,~] = NL_predict(x,RegModel);

clr = lines(2);

figure
plot(y,y_hat,'.')
grid on

figure
bar(log10(RegModel.Parameters.t_statistic))

%% Optimizing basis
close all
clc

thr = sort(RegModel.Parameters.t_statistic);
MSE = zeros(numel(thr),2);
R2 = zeros(numel(thr),2);
Fstat = zeros(numel(thr),1);
BIC = zeros(numel(thr),1);
for i=1:numel(thr)
    basis_props.max_order = pa*[1 1 1];
    basis_props.basis_indices = RegModel.Parameters.t_statistic >= thr(i);
    basis_props.type = 'hhh';
    M = NonLinRegression(x,y,basis_props,'qr');

    MSE(i,:) = [M.Performance.MSE M.Performance.MSEloo];
    R2(i,:) = [M.Performance.R2 M.Performance.adjR2];
    Fstat(i) = M.Performance.F_stat;
    BIC(i) = M.Performance.BIC;

end

figure
subplot(221)
semilogy((1:numel(thr))-1,MSE,'-o')
grid on
xlabel('No. of removed bases')
ylabel('MSE')
legend({'Training','LOO'})

subplot(222)
plot((1:numel(thr))-1,R2,'-o')
grid on
xlabel('No. of removed bases')
ylabel('Coefficient of determination')
legend({'R^2','Adjusted R^2'})

subplot(223)
semilogy((1:numel(thr))-1,Fstat,'-o')
grid on
xlabel('No. of removed bases')
ylabel('F-statistic')

subplot(224)
semilogy((1:numel(thr))-1,BIC,'-o')
grid on
xlabel('No. of removed bases')
ylabel('BIC')

[~,ind_opt] = max(Fstat);
basis_props.basis_indices = RegModel.Parameters.t_statistic >= thr(ind_opt);

figure
imagesc(1:pa*pa,1:pa,reshape(basis_props.basis_indices,pa,pa*pa))
% set(gca,'XTickLabel',XTickLbl,'YTickLabel',YTickLbl)
axis xy