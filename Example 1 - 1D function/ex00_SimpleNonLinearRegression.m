clear
close all
clc

addpath('..\Core\')

%% Producing simulations of the sinc function
close all
clc

omega = 2*pi;
sigmaW2 = 1e-3;

n = 1;
N = 2e2;
x = 2*rand(N,1) - 1;
x = sort(x);
f = sin( omega*x )./(omega*x) + 0.2*atan(omega*x);
y = f + sqrt(sigmaW2)*randn(N,1);

c = cvpartition(N,'HoldOut',0.5);

X = x(c.training);
Y = y(c.training);

figure
plot(x,f)
hold on
plot(X,Y,'.')


%% Evaluating the best structure for non-linear regression
close all
clc

p_max = 20;
MSE = zeros(p_max,2);
R2 = zeros(p_max,2);
F_stat = zeros(p_max,1);
BIC = zeros(p_max,1);

for i=1:p_max
    basis_props.max_order = i;
    basis_props.basis_indices = 1:i;
    basis_props.type = 'h';
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

x_ast = linspace(-1,1)';

pa = 12;
basis_props.max_order = pa;
basis_props.basis_indices = 1:pa;
basis_props.type = 'h';
RegModel = NonLinRegression(X,Y,basis_props,'qr');
[y_hat,se_y] = NL_predict(x_ast,RegModel);

clr = lines(3);

xx = [x_ast; flipud(x_ast)];
yy = [y_hat+3*se_y; flipud(y_hat-3*se_y)];

figure
p3  = fill(xx,yy,brighten(clr(1,:),0.75),'LineStyle','none');
hold on
p2 = plot(x_ast,y_hat,'Color',clr(1,:),'LineWidth',2);
p1 = plot(x,y,'.','Color',clr(2,:));
grid on
xlabel('Predictor')
ylabel('Response')
legend([p1 p2 p3],{'Training data','Predictive mean','Standard error'})

figure
bar(basis_props.basis_indices, log10(RegModel.Parameters.t_statistic))