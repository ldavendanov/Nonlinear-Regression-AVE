clear
close all
clc

addpath('..\Core\')

%% Producing simulations of the 2D sinc function

omega = 2*pi;
sigmaW2 = 1e-4;

n = 2;
N = 8e2;
x = 2*rand(N,2) - 1;
L = 0.5*[2 0; 0 1];
r = diag(x*L*x');
f = sin( omega*r )./(omega*r);% + 0.1*atan( omega*r );
y = f + sqrt(sigmaW2)*randn(N,1);

figure
plot3(x(:,1),x(:,2),y,'.')
grid on

c = cvpartition(N,'HoldOut',0.25);           % Split training and validation using hold-out method

% Training samples
X = x(c.training,:);
Y = y(c.training);

%% Evaluating the best structure for non-linear regression
close all
clc

p_max = 12;
MSE = zeros(p_max,2);
R2 = zeros(p_max,2);
F_stat = zeros(p_max,1);
BIC = zeros(p_max,1);

for i=1:p_max
    basis_props.max_order = i*[1 1];
    basis_props.basis_indices = true(prod(basis_props.max_order),1);
    basis_props.type = 'hh';
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
[X1_ast,X2_ast] = ndgrid(x_ast);

X_ast = [X1_ast(:) X2_ast(:)];

pa = 9;
basis_props.max_order = pa*[1 1];
basis_props.basis_indices = true(prod(basis_props.max_order),1);
basis_props.type = 'hh';
RegModel = NonLinRegression(X,Y,basis_props,'qr');
[Y_hat,se_Y] = NL_predict(X_ast,RegModel);
[y_hat,~] = NL_predict(x,RegModel);

clr = lines(2);

figure
subplot(121)
surf(x_ast,x_ast,reshape(Y_hat,100,100)','LineStyle','none','FaceAlpha',0.75)
hold on
plot3(X(:,1),X(:,2),Y,'.','Color',clr(2,:))
zlim([-0.4 1.2])
set(gca,'CLim',[-0.25 1.2])

subplot(122)
imagesc(x_ast,x_ast,reshape(log10(se_Y),100,100)')
hold on
plot(X(:,1),X(:,2),'.','Color',clr(2,:))
colorbar

figure
plot(y,y_hat,'.')
grid on

XTickLbl = {'1','x'};
YTickLbl = {'1','x'};
for i=3:pa
    XTickLbl{i} = ['x_1^{',num2str(i-1),'}'];
    YTickLbl{i} = ['x_2^{',num2str(i-1),'}'];
end

figure
imagesc(1:pa,1:pa,reshape(log10(RegModel.Parameters.t_statistic),pa,pa))
set(gca,'XTickLabel',XTickLbl,'YTickLabel',YTickLbl)
axis xy

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
    basis_props.max_order = pa*[1 1];
    basis_props.basis_indices = RegModel.Parameters.t_statistic >= thr(i);
    basis_props.type = 'hh';
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
imagesc(1:pa,1:pa,reshape(basis_props.basis_indices,pa,pa))
set(gca,'XTickLabel',XTickLbl,'YTickLabel',YTickLbl)
axis xy
axis square

%%
close all
clc

RegModel = NonLinRegression(X,Y,basis_props,'qr');
[Y_hat,se_Y] = NL_predict(X_ast,RegModel);
[y_hat,~] = NL_predict(x,RegModel);

clr = lines(2);

figure
subplot(121)
surf(x_ast,x_ast,reshape(Y_hat,100,100)','LineStyle','none','FaceAlpha',0.75)
hold on
plot3(X(:,1),X(:,2),Y,'.','Color',clr(2,:))
zlim([-0.4 1.2])
set(gca,'CLim',[-0.25 1.2])

subplot(122)
imagesc(x_ast,x_ast,reshape(log10(se_Y),100,100)')
hold on
plot(X(:,1),X(:,2),'.','Color',clr(2,:))
colorbar

figure
plot(y,y_hat,'.')
grid on