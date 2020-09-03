function f = tensorbasis(x,basis_opt)
% Computes a multivariate tensor product basis made up from univariate
% basis determined by the options in 'basis_opt'

[n,N] = size(x);
m = prod(basis_opt.order);
f = zeros(m,N);

% Calculating the univariate basis
f0 = cell(n,1);
for i=1:n
    f0{i} = basis(x(i,:),basis_opt.order(i),basis_opt.type(i));
end

% Calculating the tensor basis
for j=1:N
    f_ = kron(f0{1}(:,j),f0{2}(:,j));
    for i=3:n
        f_ = kron(f_,f0{i}(:,j));
    end
    f(:,j) = f_;
end
