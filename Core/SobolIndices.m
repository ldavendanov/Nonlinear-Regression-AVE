function [S1,ST] = SobolIndices( W, basis_opt )

% Extracting information from the input
[n,p] = size(W);                                                            % Number of outputs and functional basis order
m = numel(basis_opt.order);                                                 % Number of inputs

%-- Constructing the multi-index vector
alpha = zeros(p,m);
l = cell(m,1);
for i=1:m
    l{i} = ones(1,basis_opt.order(i));
end

for i=1:m
    d = l;
    d{i} = 0:basis_opt.order(i)-1;
    a = d{1};
    for j=2:m
        a = kron( d{j}, a );
    end
    
    alpha(:,m-i+1) = a;
    
end

%-- Calculating the normalization factors of each basis
l = cell(m,1);
for i=1:m
    switch basis_opt.type(i)
        case 'l'
            l{i} = 2./( 2*(0:basis_opt.order(i)-1) + 1 );
        case 'h'
            l{i} = sqrt(2*pi)*factorial( 0:basis_opt.order(i)-1 );
    end
end
L = l{1};
for i=1:m-1
    L = kron(l{i+1},L);
end

%-- Calculating the Sobol indices (single and total effect indices)
if n==1
    
    ST = zeros(1,m);
    S1 = zeros(1,m);
    for i=1:m
        dim_indx = false(1,m);
        dim_indx(i) = true;
        ind = alpha(:,dim_indx) > 0;
        ind2 = all( alpha(:,~dim_indx) == 0, 2 );
        ST(i) = sum( W(ind).^2.*L(ind) ) / sum( W(2:end).^2.*L(2:end) );
        S1(i) = sum( W(ind&ind2).^2.*L(ind&ind2) ) / sum( W(2:end).^2.*L(2:end) );
    end
    
else
    
    ST = zeros(1,m);
    S1 = zeros(1,m);
    for i=1:m
        dim_indx = false(1,m);
        dim_indx(i) = true;
        ind = alpha(:,dim_indx) > 0;
        ind2 = all( alpha(:,~dim_indx) == 0, 2 );
        ST(i) = sum( dot( W(:,ind), W(:,ind) ).*L(ind) ) / sum( dot( W(:,2:end), W(:,2:end) ) .*L(2:end) );
        S1(i) = sum( dot( W(:,ind&ind2), W(:,ind&ind2) ).*L(ind&ind2) ) /...
            sum( dot( W(:,2:end), W(:,2:end) ).*L(2:end) );
    end
    
end