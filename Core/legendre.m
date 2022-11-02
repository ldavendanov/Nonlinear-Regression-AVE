function P = legendre(x,pb)

x = x(:);
N = length(x);

P = ones(N,pb);
if pb >= 2
    P(:,2) = x;
end

% Calculting the Hermite polynomials
for i=3:pb
    n = i-1;
    P(:,i) = ( (2*n+1)*x.*P(:,i-1) - n*P(:,i-2) ) / ( n+1 );
end
