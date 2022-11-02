function H = hermite_old(x,pb)

x = x(:);
N = length(x);

H = ones(N,pb);
if pb >= 2
    H(:,2) = 2*x;
end

% Calculting the Hermite polynomials
for i=3:pb
    n = i-1;
    H(:,i) = 2*x.*H(:,i-1) - 2*(n-1)*H(:,i-2);
end