function f = basis(x,p,type,basis_indices)

if nargin < 4
    ba = 1:p;
else
    ba = basis_indices;
end
switch type
    case 'h'    % Hermite polynomials
        f = hermite_old(x,p);
    case 'f'    % Fourier basis
        f = fourier(x,p);
    case 'l'    % Legendre polynomial basis
        f = legendre(x,p);
end

f = f(:,ba);
