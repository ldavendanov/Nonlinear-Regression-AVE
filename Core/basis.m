function f = basis(x,order,type)

switch type
    case 'h'    % Hermite polynomials
        f = hermite_old(x,order);
    case 'f'    % Fourier basis
        f = fourier(x,order);
    case 'l'    % Legendre polynomial basis
        f = legendre(x,order);
end