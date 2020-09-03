function f = fourier(x,n)
% Calculates a Fourier basis with a constant plus 'n' pairs of
% sines/cosines of the variable 'x' normalized in the interval [-1 1]

N = length(x);
f = zeros(n,N);

f(1,:) = 1;
for i=1:(n-1)/2
    f(2*i,:) = sin(i*2*pi*x);
    f(2*i+1,:) = cos(i*2*pi*x);
end