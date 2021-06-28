function [f,y] = sinc2D( x )

N = size(x,2);

omega = 2*pi;
sigmaW2 = 1e-3;
L = [1.5 0; 0 1];

r = diag(x'*L*x)';
f = sin( omega*r )./(omega*r) + 0.1*atan( omega*r );
y = f + sqrt(sigmaW2)*randn(1,N);