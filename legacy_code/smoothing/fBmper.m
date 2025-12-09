%
% fBmper.m
%
% fractional (pseudo-)Brownian motion generator
%
%
% References:
%
%   [1]     M. Unser and T. Blu, `Self-similarity: Part I -- Splines and
%           operators', IEEE Trans. Sig. Proc. (in print).
%
%   [2]     T. Blu and M. Unser, `Self-similarity: Part II -- Optimal
%           estimation of fractal processes', IEEE Trans. Sig. Proc.,
%           in press.
%
%   [3]     M. Unser, T. Blu, "Fractional Splines and Wavelets," SIAM
%           Review, vol. 42, no. 1, pp. 43-67, March 2000.
%
%
% Author:   Pouya Dehghani Tafti <p.d.tafti@ieee.org>, based on original
%           code by Dr Thierry Blu.
%
%           Biomedical Imaging Group (BIG)
%           Ecole Polytechnique Federale de Lausanne
%           Switzerland
%
% This software can be downloaded at <http://bigwww.epfl.ch/>.
%
% $ version 1.1 $ 28.08.2006 $



function [t,y]=fBmper(epsH,H,m,N)

Y=fft(randn(1,m*N));
Y=Y(2:end);

omega=(1:(m*N-1))*2*pi/(m*N);

Y=m^(-H)*epsH*Y./abs(2*sin(omega/2)).^(H+0.5).*sqrt(fractsplineautocorr(H-0.5,omega/2/pi));

Y=[-real(sum(Y)) Y];

y = real(ifft(Y));
t = 0:(1/m):N-1/m;
