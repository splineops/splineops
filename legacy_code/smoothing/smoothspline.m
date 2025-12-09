%
% smoothspline.m
%
% fractional smoothing spline
%
% Returns samples of the smoothing spline for a given input sequence,
% sampled at "m" times the rate of input.  The input is assumed to be
% sampled at integers 0..N-1.
%
%
% Usage:
%
%   [t,ys] = smoothspline(y,lambda,m,gamma)
%
% Parameters:
%   y       input signal
%   lambda  lambda parameter for the smoothing spline
%   m       upsampling factor
%   gamma   order of the spline operator.  order of the spline itself will
%           be twice this value.
%
% Return values:
%   ys      smoothing spline sequence
%   t       sampling points
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
% Author:   Pouya Dehghani Tafti <p.d.tafti@ieee.org>, partially based on
%           code by Dr Thierry Blu.
%
%           Biomedical Imaging Group (BIG)
%           Ecole Polytechnique Federale de Lausanne
%           Switzerland
%
% This software can be downloaded at <http://bigwww.epfl.ch/>.
%
% $ version 1.1 $ 29.08.2006 $



function [t,ys] = smoothspline(y,lambda,m,gamma)


%% make sure that y is a row vector
[dim1 dim2] = size(y);
if dim1 ~= 1
    if dim2 ~= 1
        error('smoothspline:wrongdims','incorrect dimensions');
    end
    y = y';
end
N = length(y);


% find DFT
Y = fft(y);
omega = (1:(N*m-1)) * 2*pi/(N*m);


%% upsample Y
Ym = periodize(Y,m);


%% form internal tables used in several calcs
sinm2g = abs(2 * sin(m*omega/2)).^(2*gamma);
sin2g  = abs(2 * sin(  omega/2)).^(2*gamma);


% %% calculate A_gamma(omega) using Eq. (29) of Part II
% NUMIT = 10; % # terms used = 2*NUMIT + 1
% Ag = sin2g / (2*pi*NUMIT)^(2*gamma) .* (2*NUMIT/(2*gamma-1) - 1 + gamma/(3*NUMIT) - gamma^2/(2*pi^2*NUMIT^2) + omega.^2 * gamma/(2*pi^2*NUMIT^2));
% for ii=-NUMIT:NUMIT
%     Ag = Ag + sin2g ./ (omega + 2*pi*ii).^(2*gamma);
% end


%% calculate A_gamma(omega)
alpha = gamma - 1;
Ag = fractsplineautocorr(alpha,[0 omega]/2/pi);


%% calculate A_gamma(m omega)
% Agm = fractsplineautocorr(alpha,m*[0 omega]/2/pi);
Agm = periodize(downsample(Ag,m),m);


Ag = Ag(2:end);
Agm = Agm(2:end);


%% calculate H_m, the smoothing spline filter
Hm = m^(-2*gamma+1) * (sinm2g ./ sin2g) .* Ag ./ (Agm + lambda .* sinm2g);
Hm = [m Hm];


%% generate outputs
ys = real(ifft(Hm .* Ym));
t = 0:(1/m):N-1/m;



%%%%%%%%%%

function xp = periodize(x,m)
% periodizes the input "x" by concatenating "m" copies os it.
xp = [];
for ii=1:m
    xp = [xp x];
end 
