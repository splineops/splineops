%
% fsdemo.m
%
% fBm estimation demo
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
% Author:   Pouya Dehghani Tafti <p.d.tafti@ieee.org>
%
%           Biomedical Imaging Group (BIG)
%           Ecole Polytechnique Federale de Lausanne
%           Switzerland
%
% This software can be downloaded at <http://bigwww.epfl.ch/>.
%
% $ version 1.1 $ 29.08.2006 $



clear;
close all;
clc;


%% define programme constants
m = 4;      % upsampling factor
N = 256;    % number of samples


%% print welcome messsage
disp('fsdemo.m');
disp('--------');
disp('Optimal estimation of fractional Brownian motion (fBm)');
disp('using fractional splines.                       ');
disp(sprintf('\n'));
disp('Biomedical Imaging Group, EPFL, 2006.');
disp(sprintf('\n'));

disp('      In the first part of this demo, a realization of an fBm process ');
disp(['      of length ' num2str(N) ' is generated and corrupted with noise of given ']);
disp('      strength.  The sequence is then denoised and oversampled by a   ');
disp(['      factor of ' num2str(m) ' using the optimal fractional spline estimator. ']);
disp(sprintf('\n'));


%% read parameters H, epsH, sigma
H = -1;
while (H <= 0) || (H >= 1)
    H       = input('Enter Hurst parameter [0<H<1] > ');
end
% since fBm is non-stationary, we ask for the SNR at a specific point in
% time.
SNRmeas    = input(['Enter measurement SNR at the mid-point (t = ' num2str(N/2) ') [dB] > ']);


%% create (pseudo-)fBm signal
epsH = 1;
[t0,y0] = fBmper(epsH,H,m,N);
Ch = epsH^2 / (gamma(2*H+1) * sin(pi*H));
POWmid = Ch * (N/2)^(2*H); % theoretical fBm variance at the midpoint


%% measurement: downsample and add noise
t = downsample(t0,m);
y = downsample(y0,m);
sigma = (sqrt(POWmid) / 10^(SNRmeas/20));
noise = randn(1,N);
noise = sigma * noise / sqrt(mean(noise.^2));
y = y + noise;


%% find smoothing spline fit
lambda = (sigma / epsH)^2;
gamma_ = H + .5;
[ts,ys] = smoothspline(y,lambda,m,gamma_);
 

%% add non-stationary correction term
cnn = [1 zeros(1,N-1)]; % normalized white noise autocorrelation
[tes,r] = smoothspline(cnn,lambda,m,gamma_);
r = r * ys(1) / r(1);
yest = ys - r;

    
%% calc MSE and SNR and print
% since fBm is non-stationary, we calculate the SNR at a specific point in
% time (the mid-point).
MSE0 = mean(noise.^2);                      % measurement MSE
MSE  = mean(downsample((yest-y0).^2,m));    % denoised sequence MSE
MSEm = mean((yest-y0).^2);                  % denoise and oversampled signal MSE

SNR0 = 10 * log10(POWmid / MSE0);
SNR  = 10 * log10(POWmid / MSE );
SNRm = 10 * log10(POWmid / MSEm);

disp(sprintf('\n'));
disp(['Number of measurements is ' num2str(N) ', oversampling factor is ' num2str(m) '.']);
disp(['mSNR (SNR at the mid-point) of the measured sequence      is ' num2str(SNR0)      ' dB.']);
disp(['mSNR improvement of the denoised sequence                 is ' num2str(SNR-SNR0)  ' dB.']);
disp(['mSNR improvement of the denoised and oversampled sequence is ' num2str(SNRm-SNR0) ' dB.']);


%% plot
figure;
plot(t0(1:end/2)  , y0(1:end/2)   , 'k'   , ...  % ground truth
     t(1:end/2)   , y(1:end/2)    , 'k+:'  , ... % noisy measurements
     ts(1:end/2)  , ys(1:end/2)   , 'r--' , ...  % uncorrected estimation
     tes(1:end/2) , yest(1:end/2) , 'r'   );     % estimation + correction
legend('fBm','noisy fBm samples','stationary estimation','non-stationary estimation');
title(['Estimation of fBm (H =' num2str(H) ', \epsilon_H^2 =', num2str(epsH), ', \sigma_N^2 =', num2str(sigma^2),')']);
xlabel('time');
ylabel('B_H');
axis tight;


%% verification
disp(sprintf('\n'));
disp('NOTE: To verify the optimality of the estimator we compare the MSE    ');
disp('      for estimators with different values of gamma and lambda.       ');
disp('      To avoid excessive computation, verification is performed using ');
disp('      only one realization of fBm.                                    ');
disp(sprintf('\n'));
vala = 0;
vala = input('Enter 1 to verify reuslts (may take some time) or 0 to end > ');
if (vala ~= 1)
    disp(sprintf('\n'));
    disp('Done.');
    return
end


%% check for optimality of lambda
Lambda = lambda * (0:.1:3);
MSEiii=0;
for ii=1:length(Lambda)
    [tsc,ysc] = smoothspline(y,Lambda(ii),m,gamma_);
    [trc,rc ] = smoothspline(cnn,Lambda(ii),m,gamma_);
    yestc = ysc - rc * ysc(1) / rc(1);
    MSEiii(ii) = mean(downsample((yestc-y0).^2,m));
end
figure;
plot(Lambda,MSEiii,'k',lambda,MSE,'r+');
legend('','theoretical optimum point');
title('MSE vs \lambda');
xlabel('\lambda');
ylabel('MSE');


%% check for optimality of gamma_
Gamma = 0.55:.01:1.45;
MSEii=0;
for ii=1:length(Gamma)
    g=Gamma(ii);
    [tsc,ysc] = smoothspline(y,lambda,m,g);
    [trc,rc ] = smoothspline(cnn,lambda,m,g);
    yestc = ysc - rc * ysc(1) / rc(1);
    MSEii(ii) = mean(downsample((yestc-y0).^2,m));
end
figure;
plot(Gamma,MSEii,'k',gamma_,MSE,'r+'); 
legend('','theoretical optimum point');
title('MSE vs \gamma');
xlabel('\gamma');
ylabel('MSE');


disp(sprintf('\n'));
disp('Done.');