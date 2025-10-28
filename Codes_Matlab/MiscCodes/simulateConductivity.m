function [sigma,delta_sigma,sigma2] = simulateConductivity(Meshsim,inclusiontypes)
sigma = ones(size(Meshsim.g,1),1);
contrast = -0.5;
cp = [0.5,0.5]';
r = 0.2;
ind = find( vecnorm( Meshsim.g' - cp) <= r );
delta_sigma = zeros(size(sigma));
delta_sigma(ind) = contrast;
cp = [0,0]';
r = 0.2;
ind = find( vecnorm( Meshsim.g' - cp) <= r );
if inclusiontypes == 2
    delta_sigma(ind) = contrast;
else
    delta_sigma(ind) = abs(contrast);
end
sigma2 = sigma + delta_sigma;
end