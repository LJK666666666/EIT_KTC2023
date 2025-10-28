%This script runs simulations to create synthetic data to use for EIT
%reconstructions.

clear, close all
clc

path(path,'MiscCodes/')

%number of distinct types of segments (2 = background, one type of inclusion)
%(3 = background, two types of inclusions)
segments = 3; %this affects only that are we converting into a binary or a "trinary" image in the end

%set up data simulation mesh
Nel = 32; %number of electrodes
load('Mesh_dense.mat')
Meshsim = Mesh;
Mesh2sim = Mesh2;

%measurement pattern
z = (1)*ones(Nel,1); %contact impedances
[Inj,Mpat,vincl] = setMeasurementPattern(Nel); %current injection pattern and voltage measurement pattern

%simulated initial conductivity and the change
[sigma,delta_sigma,sigma2] = simulateConductivity(Meshsim,segments);
sgplot = sigmaplotter(Meshsim,[2,3],'parula');
sgplot.basic2Dplot([sigma; delta_sigma],{'initial conductivity','conductivity change'});

%set up forward solver
solver = EITFEM(Meshsim, Mesh2sim, Inj, Mpat, vincl, []);

%simulate data
Iel2_true = solver.SolveForward([sigma],z);
Iel_true = solver.SolveForward([sigma + delta_sigma],z);

%add some noise
noise_std1 = 0.1; %standard deviation of the noise as percentage of each voltage measurement
noise_std2 = 0.001; %standard deviation of 2nd noise component (this is proportional to the largest measured value)
solver.SetInvGamma(noise_std1,noise_std2,Iel2_true) %compute the noise precision matrix
Iel2_noisy = Iel2_true + solver.Ln\randn(size(Iel2_true));
Iel_noisy = Iel_true +  solver.Ln\randn(size(Iel_true));
deltaI = Iel_noisy - Iel2_noisy;

%create inversion mesh
load('Mesh_sparse.mat')
disp(['Nodes in simulation 1st order mesh: ' num2str(length(Meshsim.g))]);
disp(['Nodes in inversion 1st order mesh: ' num2str(length(Mesh.g))]);

%set up the Gaussian smoothness prior for conductivity change
sigma0 = ones(length(Mesh.g),1);
corrlength = 1*0.115;
var_sigma = 0.05^2;
mean_sigma = sigma0;
smprior = SMprior(Mesh.g,corrlength,var_sigma,mean_sigma);

%set up the forward solver for inversion
solver = EITFEM(Mesh, Mesh2, Inj, Mpat, vincl, []);
solver.SetInvGamma(noise_std1,noise_std2,deltaI)

%compute linear difference reconstruction for the conductivity change
J = solver.Jacobian(sigma0,z);
deltareco = (J'*solver.InvGamma_n*J + smprior.L'*smprior.L)\J'*solver.InvGamma_n*deltaI;
sgplot = sigmaplotter(Mesh,[5],'parula');
sgplot.basic2Dplot([deltareco],{'linear difference reconstruction'});

%interpolate the reconstruction into a pixel image
deltareco_pixgrid = interpolateRecoToPixGrid(deltareco,Mesh);
figure(6), imagesc(deltareco_pixgrid), colorbar, axis image
set(gcf,'Units','normalized','OuterPosition',[0.3 0.6 0.3 0.4])

%treshold the image histogram using Otsu's method
switch segments
    case 2
        [level,x] = Otsu(deltareco_pixgrid(:),256,7);
    case 3
        [level,x] = Otsu2(deltareco_pixgrid(:),256,7);
end

deltareco_pixgrid_segmented = zeros(size(deltareco_pixgrid));
switch segments
    case 2
        ind = find(deltareco_pixgrid < x(level(1)));
        deltareco_pixgrid_segmented(ind) = 1;
    case 3
        ind = find(deltareco_pixgrid < x(level(1)));
        deltareco_pixgrid_segmented(ind) = 1;
        ind = find(deltareco_pixgrid > x(level(2)));
        deltareco_pixgrid_segmented(ind) = 2;
end


figure(8), imagesc(deltareco_pixgrid_segmented), colormap gray, colorbar, axis image
title('segmented linear difference reconstruction')
set(gcf,'Units','normalized','OuterPosition',[0.6 0.2 0.3 0.4])

figure, imagesc(deltareco_pixgrid), colormap gray, colorbar, axis image

%create ground truth segmented image of the simulated conductivity change
delta_pixgrid = interpolateRecoToPixGrid(delta_sigma,Meshsim);

switch segments
    case 2
        [level,x] = Otsu(delta_pixgrid,256,9);
    case 3
        [level,x] = Otsu2(delta_pixgrid,256,9);
end

delta_pixgrid_segmented = zeros(size(deltareco_pixgrid));
switch segments
    case 2
        ind = find(delta_pixgrid < x(level(1)));
        delta_pixgrid_segmented(ind) = 1;
    case 3
        ind = find(delta_pixgrid < x(level(1)));
        delta_pixgrid_segmented(ind) = 1;
        ind = find(delta_pixgrid > x(level(2)));
        delta_pixgrid_segmented(ind) = 2;
end

figure, imagesc(delta_pixgrid_segmented), colormap gray, colorbar, axis image
title('ground truth image')
set(gcf,'Units','normalized','OuterPosition',[0.6 0.6 0.3 0.4])

score = scoringFunction(delta_pixgrid_segmented,deltareco_pixgrid_segmented);
disp(['SCORE = ' num2str(score)])


