function main(inputFolder,outputFolder,categoryNbr)
path(path,'MiscCodes/')

PAUSE_FOR_EACH_TARGET = 1;%set if you want to pause after reconstructing each target. if 0, do not plot anything
Nel = 32; %number of electrodes
z = 1e-6*ones(Nel,1); %contact impedances
load([inputFolder '/ref.mat']) %reference data from water chamber

%undersampling of the data for the difficulty levels
vincl = true(31,76); %which measurements are used in the inversion - 31 voltage values for each of the 76 current injections
rmind = 1:2*(categoryNbr - 1);
for ii=1:76
    if any(Injref(rmind,ii) ~= 0)
        vincl(:,ii) = false; %remove all voltage measurements for this current injection
    end
    vincl(rmind,ii) = false; %remove all voltage measurements collected using these electrodes
end
vincl = vincl(:);

%this can be used to create more meshes using Gmsh: (provided Gmsh is
%installed)
%[Mesh2,Mesh,elcenterangles] = create2Dmesh_circ(Nel,5,1,4); 

load('Mesh_sparse.mat')
%disp(['Nodes in 1st order mesh: ' num2str(length(Mesh.g))]);

sigma0 = ones(length(Mesh.g),1); %linearization point

%prior
corrlength = 0.115;
var_sigma = 0.05^2;
mean_sigma = sigma0;
smprior = SMprior(Mesh.g,corrlength,var_sigma,mean_sigma);

%set up the forward solver for inversion
solver = EITFEM(Mesh, Mesh2, Injref, Mpat, vincl, []);

%set up the noise model for inversion
noise_std1 = 0.05; %standard deviation for first noise component (relative to each voltage measurement)
noise_std2 = 0.01; %standard deviation for second noise component (relative to the largest voltage measurement)
solver.SetInvGamma(noise_std1,noise_std2,Uelref)

files=dir(fullfile(inputFolder,'*.mat')); %one of these is a reference data, the others are from targets with inclusions

for objectno = 1:length(files)-1
    close all
    load([inputFolder '/data' num2str(objectno) '.mat'])
    deltaU = Uel - Uelref; %the difference data

    if PAUSE_FOR_EACH_TARGET
        Usim = solver.SolveForward(sigma0,z);
        figure, plot(Uelref(vincl)), hold on, plot(Usim)
        set(gcf,'Units','normalized','OuterPosition',[0.0 0.2 0.3 0.4])
    end

    %compute linear difference reconstruction for the conductivity change
    J = solver.Jacobian(sigma0,z);
    %Jz = solver.Jacobianz(sigma0,z); %contact impedance jacobian - not used by the simple reconstruction algorithm
    deltareco = (J'*solver.InvGamma_n(vincl,vincl)*J + smprior.L'*smprior.L)\J'*solver.InvGamma_n(vincl,vincl)*deltaU(vincl);

    if PAUSE_FOR_EACH_TARGET
        %plot the reconstruction
        sgplot = sigmaplotter(Mesh,5,'parula');
        sgplot.basic2Dplot(deltareco,{['linear difference reconstruction ' num2str(objectno)]});
    end

    %interpolate the reconstruction into a pixel image
    deltareco_pixgrid = interpolateRecoToPixGrid(deltareco,Mesh);
    if PAUSE_FOR_EACH_TARGET
        figure(6), imagesc(deltareco_pixgrid), colorbar, axis image
        set(gcf,'Units','normalized','OuterPosition',[0.3 0.6 0.3 0.4])
    end

    %treshold the image histogram using Otsu's method
    [level,x] = Otsu2(deltareco_pixgrid(:),256,7);

    reconstruction = zeros(size(deltareco_pixgrid));

    ind1 = find(deltareco_pixgrid < x(level(1)));
    ind2 = find(deltareco_pixgrid >= x(level(1)) & deltareco_pixgrid <= x(level(2)));
    ind3 = find(deltareco_pixgrid > x(level(2)));

    %Check which class is the background (assumed to be the one with the
    %most pixels in it).
    [bgnum,bgclass] = max([length(ind1) length(ind2) length(ind3)]);
    switch bgclass
        case 1 %background is the class with lowest values - assign other two classes as conductive inclusions
            reconstruction([ind2]) = 2;
            reconstruction([ind3]) = 2;
        case 2 %background is the middle class - assign the lower class as resistive inclusions and the higher class as conductive
            reconstruction([ind1]) = 1;
            reconstruction([ind3]) = 2;
        case 3 %background is the class with highest values - assign the other two classes as resistive inclusions
            reconstruction([ind1]) = 1;
            reconstruction([ind2]) = 1;
    end

    if PAUSE_FOR_EACH_TARGET
        %plot the segmented reconstruction in the pixel grid
        figure(9), imagesc(reconstruction)
        colormap gray
        colorbar, axis image
        title('segmented linear difference reconstruction')
        set(gcf,'Units','normalized','OuterPosition',[0.6 0.2 0.3 0.4])
    end

    save([outputFolder '/' num2str(objectno) '.mat'],'reconstruction')

    if PAUSE_FOR_EACH_TARGET
        pause();
    end
end

end