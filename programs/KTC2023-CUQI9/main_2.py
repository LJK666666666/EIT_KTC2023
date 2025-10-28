#%%
import argparse
import numpy as np
import scipy as sp
import KTCFwd
import KTCMeshing
import KTCPlotting
import KTCScoring
import KTCAux
import matplotlib.pyplot as plt
import glob
from skimage.segmentation import chan_vese
from EITLib import NL_main
from EITLib.KTCRegularization_NLOpt import SMPrior
from EITLib.segmentation import scoring_function, otsu
import os

#%%
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("inputFolder")
    parser.add_argument("outputFolder")
    parser.add_argument("categoryNbr", type=int)
    parser.add_argument("niter", type=int)

    parser.add_argument("TV_factor", type=float)
    parser.add_argument("Tikhonov_factor", type=float)
    parser.add_argument("CUQI1_factor", type=float)
    parser.add_argument("segmentation_method", type=str)
 
    args = parser.parse_args()

    inputFolder = args.inputFolder
    outputFolder = args.outputFolder
    categoryNbr = args.categoryNbr

    niter = args.niter 

    TV_factor = args.TV_factor
    Tikhonov_factor = args.Tikhonov_factor
    CUQI1_factor = args.CUQI1_factor

    segmentation_method = args.segmentation_method

    # if output folder does not exist, create it
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    Nel = 32  # number of electrodes
    z = (1e-6) * np.ones((Nel, 1))  # contact impedances
    mat_dict = sp.io.loadmat(inputFolder + '/ref.mat') #load the reference data
    Injref = mat_dict["Injref"] #current injections
    Uelref = mat_dict["Uelref"] #measured voltages from water chamber
    Mpat = mat_dict["Mpat"] #voltage measurement pattern
    vincl = np.ones(((Nel - 1),76), dtype=bool) #which measurements to include in the inversion
    rmind = np.arange(0,2 * (categoryNbr - 1),1) #electrodes whose data is removed

    #remove measurements according to the difficulty level
    for ii in range(0,75):
        for jj in rmind:
            if Injref[jj,ii]:
                vincl[:,ii] = 0
            vincl[jj,:] = 0

    vincl = vincl.T.flatten()
    #recon = NL_main(Uelref, Uelref, Mpat, categoryNbr)

    # Get a list of .mat files in the input folder
    mat_files = sorted(glob.glob(inputFolder + '/data*.mat'))
    for objectno in range (0,len(mat_files)): #compute the reconstruction for each input file
        mat_dict2 = sp.io.loadmat(mat_files[objectno])
        Inj = mat_dict2["Inj"]
        Uel = mat_dict2["Uel"]
        Mpat = mat_dict2["Mpat"]
        deltaU = Uel - Uelref
        #############################  Changed code

        deltareco_pixgrid = NL_main.NL_main_2(Uel, Uelref, Inj, categoryNbr, niter=niter, output_dir_name=outputFolder, TV_factor=TV_factor, Tikhonov_factor=Tikhonov_factor, CUQI1_factor=CUQI1_factor)

        # save deltareco_pixgrid
        np.savez(outputFolder + '/' + str(objectno + 1) + '.npz', deltareco_pixgrid=deltareco_pixgrid) 
        
        if segmentation_method == 'otsu':
            deltareco_pixgrid_segmented = otsu(deltareco_pixgrid)
        elif segmentation_method == 'cv':
            deltareco_pixgrid_segmented = KTCScoring.cv_NLOpt(deltareco_pixgrid, log_par=1.5, linear_par=1, exp_par=0)
        else:
            # 明确抛出错误，而不是让未赋值的局部变量触发 UnboundLocalError。
            raise ValueError(f"Unknown segmentation_method '{segmentation_method}'. Supported: 'otsu' or 'cv'.")

        ###################################  End of changed code
        reconstruction = deltareco_pixgrid_segmented
        mdic = {"reconstruction": reconstruction}
        print(outputFolder + '/' + str(objectno + 1) + '.mat')
        sp.io.savemat( outputFolder + '/' + str(objectno + 1) + '.mat',mdic)

        # save reconstruction as png
        plt.imshow(reconstruction)
        # read real phantom from file
        phantom = sp.io.loadmat('GroundTruths/true'+str(objectno+1)+'.mat')['truth']
        # add title that shows the category number and score
        plt.title('Category ' + str(categoryNbr) + ', score = ' + str(scoring_function(reconstruction, phantom)))
        plt.savefig(outputFolder + '/' + str(objectno + 1) + '.png')

if __name__ == "__main__":
    main()
