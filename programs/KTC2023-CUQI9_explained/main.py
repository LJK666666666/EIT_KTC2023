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
import os

#%%
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("inputFolder")
    parser.add_argument("outputFolder")
    parser.add_argument("categoryNbr", type=int)
    args = parser.parse_args()

    inputFolder = args.inputFolder
    outputFolder = args.outputFolder
    categoryNbr = args.categoryNbr

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
        #500000.0_0.5_10000000000.0_ 

        deltareco_pixgrid = NL_main.NL_main(Uel, Uelref, Inj, categoryNbr, niter=170, output_dir_name=outputFolder)

        # save deltareco_pixgrid
        np.savez(outputFolder + '/' + str(objectno + 1) + '.npz', deltareco_pixgrid=deltareco_pixgrid) 
        
        deltareco_pixgrid_segmented = KTCScoring.cv_NLOpt(deltareco_pixgrid, log_par=1.5, linear_par=1, exp_par=0)

        ###################################  End of changed code
        reconstruction = deltareco_pixgrid_segmented
        mdic = {"reconstruction": reconstruction}
        print(outputFolder + '/' + str(objectno + 1) + '.mat')
        sp.io.savemat( outputFolder + '/' + str(objectno + 1) + '.mat',mdic)

if __name__ == "__main__":
    main()
