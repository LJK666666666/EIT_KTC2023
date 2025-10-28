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
from pathlib import Path
import csv

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

    # Prepare ground truth directory based on level number
    repo_root = Path(__file__).resolve().parents[2]
    gt_dir = repo_root / 'EvaluationData' / 'GroundTruths' / f'level_{categoryNbr}'
    # Prepare CSV for scoring results
    csv_path = Path(outputFolder) / 'scoring_results.csv'
    if not csv_path.exists():
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'file', 'level', 'score'])
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

        # Save images (no titles) for both raw reconstruction and segmented result
        # Raw reconstruction image
        fig1, ax1 = plt.subplots()
        im1 = ax1.imshow(deltareco_pixgrid, cmap='viridis')
        ax1.axis('image')
        ax1.set_xticks([])
        ax1.set_yticks([])
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        fig1.savefig(outputFolder + '/' + str(objectno + 1) + '_reco.png', bbox_inches='tight', dpi=200)
        plt.close(fig1)

        # Segmented reconstruction image
        fig2, ax2 = plt.subplots()
        im2 = ax2.imshow(deltareco_pixgrid_segmented, cmap='gray')
        ax2.axis('image')
        ax2.set_xticks([])
        ax2.set_yticks([])
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        fig2.savefig(outputFolder + '/' + str(objectno + 1) + '_seg.png', bbox_inches='tight', dpi=200)
        plt.close(fig2)

        # Load ground truth for comparison and scoring
        gt_mat = gt_dir / f"{objectno + 1}_true.mat"
        gt_png = gt_dir / f"{objectno + 1}_true.png"
        groundtruth = None
        if gt_mat.exists():
            try:
                gt_data = sp.io.loadmat(str(gt_mat))
                for key in ['truth', 'groundtruth', 'gt', 'image']:
                    if key in gt_data:
                        groundtruth = np.squeeze(gt_data[key])
                        break
            except Exception:
                groundtruth = None
        if groundtruth is None and gt_png.exists():
            try:
                gt_img = plt.imread(str(gt_png))
                # If RGB/RGBA, convert to grayscale by taking first channel
                if gt_img.ndim == 3:
                    gt_img = gt_img[..., 0]
                groundtruth = gt_img
            except Exception:
                groundtruth = None

        # Compose comparison figures if groundtruth is available
        if groundtruth is not None:
            # Ensure groundtruth is 2D
            groundtruth = np.squeeze(groundtruth)
            # Raw vs Ground Truth
            figc1, axes = plt.subplots(1, 2, figsize=(8, 4))
            im_l = axes[0].imshow(groundtruth, cmap='gray')
            axes[0].axis('image')
            axes[0].set_xticks([]); axes[0].set_yticks([])
            plt.colorbar(im_l, ax=axes[0], fraction=0.046, pad=0.04)
            im_r = axes[1].imshow(deltareco_pixgrid, cmap='viridis')
            axes[1].axis('image')
            axes[1].set_xticks([]); axes[1].set_yticks([])
            plt.colorbar(im_r, ax=axes[1], fraction=0.046, pad=0.04)
            figc1.savefig(outputFolder + '/' + str(objectno + 1) + '_compare_reco.png', bbox_inches='tight', dpi=200)
            plt.close(figc1)

            # Segmented vs Ground Truth
            figc2, axes2 = plt.subplots(1, 2, figsize=(8, 4))
            im_l2 = axes2[0].imshow(groundtruth, cmap='gray')
            axes2[0].axis('image')
            axes2[0].set_xticks([]); axes2[0].set_yticks([])
            plt.colorbar(im_l2, ax=axes2[0], fraction=0.046, pad=0.04)
            im_r2 = axes2[1].imshow(deltareco_pixgrid_segmented, cmap='gray')
            axes2[1].axis('image')
            axes2[1].set_xticks([]); axes2[1].set_yticks([])
            plt.colorbar(im_r2, ax=axes2[1], fraction=0.046, pad=0.04)
            figc2.savefig(outputFolder + '/' + str(objectno + 1) + '_compare_seg.png', bbox_inches='tight', dpi=200)
            plt.close(figc2)

            # Compute and save score using scoringFunction
            try:
                score = KTCScoring.scoringFunction(groundtruth, deltareco_pixgrid_segmented)
                with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([objectno + 1, f"{objectno + 1}", categoryNbr, float(score)])
            except Exception:
                pass

    # Save images (no titles) for both raw reconstruction and segmented result
    # Raw reconstruction image
    fig1, ax1 = plt.subplots()
    im1 = ax1.imshow(deltareco_pixgrid, cmap='viridis')
    ax1.axis('image')
    ax1.set_xticks([])
    ax1.set_yticks([])
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    fig1.savefig(outputFolder + '/' + str(objectno + 1) + '_reco.png', bbox_inches='tight', dpi=200)
    plt.close(fig1)

    # Segmented reconstruction image
    fig2, ax2 = plt.subplots()
    im2 = ax2.imshow(deltareco_pixgrid_segmented, cmap='gray')
    ax2.axis('image')
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    fig2.savefig(outputFolder + '/' + str(objectno + 1) + '_seg.png', bbox_inches='tight', dpi=200)
    plt.close(fig2)

if __name__ == "__main__":
    main()
