import KTCScoring
from skimage.segmentation import chan_vese
from scipy import ndimage
import numpy as np
from skimage.metrics import structural_similarity as ssim


def cv(deltareco_pixgrid, mu=0.1, lambda1=1, lambda2=1, init_level_set="checkerboard"):
    mu = np.mean(deltareco_pixgrid)
    # Feel free to play around with the parameters to see how they impact the result
    cv = chan_vese(abs(deltareco_pixgrid), mu=mu, lambda1=lambda1, lambda2=lambda2, tol=1e-6,
                max_num_iter=1000, dt=2.5, init_level_set=init_level_set,
                extended_output=True)

    labeled_array, num_features = ndimage.label(cv[0])
    # Initialize a list to store masks for each region
    region_masks = []

    # Loop through each labeled region
    deltareco_pixgrid_segmented = np.zeros((256,256))

    for label in range(1, num_features + 1):
        # Create a mask for the current region
        region_mask = labeled_array == label
        region_masks.append(region_mask)
        if np.mean(deltareco_pixgrid[region_mask]) < mu:
            deltareco_pixgrid_segmented[region_mask] = 1
        else:
            deltareco_pixgrid_segmented[region_mask] = 2

    return deltareco_pixgrid_segmented


def otsu(deltareco_pixgrid):
    level, x = KTCScoring.Otsu2(deltareco_pixgrid.flatten(), 256, 7)
    deltareco_pixgrid_segmented = np.zeros_like(deltareco_pixgrid)

    ind0 = deltareco_pixgrid < x[level[0]]
    ind1 = np.logical_and(deltareco_pixgrid >= x[level[0]],deltareco_pixgrid <= x[level[1]])
    ind2 = deltareco_pixgrid > x[level[1]]
    inds = [np.count_nonzero(ind0),np.count_nonzero(ind1),np.count_nonzero(ind2)]
    bgclass = inds.index(max(inds)) #background class


    if bgclass == 0:
            deltareco_pixgrid_segmented[ind1] = 2
            deltareco_pixgrid_segmented[ind2] = 2
    if bgclass ==  1:
            deltareco_pixgrid_segmented[ind0] = 1
            deltareco_pixgrid_segmented[ind2] = 2
    if bgclass == 2:
            deltareco_pixgrid_segmented[ind0] = 1
            deltareco_pixgrid_segmented[ind1] = 1

    return deltareco_pixgrid_segmented.reshape((256,256))


def scoring_function(truth, recon):
    
    truth_c = np.zeros((256,256))
    truth_c[truth == 2] = 1
    recon_c = np.zeros((256,256))
    recon_c[recon == 2] = 1
    s_c = ssim(truth_c, recon_c, data_range = 2.0, gaussian_weights = True,
          K1 = 1*1e-4, K2 = 9*1e-4, sigma = 80.0, win_size = 255)
    
    truth_r = np.zeros((256,256))
    truth_r[truth == 1] = 1
    recon_r = np.zeros((256,256))
    recon_r[recon == 1] = 1
    s_r = ssim(truth_r, recon_r, data_range = 2.0, gaussian_weights = True,
          K1 = 1*1e-4, K2 = 9*1e-4, sigma = 80.0, win_size = 255)
    return 0.5*(s_c+s_r)
    