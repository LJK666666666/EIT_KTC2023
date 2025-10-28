
import os 
from scipy.io import loadmat, savemat
import torch 
import argparse 
import numpy as np 
from scipy.stats import mode
from pathlib import Path 
import matplotlib.pyplot as plt 

from configs.default_config import get_default_configs
from src import get_standard_score, get_standard_sde, wrapper_ddim, BaseSampler, LinearisedRecoFenics
from src import ExponentialMovingAverage, FastScoringFunction

# regularisation parameters for initial reconstruction 
level_to_alphas = {
    1 : [[1956315.789, 0.,0.],[0., 656.842 , 0.],[0.,0.1,6.105],[1956315.789/3., 656.842/3,6.105/3.], [1e4, 0.1,5.]], 
    2 : [[1890000, 0.,0.],[0., 505.263, 0.],[0.,0.1,12.4210],[1890000/3., 505.263/3.,12.421/3.], [1e4, 0.1,5.]], 
    3 : [[1890000, 0.,0.],[0., 426.842, 0.],[0.,0.1,22.8421],[2143157/3., 426.842/3.,22.8421/3.], [6e5, 3,14]],
    4 : [[1890000, 0.,0.],[0., 1000., 0.],[0.,0.1,43.052],[1890000/3., 1000./3.,43.052/3.], [6e5, 8,16]], 
    5 : [[1890000, 0.,0.],[0., 843.6842, 0.],[0.,0.1,30.7368],[1890000/3., 843.684/3.,30.7368/3.], [6e5, 10,18]], 
    6 : [[40000, 0.,0.],[0., 895.789, 0.],[0.,0.1,74.947],[40000/3., 895.78/3.,74.947/3.], [6e5, 25,20]], 
    7 : [[40000, 0.,0.],[0., 682.105, 0.],[0.,0.1,18.421],[40000/3., 687.3684/3.,18.421/3.], [6e5, 30,22]], 
}

# hyperparameters for conditional sampling
# 01.11.2023
level_to_hparams = {
    1: {'eta':0.01, 'num_samples': 60, 'num_steps': 10, 'use_ema':True},
    2: {'eta':0.01, 'num_samples': 60, 'num_steps': 10, 'use_ema':False},
    3: {'eta':0.3, 'num_samples': 10, 'num_steps': 20, 'use_ema':False},
    4: {'eta':0.9, 'num_samples': 10, 'num_steps': 100, 'use_ema':False},
    5: {'eta':0.1, 'num_samples': 25, 'num_steps': 100, 'use_ema':True},
    6: {'eta':0.1, 'num_samples': 25, 'num_steps': 100, 'use_ema':True},
    7: {'eta':0.8, 'num_samples': 15, 'num_steps': 100, 'use_ema':False},
}

level_to_model_path = { 
    1: "diffusion_models/level_1/version_01/",  
    2: "diffusion_models/level_2/version_01/",
    3: "diffusion_models/level_3/version_01/",
    4: "diffusion_models/level_4/version_01/",
    5: "diffusion_models/level_5/version_01/",
    6: "diffusion_models/level_6/version_01/",
    7: "diffusion_models/level_7/version_01/",
}



parser = argparse.ArgumentParser(description='reconstruction using conditional diffusion')
parser.add_argument('level')

def coordinator(args):

    torch.manual_seed(42)

    level = int(args.level)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Level: ", args.level)

    save_path = f"examples/level_{level}/"
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    ### load conditional diffusion model 
    config = get_default_configs()
        
    sde = get_standard_sde(config=config)
    score = get_standard_score(config=config, sde=sde, use_ema=False, load_model=False)

    sampling_params = level_to_hparams[level]

    score.load_state_dict(torch.load(os.path.join(level_to_model_path[level],  "model_training.pt")))
    if sampling_params["use_ema"]:
        ema = ExponentialMovingAverage(score.parameters(), decay=0.999)
        ema.load_state_dict(torch.load(os.path.join(level_to_model_path[level], "ema_model_training.pt")))
        ema.copy_to(score.parameters())

    score.to(device)
    score.eval() 

    sampler = BaseSampler(
        score=score,
        sde=sde,
        predictor=wrapper_ddim, 
        sample_kwargs={
            'num_steps': sampling_params["num_steps"], 
            'batch_size': sampling_params["num_samples"],
            'im_shape': [1,256,256],
            'travel_length': 1, 
            'travel_repeat': 1, 
            'predictor': {'eta' : sampling_params["eta"]}
            },
        device=device)

    ### read files from args.input_folder 
    # there will be ref.mat in the input_folder, dont process this 

    y_ref = loadmat(f"ChallengeData/level_{level}/ref.mat")
    Injref = y_ref["Injref"]
    Mpat = y_ref["Mpat"]
    Uelref = y_ref["Uelref"]

    mesh_name = "sparse"
    B = Mpat.T

    Nel = 32
    vincl_level = np.ones(((Nel - 1),76), dtype=bool) 
    rmind = np.arange(0,2 * (level - 1),1) #electrodes whose data is removed

    #remove measurements according to the difficulty level
    for ii in range(0,75):
        for jj in rmind:
            if Injref[jj,ii]:
                vincl_level[:,ii] = 0
            vincl_level[jj,:] = 0

    reconstructor = LinearisedRecoFenics(Uelref, B, vincl_level, mesh_name=mesh_name)

    alphas = level_to_alphas[level]

    mean_score = 0
    for i in [1,2,3,4]:

        y = np.array(loadmat(f"ChallengeData/level_{level}/data{i}.mat")["Uel"])
        x = loadmat(f"GroundTruths/true{i}.mat")["truth"]

        ## get initial reconstruction 
        delta_sigma_list = reconstructor.reconstruct_list(y, alphas)

        delta_sigma_0 = reconstructor.interpolate_to_image(delta_sigma_list[0])
        delta_sigma_1 = reconstructor.interpolate_to_image(delta_sigma_list[1])
        delta_sigma_2 = reconstructor.interpolate_to_image(delta_sigma_list[2])
        delta_sigma_3 = reconstructor.interpolate_to_image(delta_sigma_list[3])
        delta_sigma_4 = reconstructor.interpolate_to_image(delta_sigma_list[4])

        sigma_reco = np.stack([delta_sigma_0, delta_sigma_1, delta_sigma_2, delta_sigma_3, delta_sigma_4])

        reco = torch.from_numpy(sigma_reco).float().to(device).unsqueeze(0)
        reco = torch.repeat_interleave(reco, repeats=sampling_params["num_samples"], dim=0)
        
        x_mean = sampler.sample(reco, logging=False)

        x_round = torch.round(x_mean).cpu().numpy()[:,0,:,:]
        x_round[x_round > 2] = 2.

        u = mode(x_round)[0][0,...]

        fig, (ax1, ax2) = plt.subplots(1,2)

        ax1.imshow(x)
        ax1.set_title("Ground truth")
        ax1.axis("off")

        ax2.imshow(u)
        ax2.set_title("Prediction")
        ax2.axis("off")

        plt.savefig(os.path.join(save_path, f"img_{i}.png"))
        plt.close()

if __name__ == '__main__':
    args = parser.parse_args()
    coordinator(args)
