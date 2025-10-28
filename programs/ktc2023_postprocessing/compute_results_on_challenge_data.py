
import os 
from scipy.io import loadmat, savemat
import torch 
import torch.nn.functional as F

import argparse 
import numpy as np 
from scipy.stats import mode
from pathlib import Path 

# only for testing 
import matplotlib.pyplot as plt 

from configs.postprocessing_config import get_configs
from src import get_model, LinearisedRecoFenics, FastScoringFunction


level_to_model_path = { 
    1: "postprocessing_model/version_01/model.pt",
    2: "postprocessing_model/version_01/model.pt",
    3: "postprocessing_model/version_01/model.pt",
    4: "postprocessing_model/version_01/model.pt",
    5: "postprocessing_model/version_01/model.pt",
    6: "postprocessing_model/version_01/model.pt",
    7: "postprocessing_model/version_01/model.pt",
}



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



parser = argparse.ArgumentParser(description='reconstruction using postprocessing on challenge data')
parser.add_argument('level')

def coordinator(args):
    level = int(args.level)
    device =  "cuda" if torch.cuda.is_available() else "cpu"


    print("Level: ", args.level)

    ### load conditional diffusion model 
    config = get_configs()
        
    model = get_model(config)
    model.load_state_dict(torch.load(level_to_model_path[level]))
    model.eval()
    model.to(device)

    save_path = f"examples/level_{level}/"
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

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
        print("Start processing ", i)
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
        level_input = torch.tensor([level]).to("cuda")

        with torch.no_grad():
            pred = model(reco, level_input)

            pred_softmax = F.softmax(pred, dim=1)
            pred_argmax = torch.argmax(pred_softmax, dim=1).cpu().numpy()[0,:,:]

        #challenge_score = FastScoringFunction(x, pred_argmax)
        #mean_score += challenge_score
        #print(f"Score on data {i} is: {challenge_score}")

        fig, (ax1, ax2) = plt.subplots(1,2)

        ax1.imshow(x)
        ax1.set_title("Ground truth")
        ax1.axis("off")

        ax2.imshow(pred_argmax)
        ax2.set_title("Prediction")
        ax2.axis("off")

        plt.savefig(os.path.join(save_path, f"img_{i}.png"))
        plt.close()

    #print(f"Mean score at level {level} is: {mean_score/4.}")


if __name__ == '__main__':
    args = parser.parse_args()
    coordinator(args)
