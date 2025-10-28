
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

from configs.fcunet_config import get_configs
from src import get_fcunet, FastScoringFunction


def create_vincl(level, Injref, Nel=32):
    vincl_level = np.ones(((Nel - 1),76), dtype=bool) 
    rmind = np.arange(0,2 * (level - 1),1) #electrodes whose data is removed

    #remove measurements according to the difficulty level
    for ii in range(0,75):
        for jj in rmind:
            if Injref[jj,ii]:
                vincl_level[:,ii] = 0
            vincl_level[jj,:] = 0

    return vincl_level




level_to_model_path = { 
    1: "/localdata/AlexanderDenker/KTC2023/FCUNet/finetune/version_12/model.pt",   #"/localdata/AlexanderDenker/KTC2023/level_cond_unet/version_01/model.pt"
    2: "fcunet_model/model.pt",
    3: "fcunet_model/model.pt",
    4: "fcunet_model/model.pt",
    5: "fcunet_model/model.pt",
    6: "fcunet_model/model.pt",
    7: "fcunet_model/model.pt",
}



parser = argparse.ArgumentParser(description='reconstruction using postprocessing on challenge data')
parser.add_argument('level')

def coordinator(args):
    level = int(args.level)
    device =  "cuda" if torch.cuda.is_available() else "cpu"


    print("Level: ", args.level)

    ### load conditional diffusion model 
    config = get_configs()
        
    model = get_fcunet(config)
    model.load_state_dict(torch.load(level_to_model_path[1]))
    model.eval()
    model.to(device)

    save_path = f"examples/level_{level}/"
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    y_ref = loadmat(f"ChallengeData/level_{level}/ref.mat")
    Injref = y_ref["Injref"]
    Mpat = y_ref["Mpat"]
    Uelref = y_ref["Uelref"]

    vincl = create_vincl(level, Injref).T.flatten()
    

    mean_score = 0
    for i in [1,2,3,4]:
        print("Start processing ", i)
        y = np.array(loadmat(f"ChallengeData/level_{level}/data{i}.mat")["Uel"])
        x = loadmat(f"GroundTruths/true{i}.mat")["truth"]

        y = y - Uelref
        y[~vincl] = 0

        y = torch.from_numpy(y).float().to(device).T
        level_input = torch.tensor([level]).to(device)

        with torch.no_grad():
            pred = model(y, level_input)

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
