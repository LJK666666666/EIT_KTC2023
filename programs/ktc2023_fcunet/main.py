
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
from src import get_fcunet


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
    1: "fcunet_model/model.pt",
    2: "fcunet_model/model.pt",
    3: "fcunet_model/model.pt",
    4: "fcunet_model/model.pt",
    5: "fcunet_model/model.pt",
    6: "fcunet_model/model.pt",
    7: "fcunet_model/model.pt",
}


parser = argparse.ArgumentParser(description='reconstruction using FCUNet')
parser.add_argument('input_folder')
parser.add_argument('output_folder')
parser.add_argument('level')

def coordinator(args):
    level = int(args.level)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Input folder: ", args.input_folder)
    print("Output folder: ", args.output_folder)
    print("Level: ", args.level)

    output_path = Path(args.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    ### load model 
    config = get_configs()
        
    model = get_fcunet(config)
    model.load_state_dict(torch.load(level_to_model_path[level]))
    model.eval()
    model.to(device)

    ### read files from args.input_folder 
    # there will be ref.mat in the input_folder, dont process this 
    f_list = [f for f in os.listdir(args.input_folder) if f != "ref.mat"]

    y_ref = loadmat(os.path.join(args.input_folder, "ref.mat"))
    Injref = y_ref["Injref"]
    Uelref = y_ref["Uelref"]


    vincl = create_vincl(level, Injref).T.flatten()
    

    for f in f_list: 
        print("Start processing ", f)
        y = np.array(loadmat(os.path.join(args.input_folder, f))["Uel"])

        y = y - Uelref
        y[~vincl] = 0

        y = torch.from_numpy(y).float().to(device).T
        level = torch.tensor([level]).to(device)

        with torch.no_grad():
            pred = model(y, level)

            pred_softmax = F.softmax(pred, dim=1)
            pred_argmax = torch.argmax(pred_softmax, dim=1).cpu().numpy()[0,:,:]

        
        
        mdic = {"reconstruction": pred_argmax.astype(int)}

        objectno = f.split(".")[0][-1]

        savemat( os.path.join(output_path, str(objectno) + ".mat"),mdic)
        

    ### save reconstructions to args.output_folder 
    # as a .mat file containing a 256x256 pixel array with the name {file_idx}.mat 
    # the pixel array must be named "reconstruction" and is only allowed to have 0, 1 or 2 as values.


if __name__ == '__main__':
    args = parser.parse_args()
    coordinator(args)
