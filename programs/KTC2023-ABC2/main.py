import re
import os
import argparse

from utils import process, solver

# Get input arguments
parser = argparse.ArgumentParser(description=
            'Reconstruct EIT images with different amounts of missing data.')

parser.add_argument('input_path', type=str,
                    help='Path with meausrements to be reconstructed.')
parser.add_argument('output_path', type=str,
                    help='Path to save the segmentation of the reconstructed images.')
parser.add_argument('category', type=int,
                    choices=range(1,8), metavar='[1-7]',
                    help='Difficulty category number.')

args = parser.parse_args()

# Get image names
img_names = sorted(os.listdir(args.input_path))
r = re.compile(".*\.mat")
img_names = list(filter(r.match,img_names))
# 过滤掉参考文件 ref.mat
img_names = [name for name in img_names if name != 'ref.mat']
print(f"{len(img_names)} images were found.")

for idx, img_name in enumerate(img_names, 1):
    print(f"\nProcessing image {idx}/{len(img_names)}: {img_name}")
    path_in = os.path.join(args.input_path, img_name)
    path_out = os.path.join(args.output_path, img_name)
    path_out = path_out[0:-3] + 'png'

    img_seg = solver.solve(path_in, args.category)

    process.save_img(img_seg, path_out)
    print(f"Saved result to: {path_out}")