import numpy as np
import os
from PIL import Image

def save_img(img_np, fname):
  # 自动创建目录（如果不存在）
  os.makedirs(os.path.dirname(fname), exist_ok=True)

  img_np = normalize(img_np,0,2)
  ar = np.clip(img_np*255,0,255).astype(np.uint8)

  img = Image.fromarray(ar)
  img.save(fname)

def normalize(x, vmin=None, vmax=None):
  if vmin is None:
    vmin = x.min()
  if vmax is None:
    vmax = x.max()

  return (x - vmin)/(vmax - vmin)
