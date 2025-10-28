import os
import numpy as np
from scipy.io import loadmat

import torch 
from torch.utils.data import Dataset

import math, random
from typing import List, Tuple
from PIL import Image, ImageDraw
from scipy.interpolate import NearestNDInterpolator, interpn
from scipy.ndimage import rotate

from ..ktc_methods import load_mesh


class SimulatedTargets(Dataset):
    """
    Generates a dataset containing synthesized phantoms.
    WARNING: two calls of getitem produce seperate samples
    """
    def __init__(self, length=1000, mesh_name = "Mesh_dense.mat"):
        assert mesh_name in ["Mesh_dense.mat", "Mesh_sparse.mat"], "Unknown mesh: " + mesh_name

        self.length = length 
        self.Nel = 32 # Number of electrodes 
        self.z = (1e-6)*np.ones((self.Nel, 1))  # contact impedances

        self.mesh, self.mesh2 = load_mesh(mesh_name=mesh_name)

    def __len__(self):
        return self.length

    def __getitem__(self, IDX):

        sigma_pix = create_phantoms()

        phantom_pix = torch.from_numpy(sigma_pix).float()

        return phantom_pix

class HanddrawnImages(Dataset):
    """
    Return handdrawn images

    """
    def __init__(self, path_to_images = "/home/adenker/projects/ktc2023/dl_for_ktc2023/data/KTC_handdrawn_images", rotate=True):

        self.path_to_images = path_to_images
        self.rotate = rotate 

        self.img_files = [f for f in os.listdir(self.path_to_images) if f.endswith(".png")]

        pixwidth = 0.23 / 256

        pixcenter_x = np.linspace(-0.115 + pixwidth / 2, 0.115 - pixwidth / 2 + pixwidth, 256)
        pixcenter_y = pixcenter_x
        self.X, self.Y = np.meshgrid(pixcenter_x, pixcenter_y)


    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, IDX):

                
        im_frame = Image.open(os.path.join(self.path_to_images, self.img_files[IDX])).convert("L")
        img = np.array(im_frame)


        img[self.X**2 + self.Y**2 > 0.113**2] = 0.0

        unique_elem = np.sort(np.unique(img))

        print("Working on image ", self.img_files[IDX])
        
        
        assert len(unique_elem) <= 3, "ONLY THREE CLASSES ARE ALLOWED"
        assert len(unique_elem) > 1, "AT LEAST TWO CLASSES"

        if len(unique_elem) == 3: 
            # we have all three classes 
            img[img == unique_elem[1]] = 1.
            img[img == unique_elem[2]] = 2.
        else:
            # only two classes 
            # randomly decide if its conductive or resistive
            if np.random.rand() < 0.5:
                img[img == unique_elem[1]] = 1.
            else:
                img[img == unique_elem[1]] = 2.

        if self.rotate:
            angle = np.random.randint(0, 180)
            img = np.round(rotate(img,angle, mode="constant", cval=0.0, reshape=False, order=0))

        return img


"""
Return only gt and measurment

"""
class TrainingData(Dataset):
    def __init__(self, Uref, InvLn, base_path="/localdata/AlexanderDenker/KTC2023/dataset"):

        self.base_path = base_path

        #self.file_list = [f.split(".")[0].split("_")[-1] for f in os.listdir(os.path.join(self.base_path, "gt" ))]
        self.file_list = [f for f in os.listdir(os.path.join(self.base_path, "gt" ))]

        self.length = len(self.file_list)
        print("Number of images " + str(self.length))

        self.Uref = Uref
        self.InvLn = InvLn

    def __len__(self):
        return self.length

    def __getitem__(self, IDX):
        gt_name = self.file_list[IDX]
        reco_name = self.file_list[IDX].replace("gt", "u")

        gt_np = np.load(os.path.join(self.base_path, "gt", gt_name))
        measurements = np.load(os.path.join(self.base_path, "measurements", reco_name))
        
        Uref_nois = self.Uref + self.InvLn * np.random.randn(self.Uref.shape[0],1)
        measurements = measurements - Uref_nois

        gt = np.zeros((3, 256,256))
        gt[0, :, :] = (gt_np == 0)
        gt[1, :, :] = (gt_np == 1) 
        gt[2, :, :] = (gt_np == 2) 

        gt = torch.from_numpy(gt).float() 
        
        return torch.from_numpy(measurements).float()[:,0], gt




"""
SimData returns GT, Initial Reco and Level
"""
class SimData(Dataset):
    def __init__(self, level, base_path="/localdata/AlexanderDenker/KTC2023/dataset"):

        self.level = level
        self.base_path = os.path.join(base_path, "level_" + str(self.level))

        #self.file_list = [f.split(".")[0].split("_")[-1] for f in os.listdir(os.path.join(self.base_path, "gt" ))]
        self.file_list = [f for f in os.listdir(os.path.join(self.base_path, "gt" ))]

        self.length = len(self.file_list)
        print("Number of images " + str(self.length) + " for level " + str(self.level))

    def __len__(self):
        return self.length

    def __getitem__(self, IDX):
        gt_name = self.file_list[IDX]
        reco_name = self.file_list[IDX].replace("gt", "recos")

        gt_np = np.load(os.path.join(self.base_path, "gt", gt_name))
        init_reco = np.load(os.path.join(self.base_path, "gm_reco", reco_name))
        
        gt = np.zeros((3, 256,256))
        gt[0, :, :] = (gt_np == 0)
        gt[1, :, :] = (gt_np == 1) 
        gt[2, :, :] = (gt_np == 2) 

        gt = torch.from_numpy(gt).float() 
        
        return torch.from_numpy(init_reco).float(), gt, torch.tensor(self.level).float()

class MmapDataset(Dataset):
    def __init__(self, level, num_samples, base_path="/localdata/AlexanderDenker/KTC2023/mmap_dataset"):
        super(MmapDataset, self).__init__()

        self.level = level

        gt_file = os.path.join(base_path, f"gt_level={self.level}_size={num_samples}.npy")
        reco_file = os.path.join(base_path, f"recos_level={self.level}_size={num_samples}.npy")

        self.gts = np.load(gt_file, mmap_mode='r')#np.memmap(x_file, mode='r', shape=(num_samples, 3, 96, 96), dtype='float32')
        self.recos = np.load(reco_file, mmap_mode='r')#np.memmap(y_file, mode='r', shape=(num_samples, 3, 96, 96), dtype='float32')

    def __getitem__(self, item):
        if self.level == 5 and item == 10506:
            # this file is corrupted
            gt_np = np.copy(self.gts[0])
            reco = np.copy(self.recos[0])                  
        else:
            gt_np = np.copy(self.gts[item])
            reco = np.copy(self.recos[item])
        #gt_seg = np.zeros((3, 256,256))
        #gt_seg[0, :, :] = (gt_np == 0)
        #gt_seg[1, :, :] = (gt_np == 1) 
        #gt_seg[2, :, :] = (gt_np == 2) 

        gt_np = torch.from_numpy(gt_np).float() 

        return torch.from_numpy(reco).float(), gt_np, torch.tensor(self.level).float()
    
    def __len__(self):
        return self.gts.shape[0]



def create_phantoms(min_inclusions=1, max_inclusions = 4, max_iter=80, distance_between=25, p=[0.7,0.15,0.15]):
    
    rectangle_dict = {'min_width': 25, 'max_width': 50, "min_height": 40, "max_height": 120}
    circle_dict = {}

    # simulate the same sigma for the image 
    pixwidth = 0.23 / 256
    # pixcenter_x = np.arange(-0.115 + pixwidth / 2, 0.115 - pixwidth / 2 + pixwidth, pixwidth)
    pixcenter_x = np.linspace(-0.115 + pixwidth / 2, 0.115 - pixwidth / 2 + pixwidth, 256)
    pixcenter_y = pixcenter_x
    X, Y = np.meshgrid(pixcenter_x, pixcenter_y, indexing="ij")
    pixcenters = np.column_stack((X.ravel(), Y.ravel()))

    I = np.zeros((256, 256))
    im = Image.fromarray(np.uint8(I))

    draw = ImageDraw.Draw(im)

    num_forms = np.random.randint(min_inclusions, max_inclusions)

    circle_list = [] 

    iter = 0
    while(len(circle_list) < num_forms):
        object_type = np.random.choice(["polygon", "circle", "rectangle"], p=p)
        
        if object_type == "rectangle":
            lower_x = 50 + np.random.randint(-24, 24)
            lower_y = 50 + np.random.randint(-24, 24)

            width = np.random.randint(rectangle_dict["min_width"], rectangle_dict["max_width"])
            height = np.random.randint(rectangle_dict["min_height"], rectangle_dict["max_height"])

            center_x = lower_x + width/2
            center_y = lower_y + height/2
            avg_radius = max(width/2, height/2)

        else:

            avg_radius = np.random.randint(25, 50)
            
            center_x = 128 + np.random.randint(-54, 54)
            center_y = 128 + np.random.randint(-54, 54)

        # collision detection (check circles around objects)
        collide = False
        for x, y, r in circle_list:
            d = (center_x - x)**2 + (center_y - y)**2
            if d < (avg_radius + r + distance_between)**2:
                collide = True
                break

        if not collide:
            # polygon or circle or rectangle 

            if object_type == "rectangle":
                draw.rectangle([lower_x, lower_y, lower_x + width, lower_y + height], fill=1 if np.random.rand() < 0.5 else 2)

            elif object_type == "circle":
                draw.ellipse((center_x-avg_radius, center_y-avg_radius, center_x+avg_radius, center_y+avg_radius), fill=1 if np.random.rand() < 0.5 else 2)

            elif object_type == "polygon":
                num_vertices = np.random.randint(5, 9)
                vertices = generate_polygon(center=(center_x, center_y),
                                    avg_radius=avg_radius,
                                    irregularity=0.4,
                                    spikiness=0.3,
                                    num_vertices=num_vertices)
                
                draw.polygon(vertices, fill=1 if np.random.rand() < 0.5 else 2)

            else:
                print("Unknown Object type")

            circle_list.append((center_x, center_y, avg_radius))
            
        iter = iter + 1 
        if iter > max_iter:
            break

    sigma_pix = np.array(np.asarray(im))
    sigma_pix[X**2 + Y**2 > 0.098**2] = 0.0
    angle = np.random.randint(0, 180)
    sigma_pix = np.round(rotate(sigma_pix,angle, mode="constant", cval=0.0, reshape=False, order=0))

    #sigma = interpn([pixcenter_x, pixcenter_y], sigma_pix, mesh.g, 
    #            bounds_error=False, fill_value=0.0, method="nearest")

    #return (np.expand_dims(sigma_pix, 0), np.expand_dims(sigma, 0))

    return sigma_pix

def generate_polygon(center: Tuple[float, float], avg_radius: float,
                     irregularity: float, spikiness: float,
                     num_vertices: int) -> List[Tuple[float, float]]:
    """
    https://stackoverflow.com/questions/8997099/algorithm-to-generate-random-2d-polygon

    Start with the center of the polygon at center, then creates the
    polygon by sampling points on a circle around the center.
    Random noise is added by varying the angular spacing between
    sequential points, and by varying the radial distance of each
    point from the centre.

    Args:
        center (Tuple[float, float]):
            a pair representing the center of the circumference used
            to generate the polygon.
        avg_radius (float):
            the average radius (distance of each generated vertex to
            the center of the circumference) used to generate points
            with a normal distribution.
        irregularity (float):
            variance of the spacing of the angles between consecutive
            vertices.
        spikiness (float):
            variance of the distance of each vertex to the center of
            the circumference.
        num_vertices (int):
            the number of vertices of the polygon.
    Returns:
        List[Tuple[float, float]]: list of vertices, in CCW order.
    """
    # Parameter check
    if irregularity < 0 or irregularity > 1:
        raise ValueError("Irregularity must be between 0 and 1.")
    if spikiness < 0 or spikiness > 1:
        raise ValueError("Spikiness must be between 0 and 1.")

    irregularity *= 2 * math.pi / num_vertices
    spikiness *= avg_radius
    angle_steps = random_angle_steps(num_vertices, irregularity)

    # now generate the points
    points = []
    angle = random.uniform(0, 2 * math.pi)
    for i in range(num_vertices):
        radius = clip(random.gauss(avg_radius, spikiness), 0, 2 * avg_radius)
        point = (center[0] + radius * math.cos(angle),
                 center[1] + radius * math.sin(angle))
        points.append(point)
        angle += angle_steps[i]

    return points


def random_angle_steps(steps: int, irregularity: float) -> List[float]:
    """Generates the division of a circumference in random angles.

    https://stackoverflow.com/questions/8997099/algorithm-to-generate-random-2d-polygon

    Args:
        steps (int):
            the number of angles to generate.
        irregularity (float):
            variance of the spacing of the angles between consecutive vertices.
    Returns:
        List[float]: the list of the random angles.
    """
    # generate n angle steps
    angles = []
    lower = (2 * math.pi / steps) - irregularity
    upper = (2 * math.pi / steps) + irregularity
    cumsum = 0
    for i in range(steps):
        angle = random.uniform(lower, upper)
        angles.append(angle)
        cumsum += angle

    # normalize the steps so that point 0 and point n+1 are the same
    cumsum /= (2 * math.pi)
    for i in range(steps):
        angles[i] /= cumsum
    return angles

def clip(value, lower, upper):
    """
    https://stackoverflow.com/questions/8997099/algorithm-to-generate-random-2d-polygon

    Given an interval, values outside the interval are clipped to the interval
    edges.
    """
    return min(upper, max(value, lower))


if __name__ == "__main__":
    import matplotlib.pyplot as plt 
    from ..ktc_methods import SigmaPlotter 

    dataset = SimulatedTargets()

    sigma_img, sigma = dataset[0]

    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(sigma_img[0,:,:].T, origin="lower")

    pixwidth = 0.23 / 256
    pixcenter_x = np.linspace(-0.115 + pixwidth / 2, 0.115 - pixwidth / 2 + pixwidth, 256)
    pixcenter_y = pixcenter_x
    X, Y = np.meshgrid(pixcenter_x, pixcenter_y, indexing="ij")
    ax2.pcolormesh(X, Y, sigma_img[0,:,:])
    ax2.set_title("pcolormesh")

    plt.figure() 
    sgplot = SigmaPlotter(dataset.mesh, [2, 3], 'jet')
    sgplot.plot_solution(dataset.mesh.g, dataset.mesh.H, sigma)
    plt.colorbar()
    plt.show()