import cv2
import numpy as np
from PIL import Image
import os
import os.path as osp
from tqdm import tqdm
import multiprocessing
def Mask2Contour(folder, output_folder, category):
    category_dict = {
        "plane": (0, 127, 255),
        "ship": (0, 0, 63),
        "bridge": (0, 127, 63),
        "large vehicle": (0, 127, 127),
        "small vehicle": (0, 0, 127),
        "helicopter": (0, 0, 191),
        "harbor": (0, 100, 155),
        "swimming pool": (0, 0, 255),
        "tennis court": (0, 63, 127),
        "soccer ball field": (0, 127, 191),
        "storage tank": (0, 63, 63),
        "basketball court": (0, 63, 191),
        "baseball diamond": (0, 63, 0),
        "ground track field": (0, 63, 255),
        "roundabout": (0, 191, 127),
    }
    r, g, b = category_dict[category]

    files = os.listdir(folder)
    for file in tqdm(files):
        complete_path = os.path.join(folder, file)
        semantic_mask = cv2.imread(complete_path, cv2.IMREAD_UNCHANGED)
        channel_B = semantic_mask[:, :, 0]
        channel_G = semantic_mask[:, :, 1]
        channel_R = semantic_mask[:, :, 2]
        binary_mask = np.zeros_like(channel_B)
        binary_edge = np.zeros_like(channel_B)
        binary_mask[(channel_B == b) & (channel_G == g) & (channel_R == r)] = 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(binary_edge, contours, -1, (255, 255, 255), 2)
        out_name = f"ISAID_{category}_{file}"
        output_path = os.path.join(output_folder, out_name)
        cv2.imwrite(output_path, binary_edge)

if __name__=="__main__":
    semantic_mask_folder = r"./iSAID/mask_semantic_1024"
    categories = ["plane", "ship", "bridge", "large vehicle", "small vehicle", "helicopter", "harbor", "swimming pool", "tennis court",
                  "soccer ball field", "storage tank", "basketball court", "baseball diamond", "ground track field", "roundabout"]
    iSAID_mapping = {fr"./Datasets/edge/iSAID_{category}": category for category in categories}

    with multiprocessing.Pool(processes=len(iSAID_mapping)) as pool:
        pool.starmap(Mask2Contour,
                     [(semantic_mask_folder, output_folder, category) for output_folder, category in iSAID_mapping.items()])

