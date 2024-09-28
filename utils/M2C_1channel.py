import cv2
import numpy as np
import os
from tqdm import tqdm
# import matplotlib.pyplot as plt
import cv2
from matplotlib import pyplot as plt
import os.path as osp
np.set_printoptions(threshold=np.inf)
import os
import multiprocessing
def Mask2Contour(folder, output_folder, category):
    category_dict = {
        "algriculture": 7,
        "barren": 5,
        "building": 2,
        "forest": 6,
        "road": 3,
        "water": 4,
    }
    files = os.listdir(folder)
    for file in tqdm(files):
        mask = cv2.imread(os.path.join(folder, file), cv2.IMREAD_UNCHANGED) # mask为标注图像
        fill_canvas = np.zeros_like(mask)

        V_LIST = list(np.unique(mask))
        if 1 in V_LIST:
            V_LIST.remove(1)
        if 0 in V_LIST:
            V_LIST.remove(0)

        canvas = np.zeros_like(mask)
        binary_mask = np.zeros_like(mask)
        binary_mask[mask == category_dict[category]] = 255  # building:2   barren:5   water:4   road:3    forest:6    agriculture:7
        _, thresh = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (255, 255, 255), 2)
        fill_canvas = fill_canvas + canvas.astype(np.uint32)

        fill_canvas[fill_canvas!= 0] = 255
        fill_canvas = fill_canvas.astype(np.uint8)
        cv2.imwrite(osp.join(output_folder, file), fill_canvas)

if __name__=="__main__":
    semantic_mask_folder = r"./LoveDA/mask_semantic_1024"
    categories = ["algriculture", "barren", "building", "forest", "road", "water"]
    LoveDA_mapping = {fr"./Datasets/edge/LoveDA_{category}": category for category in categories}

    with multiprocessing.Pool(processes=len(LoveDA_mapping)) as pool:
        pool.starmap(Mask2Contour,
                     [(semantic_mask_folder, output_folder, category) for output_folder, category in
                      LoveDA_mapping.items()])