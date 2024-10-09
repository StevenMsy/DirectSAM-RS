import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
os.environ['NCCL_P2P_DISABLE'] = "1"
os.environ['NCCL_IB_DISABLE'] = "1"
from PIL import Image
import numpy as np
import torch
from torch import nn
import os.path as osp
from tqdm import tqdm
import cv2
from transformers import AutoImageProcessor
from model import SegformerWithTextFusion
from scipy.io import loadmat
from impl.toolbox import conv_tri, grad2
from ctypes import *

solver = cdll.LoadLibrary("cxx/lib/solve_csa.so")
c_float_pointer = POINTER(c_float)
solver.nms.argtypes = [c_float_pointer, c_float_pointer, c_float_pointer, c_int, c_int, c_float, c_int, c_int]


def nms_process_one_image(image, save_path=None, save=True):
    """"
    :param image: numpy array, edge, model output
    :param save_path: str, save path
    :param save: bool, if True, save .png
    :return: edge
    NOTE: in MATLAB, uint8(x) means round(x).astype(uint8) in numpy
    """

    if save and save_path is not None:
        assert os.path.splitext(save_path)[-1] == ".png"
    edge = conv_tri(image, 1)
    edge = np.float32(edge)
    ox, oy = grad2(conv_tri(edge, 4))
    oxx, _ = grad2(ox)
    oxy, oyy = grad2(oy)
    ori = np.mod(np.arctan(oyy * np.sign(-oxy) / (oxx + 1e-5)), np.pi)
    out = np.zeros_like(edge)
    r, s, m, w, h = 1, 5, float(1.01), int(out.shape[1]), int(out.shape[0])
    solver.nms(out.ctypes.data_as(c_float_pointer),
               edge.ctypes.data_as(c_float_pointer),
               ori.ctypes.data_as(c_float_pointer),
               r, s, m, w, h)
    edge = np.round(out * 255).astype(np.uint8)
    if save:
        cv2.imwrite(save_path, edge)
    return edge


def nms_process(model_name_list, result_dir, save_dir, key=None, file_format=".mat"):  #file format means input format，可以是mat和npy的.
    if not isinstance(model_name_list, list):
        model_name_list = [model_name_list]
    assert file_format in {".mat", ".npy"}
    assert os.path.isdir(result_dir)

    for model_name in model_name_list:
        model_save_dir = os.path.join(save_dir, model_name)
        if not os.path.isdir(model_save_dir):
            os.makedirs(model_save_dir)

        for file in tqdm(os.listdir(result_dir)):
            # print(file)
            save_name = os.path.join(model_save_dir, "{}.png".format(os.path.splitext(file)[0])) #after the nms post process, this will generate a lot png result
            # print(save_name)
            # print(os.path.isfile(save_name))
            if os.path.isfile(save_name):
                continue
            # print(os.path.splitext(file)[-1] != file_format)
            if os.path.splitext(file)[-1] != file_format:
                continue
            abs_path = os.path.join(result_dir, file)
            if file_format == ".mat":
                assert key is not None
                image = loadmat(abs_path)[key]
            elif file_format == ".npy":
                image = np.load(abs_path)
            else:
                raise NotImplementedError
            nms_process_one_image(image, save_name, True)

def process_one_image(image, image_processor, model,word_mask, local_text_feature,device,resolution=1024):
    image_processor.size['height'] = resolution
    image_processor.size['width'] = resolution
    word_mask=word_mask.to(device)
    local_text_feature = local_text_feature.to(device)
    encoding = image_processor(image, return_tensors="pt")
    pixel_values = encoding.pixel_values.to(model.device)
    word_mask=word_mask.unsqueeze(1)
    local_text_feature=local_text_feature.unsqueeze(1)
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values,
                        word_mask=word_mask,
                        local_text_feature=local_text_feature)
    logits = outputs.logits.float().cpu()

    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )
    probs = torch.sigmoid(upsampled_logits)[0, 0].detach().numpy()
    return probs

def probs_to_masks(probs, threshold=0.1):
        
    binarilized = (probs < threshold).astype(np.uint8)
    num_objects, labels = cv2.connectedComponents(binarilized)
    masks = [labels == i for i in range(1, labels.max() + 1)]
    masks.sort(key=lambda x: x.sum(), reverse=True)
    return masks


def visualize_labels(image, masks):
    canvas = np.ones_like(image) * 255

    for i in range(len(masks)):
        mask = masks[i]
        color = np.mean(image[mask], axis=0)
        canvas[mask] = color
    return canvas

def resize_to_max_length(image, max_length):
    width, height = image.size
    if width > height:
        new_width = max_length
        new_height = int(height * (max_length / width))
    else:
        new_height = max_length
        new_width = int(width * (max_length / height))
    return image.resize((new_width, new_height))



if __name__=="__main__":
    png = False
    nms = True
    device=torch.device("cuda:0")
    input_sample_path = "./test_origin/"
    target_root = "./infer_result/"
    if not os.path.exists(target_root):
        os.makedirs(target_root)
    checkpoint = r"./weight"
    local_text_f = "./text_feature.pth"
    local_dict = torch.load(local_text_f)
    image_processor = AutoImageProcessor.from_pretrained(checkpoint, reduce_labels=True)
    # print(image_processor)
    model = SegformerWithTextFusion.from_pretrained(checkpoint, ignore_mismatched_sizes=True)
    model.to(device)
    for item in tqdm(os.listdir(input_sample_path)):
        image = Image.open(osp.join(input_sample_path, item)).convert("RGB")
        local_text_feature = local_dict["forest_feature"]
        word_mask = local_dict["forest_mask"]
        probs = process_one_image(image, image_processor, model, word_mask=word_mask, local_text_feature=local_text_feature,device=device,resolution=1024)
        if png:
            probs *= 255.0
            probs_int = probs.astype(np.uint8)
            cv2.imwrite(osp.join(target_root,item), probs_int)
            # NMS is for NPY format
        else:
            np.save(osp.join(target_root, item).replace(".png", ".npy"), probs)
    if nms:
        nms_process("nms", result_dir=target_root, save_dir=target_root, key="result", file_format=".npy")














# /home/liaoshiyu/ new_ovsegformer/checkpoint/visual_zero/checkpoint-253