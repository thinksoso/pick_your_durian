import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

from segment_anything import SamPredictor, sam_model_registry,SamAutomaticMaskGenerator
from matplotlib.font_manager import FontProperties
from metric import sep_body_mask,calculate_eccentricity,calculate_sparsity


def parse_arguments():
    parser = argparse.ArgumentParser(description="A simple command line argument parser")
    parser.add_argument("--source", type=str, required=False, help="path of source figure")
    parser.add_argument("--target", type=str, required=False, help="Save Path of result")
    args = parser.parse_args()
    
    return args

sam = sam_model_registry["vit_h"](checkpoint="./sam_vit_h_4b8939.pth")

if torch.cuda.is_available():
    sam.to(device='cuda')

mask_generator = SamAutomaticMaskGenerator(sam)

def show_anns(sorted_anns):
    if len(sorted_anns) == 0:
        return
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def save_fig(body_mask,prick_masks,target,image,ecc,spa):
    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_anns(body_mask)
    show_anns(prick_masks)
    plt.text(100, 50, f"eccentricity:{ecc}", color='red', fontsize=50)
    plt.text(100, 100, f"sparsity:{spa}", color='red', fontsize=50)
    plt.text(100, 150, f"prick nums:{len(prick_masks)}", color='red', fontsize=50)
    plt.axis('off')
    print("## saved")
    plt.savefig(target)

def judge_quality(body_mask, prick_masks):
    return calculate_eccentricity(body_mask), calculate_sparsity(prick_masks)

def seg_angthing(source,target):
    image = cv2.imread(source)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    masks = mask_generator.generate(image)
    masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    body_mask, prick_masks = sep_body_mask(masks)
    ecc, spa = judge_quality(body_mask,prick_masks)
    print("榴莲圆度：",ecc)
    print("榴莲刺稀疏度：",spa)
    print("综合评分：",round(spa+ecc,2))
    body_mask = [body_mask]
    save_fig(body_mask,prick_masks,target,image,ecc,spa)

if __name__ == "__main__":
    args = parse_arguments()
    if args.source:
        seg_angthing(args.source,args.target)
    else:
        seg_angthing("1.jpg","r1.png")