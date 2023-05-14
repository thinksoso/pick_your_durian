# 判断榴莲的好坏
# 1. 刺大而稀疏
# 2. 果型比较圆
# 3. 奶油尖尖（不太好通过分割看出来）


# 离心率计算
import numpy as np
import cv2
import math


def calculate_eccentricity(mask):
    # 计算多边形的质心（重心）：计算多边形所有顶点的坐标之和，然后除以顶点数量，可以得到多边形的质心坐标。
    # 计算多边形顶点到质心的平均距离：对于多边形的每个顶点，计算它到质心的距离，并将所有距离求和，最后除以顶点数量，可以得到平均距离。
    # 计算多边形顶点到质心的最大距离：对于多边形的每个顶点，计算它到质心的距离，并找到最大距离。
    # 计算离心率：离心率的计算公式为：离心率 = (最大距离 - 平均距离) / 最大距离。
    # 判断形状：根据离心率的值进行判断，如果离心率接近于0，则形状更接近圆形；如果离心率接近于1，则形状更接近椭圆形

    # 找到边界
    mask = mask["segmentation"]
    true_indices = np.where(mask)
    min_row, min_col = np.min(true_indices, axis=1)
    max_row, max_col = np.max(true_indices, axis=1)

    # 计算质心
    centroid_row = (min_row + max_row) / 2
    centroid_col = (min_col + max_col) / 2

    # 计算距离
    distances = np.sqrt((true_indices[0] - centroid_row)**2 + (true_indices[1] - centroid_col)**2)
    average_distance = np.mean(distances)
    max_distance = np.max(distances)

    # 计算离心率， 用倒数，越大越好
    eccentricity = -math.log2(((max_distance - average_distance)) / max_distance)
    eccentricity = round(eccentricity,2)

    return eccentricity

# 判断果形，用离心率指标
def body_shape(body_mask):
    # 找到body的图像 按面积排序
    # 判断果型，计算离心率
    ecc = calculate_eccentricity(body_mask["segmentation"])
    return ecc

def sep_body_mask(sorted_anns):
    body_mask = None
    for i,ann in enumerate(sorted_anns):
        if ann["bbox"][0] != 0 and ann["bbox"][1] != 0:
            body_mask = ann
            prick_masks_tmp = sorted_anns[i+1: ]
            break
    
    # 确定刺在body里面
    body_x1 = body_mask["bbox"][0]
    body_x2 = body_x1 + body_mask["bbox"][2]
    body_y1 = body_mask["bbox"][1]
    body_y2 = body_y1 + body_mask["bbox"][3]

    prick_masks = []
    for p in prick_masks_tmp:
        coords = p["point_coords"][0]
        if coords[0]>body_x1 and coords[0] < body_x2 and coords[1] > body_y1 and coords[1] < body_y2:
            prick_masks.append(p)
    return body_mask, prick_masks


# 计算刺的紧凑程度，希望刺是大而稀疏
def calculate_sparsity(prick_masks):
    num_thorns = len(prick_masks)

    total_distance = 0
    total_area = 0

    for i in range(num_thorns):
        prick = prick_masks[i]
        area = prick["area"]  # 假设刺的面积为刺的x坐标乘以y坐标
        total_area += area
        x1 = prick["point_coords"][0][0]
        y1 = prick["point_coords"][0][1]

        for j in range(i + 1, num_thorns):
            prick_other = prick_masks[j]
            x2 = prick_other["point_coords"][0][0]
            y2 = prick_other["point_coords"][0][1]
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            total_distance += distance

    average_distance = total_distance / (num_thorns * (num_thorns - 1) / 2)
    average_area = total_area / num_thorns

    sparsity = round(average_area/average_distance,2)
    return sparsity
