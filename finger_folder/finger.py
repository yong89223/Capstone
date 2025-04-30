import cv2 as cv
from glob import glob
import os
import numpy as np
import random
from utils import poincare
from utils.segmentation import create_segmented_and_variance_images
from utils.normalization import normalize
from utils.gabor_filter import gabor_filter
from utils.frequency import ridge_freq
from utils import orientation
from utils.crossing_number import calculate_minutiaes
from tqdm import tqdm
from utils.skeletonize import skeletonize
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import math

def extract_singularities(angles, tolerance, W, mask):
    """
    :returns: list of tuples (x_center, y_center, type), type ∈ {'loop','delta','whorl'}
    """
    points = []
    H, V = len(angles), len(angles[0])
    for i in range(3, H-2):
        for j in range(3, V-2):
            # 5×5 블록 전부가 유효한 영역인지 확인
            mask_slice = mask[(i-2)*W:(i+3)*W, (j-2)*W:(j+3)*W]
            if np.sum(mask_slice) != (W*5)**2:
                continue

            typ = poincare.poincare_index_at(i, j, angles, tolerance)
            if typ != "none":
                # 블록 중심 픽셀 좌표로 환산
                x_ctr = int((j + 0.5) * W)
                y_ctr = int((i + 0.5) * W)
                points.append((x_ctr, y_ctr, typ))
    return points

def match_singularities(s1, s2, max_dist=20):
    """
    s1, s2: [(x,y,type), ...]
    max_dist: 같은 타입 매칭 시 최대 허용 거리 (픽셀 단위)
    :returns: 매칭된 쌍 개수
    """
    used = [False]*len(s2)
    matches = 0

    for x1, y1, t1 in s1:
        # 같은 타입 후보만 필터
        candidates = [(idx, x2, y2) 
                      for idx,(x2,y2,t2) in enumerate(s2)
                      if t2==t1 and not used[idx]]

        # 가장 가까운 점 찾기
        best = None
        best_d = max_dist
        for idx, x2,y2 in candidates:
            d = math.hypot(x1-x2, y1-y2)
            if d < best_d:
                best_d, best = d, idx

        if best is not None:
            matches += 1
            used[best] = True

    return matches

def singularity_similarity(s1, s2, max_dist=20):
    """
    2*|매칭| / (|s1| + |s2|)  →  [0,1]
    """
    m = match_singularities(s1, s2, max_dist)
    n1, n2 = len(s1), len(s2)
    if n1 + n2 == 0:
        return 1.0  # 둘 다 검출되지 않으면 완전 일치로 간주
    return 2*m / (n1 + n2)

def fingerprint_pipline(input_img):
    block_size = 16

    normalized_img = normalize(input_img.copy(), float(100), float(100))

    # ROI and normalisation
    (segmented_img, normim, mask) = create_segmented_and_variance_images(normalized_img, block_size, 0.08)

    # orientations
    angles = orientation.calculate_angles(normalized_img, W=block_size, smoth=False)
    orientation_img = orientation.visualize_angles(segmented_img, mask, angles, W=block_size)


    # find the overall frequency of ridges in Wavelet Domain
    freq = ridge_freq(normim, mask, angles, block_size, kernel_size=7, minWaveLength=3, maxWaveLength=30)


    # create gabor filter and do the actual filtering
    gabor_img = gabor_filter(normim, angles, freq)

    # thinning oor skeletonize
    thin_image = skeletonize(gabor_img)

    cv.imshow("zz",thin_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    # minutias
    minutias = calculate_minutiaes(thin_image)

    # singularities
    s1 = extract_singularities(angles, 1, block_size, mask)

    return s1

if __name__ == '__main__':
    img1 = cv.imread('finger1.png', 0)
    img2 = cv.imread('finger2.png', 0)
    s1 = fingerprint_pipline(img1)
    s2 = fingerprint_pipline(img2)
    sim = singularity_similarity(s1, s2, max_dist=20)
    print(f"Singularity-based similarity: {sim:.2%}")
    





