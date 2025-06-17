
import numpy as np
import cv2 as cv


def normalise(img):
    return (img - np.mean(img))/(np.std(img))


def create_segmented_and_variance_images(im, w, threshold=.1):
    
    (y, x) = im.shape
    threshold = np.std(im)*threshold

    image_variance = np.zeros(im.shape)
    segmented_image = im.copy()
    mask = np.ones_like(im)

    for i in range(0, x, w):
        for j in range(0, y, w):
            box = [i, j, min(i + w, x), min(j + w, y)]
            block_stddev = np.std(im[box[1]:box[3], box[0]:box[2]])
            image_variance[box[1]:box[3], box[0]:box[2]] = block_stddev

    # apply threshold
    mask[image_variance < threshold] = 0

    # smooth mask with a open/close morphological filter
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(w*2, w*2))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    # normalize segmented image
    segmented_image *= mask
    im = normalise(im)
    if np.count_nonzero(mask == 0) == 0:
        roi_vals = im[mask == 1]    # 손가락 부분
        mean_val = np.mean(roi_vals)
        std_val  = np.std(roi_vals)
    else:
        bg_vals  = im[mask == 0]    # 배경 부분
        mean_val = np.mean(bg_vals)
        std_val  = np.std(bg_vals)

    # std가 0이면 분모를 1로
    std_val = std_val if std_val > 1e-6 else 1.0

    norm_img = (im - mean_val) / std_val
    norm_img *= mask                     # ROI 밖은 다시 0으로

    return segmented_image, norm_img, mask
