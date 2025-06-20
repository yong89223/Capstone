
import numpy as np
import cv2 as cv
from utils.crossing_number import calculate_minutiaes
from skimage.morphology import skeletonize as skelt
from skimage.morphology import thin
def skeletonize(image_input):
    
    image = np.zeros_like(image_input)
    image[image_input == 0] = 1.0
    output = np.zeros_like(image_input)

    skeleton = skelt(image)

    """uncomment for testing"""
    # thinned = thin(image)
    # thinned_partial = thin(image, max_iter=25)
    #
    # def minu_(skeleton, name):
    #     cv.imshow('thin_'+name, output)
    #     cv.bitwise_not(output, output)
    #     minutias = calculate_minutiaes(output, kernel_size=5); cv.imshow('minu_'+name, minutias)
    # # minu_(output, 'skeleton')
    # # minu_(output, 'thinned')
    # # minu_(output, 'thinned_partial')
    # # cv.waitKeyEx()

    output[skeleton] = 255
    cv.bitwise_not(output, output)

    return output


def thinning_morph(image, kernel):
    """
    Thinning image using morphological operations
    :param image: 2d array uint8
    :param kernel: 3x3 2d array unint8
    :return: thin images
    """
    thining_image = np.zeros_like(image)
    img = image.copy()

    while 1:
        erosion = cv.erode(img, kernel, iterations = 1)
        dilatate = cv.dilate(erosion, kernel, iterations = 1)

        subs_img = np.subtract(img, dilatate)
        cv.bitwise_or(thining_image, subs_img, thining_image)
        img = erosion.copy()

        done = (np.sum(img) == 0)

        if done:
          break

    # shift down and compare one pixel offset
    down = np.zeros_like(thining_image)
    down[1:-1, :] = thining_image[0:-2, ]
    down_mask = np.subtract(down, thining_image)
    down_mask[0:-2, :] = down_mask[1:-1, ]
    cv.imshow('down', down_mask)

    # shift right and compare one pixel offset
    left = np.zeros_like(thining_image)
    left[:, 1:-1] = thining_image[:, 0:-2]
    left_mask = np.subtract(left, thining_image)
    left_mask[:, 0:-2] = left_mask[:, 1:-1]
    cv.imshow('left', left_mask)

    # combine left and down mask
    cv.bitwise_or(down_mask, down_mask, thining_image)
    output = np.zeros_like(thining_image)
    output[thining_image < 250] = 255

    return output
