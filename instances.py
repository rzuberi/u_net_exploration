import numpy as np
import cv2
from scipy import spatial

"""
This function returns an image with the same shape as the input, with
each pixel labelled with a number representing which instance the pixel
belongs to.

The function actually first calculates the centroid of each instance from
the given mask, and then labels each pixel in the mask with the number
of the centroid which is closest to the pixel.

Inputs:
    mask: A binary image.

Returns:
    An image with the same shape as the input, with each pixel labelled
    with a number representing which instance the pixel belongs to.
    0 is background.
"""
def get_instances(mask):

    assert(mask.shape() == (2,))
    assert(np.unique(mask) == [0,1])

    binary_mask_copy = np.uint8(np.squeeze(mask).copy()) # Correct format for openCV functions
    contours, hierarchy = cv2.findContours(binary_mask_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Findings centroids
    centers = []
    for i in range(len(contours)):
        M = cv2.moments(contours[i])
        centers.append([int(M["m10"] / M["m00"]),int(M["m01"] / M["m00"])])
    centers = np.array(centers.reverse())

    # Label each pixel in the mask with the number of the centroid
    # which is closest to the pixel.
    instance_mask = np.zeros((1080,1080))
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[j][i] == 1:
                instance_mask[j][i] = spatial.KDTree(centers).query([i,j])[1]

    return instance_mask