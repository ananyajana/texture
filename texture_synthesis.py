#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 16:06:21 2019

@author: aj611
"""


"""
def GrowImage(SampleImage, Image, WindowSize):
    while Image not filled to:
        progress = 0
        PixelList = GetUnfilledNeighbors(Image)
        for each pixel in PixelList do:
            Template = GetNeighborhoodWindow(Pixel)
            BestMatches = findMatches(Template, SampleImage)
            BestMatch = RandomPick(BestMatches)
            if(BestMatch.error < MaxErrThreoshold) then
                Pixel.value = BestMatch.value
                progress = 1
            end
        end
        if progress == 0:
            then MaxErrthreshold = MaxErrthreshold * 1.1
        end
    return Image
end

def FindMatches(Template, SampleImage):
    ValidMask = 1s where Template is filles, 0s otherwise
    GaussMask = Gaussian2D(WindowsSize, Sigma)
    TotWeight = sum i, j GaussMask(i, j) * ValidMask(i, j)
    for i, j do:
        for ii, jj do:
            dist = (Template(ii, jj) - SampleImage(i - ii, j - jj))^2
            SSD(i, j) = SSD(i, j) + dist*ValidMask(ii, jj)*GaussMask(ii, jj)
        end
        SSD(i, j) = SSD(i, j) /TotWeight
    end
    PixelList - allpixels (i, j) where SSD(i, j) <= min(SSD)*(1 + ErrThreshold)
    return PixelList
end

"""

import numpy as np
from skimage import io, morphology
import random as rd
import math as mt
from matplotlib.image import imread
import matplotlib.pyplot as plt
import time

IMAGE_DIM_HEIGHT = 200
IMAGE_DIM_WIDTH = 200
MAX_PIXEL_VALUE = 255
SEED_SIZE = 3
WINDOW_SIZE = 5


print("reading image")
sample_image = imread("T1.gif")
plt.imshow(sample_image, cmap = "gray")
plt.gcf().clear()
print("printing image dimension")
sample_image_row, sample_image_col = sample_image.shape


print("normalizing image by dividing with max pixel value")
sample_image_normalized = sample_image/MAX_PIXEL_VALUE
print(sample_image_normalized)
print("printing modified image dimension")
print(sample_image_normalized.shape)
#img1 = io.imread("T2.gif")
plt.imshow(sample_image_normalized, cmap = "gray")
plt.gcf().clear()
#print(img1.format)
total_pixels = IMAGE_DIM_HEIGHT*IMAGE_DIM_WIDTH
print("total pixels to be filled")
print(total_pixels)

print("Initializing the empty image consisting of all 0s")
image = np.zeros((IMAGE_DIM_HEIGHT, IMAGE_DIM_WIDTH))


print("Picking up a random 3x3 square patch from sample image")
rand_row = rd.randint(0, sample_image_row - SEED_SIZE)
rand_col = rd.randint(0, sample_image_col - SEED_SIZE)



print("Creating the random seed from the original image")
seed = sample_image_normalized[rand_row: rand_row + SEED_SIZE, rand_col: rand_col + SEED_SIZE]
plt.imshow(seed, cmap = "gray")
#plt.gcf().clear()  

print("Pasting the 3x3  square seed in center of this almost empty image from the sample image")
image[mt.floor(IMAGE_DIM_HEIGHT/2) - 1: mt.floor(IMAGE_DIM_HEIGHT/2) + 2, \
      mt.floor(IMAGE_DIM_WIDTH/2) - 1: mt.floor(IMAGE_DIM_WIDTH/2) + 2] \
      = seed

print("pasting a 3x3 square patch of 1s in the filled_list" )
filled_pixels = SEED_SIZE*SEED_SIZE
print("filled pixels")
print(filled_pixels)


# filled_list keeps track of the pixels already filled. We use it to extract the neighborhood each time
filled_list = np.zeros((IMAGE_DIM_HEIGHT, IMAGE_DIM_WIDTH))
print("Fixing the 3x3  square seed in center of this almost empty image")
filled_list[mt.floor(IMAGE_DIM_HEIGHT/2) - 1: mt.floor(IMAGE_DIM_HEIGHT/2) + 2, \
      mt.floor(IMAGE_DIM_WIDTH/2) - 1: mt.floor(IMAGE_DIM_WIDTH/2) + 2] \
      = np.ones((SEED_SIZE, SEED_SIZE))

filled_list_neighbors =  morphology.binary_dilation(filled_list)     
potential_pixel_row, potential_pixel_col = np.nonzero(filled_list_neighbors - filled_list)

print("potential_pixel_row")
print(potential_pixel_row)
print("potential_pixel_col")
print(potential_pixel_col)

print("building the actual neighbors by picking a pixel from potential pixels" )
filled_neighbors = []

for i in range(len(potential_pixel_row)):
    pixel_row = potential_pixel_row[i]
    pixel_col = potential_pixel_col[i]
    print(i)
    print("the neighborhood consists of window size of pixels with the specific pixel at the center")
    row_min = pixel_row - mt.floor(WINDOW_SIZE/2)
    row_max = pixel_row + mt.floor(WINDOW_SIZE/2) + 1
    col_min = pixel_col - mt.floor(WINDOW_SIZE/2)
    col_max = pixel_col + mt.floor(WINDOW_SIZE/2) + 1
    
    print("For each potential pixel we check how many pixels are already filled in its window")
    #print("this is done by counting the number of 1s in its window")
    filled_neighbors.append(np.sum(filled_list[row_min:row_max, col_min:col_max]))
    
    # sorting this filled_neighbors in descending order i.e. the first argument gives the pixel
    # for which number of filled neighbors in the windows i maximum
    # this pixel is picked up to be grown because we want to minimize the loss in guessing neighborhood
    
    descending_filled_num = (-1) * np.array(filled_neighbors, dtype = "int")
    #print(filled_neighbors)
    print(descending_filled_num)
    descending_filled_num_indices = np.argsort(descending_filled_num)
    print(descending_filled_num_indices)
#print(filled_neighbors)