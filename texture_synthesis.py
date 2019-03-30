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

IMAGE_DIM_HEIGHT = 200
IMAGE_DIM_WIDTH = 200
MAX_PIXEL_VALUE = 255
SEED_SIZE = 3
WINDOW_SIZE = 21
GAUSS_SIGMA = 1.4
ERR_THRESHOLD = 0.3
MAX_ERR_THRESHOLD = 0.1
TOTAL_PIXELS_IN_WINDOW = (WINDOW_SIZE)*(WINDOW_SIZE)

#print(TOTAL_PIXELS_IN_WINDOW)


def find_matches(template, sample_image, valid_mask, gaussian_mask):
    #print("gaussian_mask")
    #print(gaussian_mask)
    #print("valid mask")
    #print(valid_mask)
    total_weight = np.sum(np.multiply(gaussian_mask, valid_mask))
    SSD = []
    center_pixel = []
    sample_image_row, sample_image_col = sample_image.shape
    pad_size = mt.floor(WINDOW_SIZE/2)
    #print("hi")
    for i in range(pad_size, sample_image_row - pad_size - 1):
        for j in range(pad_size, sample_image_col - pad_size - 1):
            row_min = i - pad_size
            row_max = i + pad_size + 1
            col_min = j - pad_size
            col_max = j + pad_size + 1
            distance = (template - sample_image[row_min:row_max, col_min:col_max])**2
            #print("distance*gaussian_mask*valid_mask")
            #print(distance*gaussian_mask*valid_mask)
            temp = np.sum(distance*gaussian_mask*valid_mask)
            temp = temp/total_weight
            SSD.append(temp)
            center_pixel.append(sample_image[i, j])
    
    #print("SSD")
    #print(SSD)
    min_err = min(SSD)
    #print("min_err")
    #print(min_err)
    best_match = []
        
    for i in range(len(SSD)):
        if SSD[i] <= min_err*(1 + ERR_THRESHOLD):
            best_match.append((SSD[i], center_pixel[i]))
    #print("hello")
    #print("best_match len")
    #print(len(best_match))
    return best_match
'''
def find_matches(template, src_image, valid_mask, gauss_mask):
    total_weight = np.sum(np.multiply(gauss_mask, valid_mask))
    pad_size = mt.floor(WINDOW_SIZE/2)
    if total_weight == 0:
        print("Trouble")

    ssd = []
    pixel_val = []
    src_row, src_col = np.shape(src_image)
    for i in range(pad_size, src_row-pad_size-1):
        for j in range(pad_size, src_col-pad_size-1):
            distance = (template - src_image[i-pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1]) ** 2
            ssd.append(np.sum(distance * gauss_mask * valid_mask) / total_weight)
            pixel_val.append(src_image[i, j])

    min_error = min(ssd)
    # print "Min Err loop= "+str(min_error)
    rPack=[]

    for i in range(len(ssd)):
        if ssd[i]<=min_error*(1+ERR_THRESHOLD):
            rPack.append((ssd[i], pixel_val[i]))

    # print len(rPack)
    if len(rPack)==0:
        print(template)
    return rPack 
''' 
def find_matches_modified(template,image_window, valid_mask, gauss_mask, centre_pixels):
    pad_size = mt.floor(WINDOW_SIZE/2)
    template = np.reshape(template, TOTAL_PIXELS_IN_WINDOW)
    gauss_mask = np.reshape(gauss_mask, TOTAL_PIXELS_IN_WINDOW)
    valid_mask = np.reshape(valid_mask, TOTAL_PIXELS_IN_WINDOW)
    total_weight = np.sum(np.multiply(gauss_mask, valid_mask))
    distance = (image_window-template)**2
    ssd = np.sum((distance*gauss_mask*valid_mask) / total_weight, axis=1)
    min_error = min(ssd)
    # print "Min err mat= "+str(min_error)
    mid = ((2 * pad_size + 1)*(2 * pad_size + 1)) / 2;
    best_matches = []
    for i in range(len(ssd)):
        if ssd[i]<=min_error*(1+ERR_THRESHOLD):
            best_matches.append((ssd[i], centre_pixels[i]))

    #return [[err, image_window[i*WINDOW_SIZE +  (mid - 1)]] for i, err in enumerate(ssd) if err <= min_error*(1+ERR_THRESHOLD)]
    return best_matches


def gaussian2D(window_size):
    m,n = [(ss-1.)/2. for ss in window_size]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*GAUSS_SIGMA*GAUSS_SIGMA) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def extract_all_frames(sample_image):
    pad_size = mt.floor(WINDOW_SIZE/2)
    possible_frames = []
    
    sample_image_row, sample_image_col = sample_image.shape
    # iterate over the entire sample image array and extratc all possible windows
    for i in range(pad_size, sample_image_row - pad_size - 1):
        for j in range(pad_size, sample_image_col - pad_size - 1):
            possible_frames.append(np.reshape(sample_image[i-pad_size:i + pad_size + 1, j - pad_size: j + pad_size + 1], (2 * pad_size + 1) ** 2))
    return np.double(possible_frames)

def extract_all_frames_centers(sample_image):
    pad_size = mt.floor(WINDOW_SIZE/2)
    possible_frames = []
    centers = []
    
    sample_image_row, sample_image_col = sample_image.shape
    # iterate over the entire sample image array and extratc all possible windows
    for i in range(pad_size, sample_image_row - pad_size - 1):
        for j in range(pad_size, sample_image_col - pad_size - 1):
            possible_frames.append(np.reshape(sample_image[i-pad_size:i + pad_size + 1, j - pad_size: j + pad_size + 1], (2 * pad_size + 1) ** 2))
            centers.append(sample_image[i, j])
    return np.double(possible_frames), np.double(centers)
    
#print("reading image")
sample_image = imread("T5.gif")
#plt.imshow(sample_image)

#plt.imshow(sample_image, cmap = "gray")
#plt.gcf().clear()
#print("printing image dimension")
sample_image_row, sample_image_col = sample_image.shape


#print("normalizing image by dividing with max pixel value")
sample_image_normalized = sample_image/MAX_PIXEL_VALUE
#sample_image_normalized = sample_image/1
#print(sample_image_normalized)
#print("printing modified image dimension")
#print(sample_image_normalized.shape)
#img1 = io.imread("T2.gif")
#plt.imshow(sample_image_normalized, cmap = "gray")
#plt.gcf().clear()
#print(img1.format)
total_pixels = IMAGE_DIM_HEIGHT*IMAGE_DIM_WIDTH
#print("total pixels to be filled")
#print(total_pixels)

# image will contain the final synthesized image
#print("Initializing the empty image consisting of all 0s")
image = np.zeros((IMAGE_DIM_HEIGHT, IMAGE_DIM_WIDTH))


#print("Picking up a random 3x3 square patch from sample image")
rand_row = rd.randint(0, sample_image_row - SEED_SIZE)
rand_col = rd.randint(0, sample_image_col - SEED_SIZE)

#print("Creating the random seed from the original image")
seed = sample_image_normalized[rand_row: rand_row + SEED_SIZE, rand_col: rand_col + SEED_SIZE]
#plt.imshow(seed, cmap = "gray")
#plt.imshow(seed)

#plt.gcf().clear()  

#print("Pasting the 3x3  square seed in center of this almost empty image from the sample image")
image[mt.floor(IMAGE_DIM_HEIGHT/2) - 1: mt.floor(IMAGE_DIM_HEIGHT/2) + 2, \
      mt.floor(IMAGE_DIM_WIDTH/2) - 1: mt.floor(IMAGE_DIM_WIDTH/2) + 2] \
      = seed

#print("pasting a 3x3 square patch of 1s in the filled_list" )
filled_pixels = SEED_SIZE*SEED_SIZE
#print("filled pixels")
#print(filled_pixels)


# filled_list keeps track of the pixels already filled. We use it to extract the neighborhood each time
filled_list = np.zeros((IMAGE_DIM_HEIGHT, IMAGE_DIM_WIDTH))
#print("Fixing the 3x3  square seed in center of this almost empty image")
filled_list[mt.floor(IMAGE_DIM_HEIGHT/2) - 1: mt.floor(IMAGE_DIM_HEIGHT/2) + 2, \
      mt.floor(IMAGE_DIM_WIDTH/2) - 1: mt.floor(IMAGE_DIM_WIDTH/2) + 2] \
      = np.ones((SEED_SIZE, SEED_SIZE))

    #size of padding is half of the window size as we need to pad all four sides of the image matrix
pad_size = mt.floor(WINDOW_SIZE/2)
# we need to zero pad both the sample image and image to take care of the pixels at the borders
filled_list_padded = np.lib.pad(filled_list, pad_size, 'constant', constant_values = 0)
image_padded = np.lib.pad(image, pad_size, 'constant', constant_values = 0) 
    
max_error_threshold = MAX_ERR_THRESHOLD
#print("hello")
possible_frames, possible_frames_centers =  extract_all_frames_centers(sample_image_normalized)
#possible_frames =  extract_all_frames(sample_image_normalized)
#print("hi")

gaussian_mask = gaussian2D((WINDOW_SIZE, WINDOW_SIZE))
#print("gaussian_mask")
#print(gaussian_mask)
#print(gaussian_mask.shape)
#plt.imshow(image, cmap = "gray")

while filled_pixels < total_pixels:
#while filled_pixels < 10:
    #print("filled pixels")
    #print(filled_pixels)
    #plt.imshow(seed)
    progress = 0
    potential_pixel_row = []
    potential_pixel_col = []
    filled_list_neighbors =  morphology.binary_dilation(filled_list)
    potential_pixel_row, potential_pixel_col = np.nonzero(filled_list_neighbors - filled_list)
    
    #print("potential_pixel_row")
    #print(potential_pixel_row)
    #print("potential_pixel_col")
    #print(potential_pixel_col)
    
    #print("building the actual neighbors by picking a pixel from potential pixels" )
    filled_neighbors = []
    
    for i in range(len(potential_pixel_row)):
        pixel_row = potential_pixel_row[i]
        pixel_col = potential_pixel_col[i]
        #print(i)
        #print("the neighborhood consists of window size of pixels with the specific pixel at the center")
        row_min = pixel_row - mt.floor(WINDOW_SIZE/2)
        row_max = pixel_row + mt.floor(WINDOW_SIZE/2) + 1
        col_min = pixel_col - mt.floor(WINDOW_SIZE/2)
        col_max = pixel_col + mt.floor(WINDOW_SIZE/2) + 1
        
        #print("For each potential pixel we check how many pixels are already filled in its window")
        #print("this is done by counting the number of 1s in its window")
        filled_neighbors.append(np.sum(filled_list[row_min:row_max, col_min:col_max]))
        
        # sorting this filled_neighbors in descending order i.e. the first argument gives the pixel
        # for which number of filled neighbors in the windows i maximum
        # this pixel is picked up to be grown because we want to minimize the loss in guessing neighborhood
        
    descending_filled_num = (-1) * np.array(filled_neighbors, dtype = "int")
    #print(filled_neighbors)
    #print(descending_filled_num)
    #print("Number of neighbors")
    #print(len(descending_filled_num))
    #we need the indices where the number of neighbors filled is maximum
    descending_filled_num_indices = np.argsort(descending_filled_num)
    print(descending_filled_num_indices)
        
    #we need to iterate over the sorted list (key, value) pair like and pick the list elements
    #print("outside")
    for x, i in enumerate(descending_filled_num_indices):
        print("x")
        print(x)
        print("i")
        print(i)
        # get the row, column  of the selected pixel 
        #use this to calculate the row_min, row_max in padded image
        sel_pix_row = potential_pixel_row[i]
        sel_pix_col = potential_pixel_col[i]
        
        # calculating the min and max of both row and columns of the region to be synthesized
        # because of the padding the nth index in original matrix is (n+pad_size) index in padded matrix
        padded_row_min = pad_size + potential_pixel_row[i] - pad_size
        padded_row_max = pad_size + potential_pixel_row[i] + pad_size + 1
        padded_col_min = pad_size + potential_pixel_col[i] - pad_size
        padded_col_max = pad_size + potential_pixel_col[i] + pad_size + 1
        
        best_matches  = find_matches_modified(image_padded[padded_row_min: padded_row_max, padded_col_min: padded_col_max], \
                                     possible_frames,filled_list_padded[padded_row_min: padded_row_max, padded_col_min: padded_col_max],\
                                     gaussian_mask, possible_frames_centers) 
        '''
        best_matches  = find_matches(image_padded[padded_row_min: padded_row_max, padded_col_min: padded_col_max], \
                                     possible_frames,filled_list_padded[padded_row_min: padded_row_max, padded_col_min: padded_col_max],\
                                     gaussian_mask)
        '''
        #print(len(descending_filled_num_indices))
        #print("here")
        #print(x)
        
        #print("len(best_matches)")
        #print(len(best_matches))
        
        random_match = rd.randint(0, len(best_matches) - 1)
        #print("random_match")
        #print(random_match)
        if best_matches[random_match][0] <= MAX_ERR_THRESHOLD:
            image_padded[pad_size + sel_pix_row][pad_size +  sel_pix_col] = best_matches[random_match][1]
            image[sel_pix_row][sel_pix_col] = best_matches[random_match][1]
            filled_list_padded[pad_size + sel_pix_row][pad_size +  sel_pix_col] = 1
            filled_list[sel_pix_row][sel_pix_col]=1
            
            filled_pixels = filled_pixels + 1
            progress = 1
            #print("inside")
            #plt.imshow(image)
    if progress == 0:
        max_error_threshold = max_error_threshold * 1.1
                
io.imsave("t5_win13_sigma0.8.jpeg", image)
#image = image * MAX_PIXEL_VALUE
#plt.imshow(image, cmap = "gray")
#plt.imshow("t1_new.gif", cmap = "gray")
plt.imshow(image)
