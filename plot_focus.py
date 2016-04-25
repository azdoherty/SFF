import numpy as np
import cv2
import os
from focus import *
from matplotlib import pyplot as plt

def generate_blur_set(img, N):
    '''
    generate a blurred image set (N images) from a single in-focus image to test focus metrics.
        input - img: grayscale image array
        return - N - integer, number of blurred images to be generated
    '''
    h,w = img.shape
    blurred_array = np.zeros((N,h,w))
    for i in range(N):
        blurred_array[i,:,:] = cv2.blur(img, (2 * (i+1) - 1, 2 * (i+1) - 1))
    return blurred_array    
   
def plot_focus(f_vec, title = 'Focus'):
    ax = plt.figure()
    plt.plot(np.arange(len(f_vec)), f_vec, '.-')
    plt.title(title)
    plt.xlabel('image number')
    plt.ylabel('focus value')
    plt.savefig(title + ".png", dpi = 400)
    
def test_run(input_image):
    blur_array = generate_blur_set(input_image, 20)
    f_vec = focus_set(blur_array)
    plot_focus(f_vec)

def test_real(input_dir):
    blur_array = image_array(input_dir)
    f_vec = focus_set(blur_array)
    plot_focus(f_vec)

    
 
if __name__ == "__main__":
    test_image = 'photos\set_0\IMG_1229.JPG'
    img = cv2.cvtColor(cv2.imread(test_image), cv2.COLOR_BGR2GRAY)
    #h,w = img.shape
    #img = cv2.resize(img, dsize = (h/4, w/4))
    test_run(img)
    #test_real('photos/MVI_1227')


    