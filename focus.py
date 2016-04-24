import numpy as np
import cv2
try:
    from assignment8 import *
except:
    print("assignment8.py not important, ability to fine adjust images not present")
import os


def adjust_images(image_dir, save = True):
    '''
    Loads an image directory, finds the most focused image, and puts all images in that images space
    '''
    img_array = image_array(image_dir)
    n,h,w = img_array.shape
    adjusted_array = np.empty((n,h,w))
    focus_vec = focus_set(img_array)
    master = np.argmax(focus_vec)
    
    for i in range(n):
        print i
        if i == master:
            cv2.imwrite(os.path.join(image_dir, 'A_{}.png'.format(i)), img_array[master,:,:])
            continue
        else:
            image_1_kp, image_2_kp, matches = findMatchesBetweenImages(img_array[i,:,:], img_array[master,:,:], 50)
            H = findHomography(image_1_kp, image_2_kp, matches)
            #adjusted_array[i,:,:] = warpImagePair(img_array[i,:,:], img_array[master,:,:], H)
            adjusted_array = warpImagePair(img_array[i,:,:], img_array[master,:,:], H)
            
            
            if save == True:
                #cv2.imwrite(os.path.join(image_dir, 'A_{}.png'.format(i)), adjusted_array[i,:,:])
                cv2.imwrite(os.path.join(image_dir, 'A_{}.png'.format(i)), adjusted_array)
            
    return adjusted_array
    
def global_focus(img):
    '''
    returns focus metric of image based on squared gradient
        input: grayscale
        return: float. representing how in focus the entire image is
    '''
    h,w = img.shape
    return np.sum((img[0:h-2,:] - img[2:h,:])**2)/(1.0*h*w)

def local_focus(img):
    Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 3)
    Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = 3)
    return Ix**2 + Iy**2
 
def image_array(image_dir):
    '''
    generate an image array of Nxhxw of grayscale images from directory, sort on fname
    '''
    flist = os.listdir(image_dir)
    flist = sorted(flist, key = lambda x: x.split('_')[1])
    img1 = cv2.cvtColor(cv2.imread(os.path.join(image_dir, flist[0])), cv2.COLOR_BGR2GRAY)
    h,w = img1.shape
    img_array = np.empty((len(flist), h, w), dtype = np.uint8)
    for i in range(len(flist)):
        img_array[i,:,:] = cv2.cvtColor(cv2.imread(os.path.join(image_dir, flist[i])), cv2.COLOR_BGR2GRAY)
        print flist[i]
    return img_array
    
def focus_set(img_array):
    '''
    determine global focus for each image 
    '''
    N,h,w = img_array.shape
    f_vec = np.empty((N))
    for i in range(N):
        f_vec[i] = global_focus(img_array[i])
    return f_vec
                
def test_run():
    '''
    image_dir = 'photos/set_1'
    flist = os.listdir(image_dir)
    flist = sorted(flist, key = lambda x: x.split('_')[1])
    im_list = []
    for i in range(len(flist)):
        im_list.append( cv2.cvtColor(cv2.imread(os.path.join(image_dir, flist[i])), cv2.COLOR_BGR2GRAY))
        cv2.imwrite('output/{}_focus.png'.format(i), local_focus(im_list[i]))
    '''
    image_dir = 'photos/set_1'
    adjust_images(image_dir)   
if __name__ == "__main__":
    test_run()
    print("main")