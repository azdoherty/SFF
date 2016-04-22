import numpy as np
import cv2
import os

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
    img_array = np.empty((len(flist), h, w))
    for i in range(len(flist)):
        img_array[i,:,:] = cv2.cvtColor(cv2.imread(os.path.join(image_dir, flist[i])), cv2.COLOR_BGR2GRAY)
        print flist[i]
    return img_array
        
def test_run():
    image_dir = 'photos/set_1'
    flist = os.listdir(image_dir)
    flist = sorted(flist, key = lambda x: x.split('_')[1])
    im_list = []
    for i in range(len(flist)):
        im_list.append( cv2.cvtColor(cv2.imread(os.path.join(image_dir, flist[i])), cv2.COLOR_BGR2GRAY))
        cv2.imwrite('output/{}_focus.png'.format(i), local_focus(im_list[i]))
        
if __name__ == "__main__":
    test_run()