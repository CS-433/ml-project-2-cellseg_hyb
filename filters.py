import skimage.filters as skf
import numpy as np
from scipy.ndimage.filters import generic_filter

def identity(im):
    return im

def thresholding(im,thresh):
    return im > thresh

class Gaussian_Filter():

    def __init__(self,sigma):
        self.sigma = sigma 

    def __call__(self,im):
        return skf.gaussian(im,self.sigma)

class Median_Filter():

    def __init__(self,footprint):
        self.footprint = footprint

    def __call__(self,im):
        return skf.median(im,self.footprint)

class Gabor_Filter():

    def __init__(self,frequency,theta):
        self.frequency = frequency
        self.theta = theta 

    def __call__(self,im):
        real, im = skf.gabor(im,self.frequency,self.theta)
        return np.sqrt(real**2 + im**2)

class StdDev_Filter():

    def __init__(self,neighbourhood):
        self.neighbourhood = neighbourhood

    def __call__(self,im):
        return generic_filter(im,np.std,size=self.neighbourhood)



    