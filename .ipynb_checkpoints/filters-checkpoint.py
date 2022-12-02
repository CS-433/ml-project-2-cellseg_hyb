import skimage.filters as skf
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from tqdm import tqdm

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



    