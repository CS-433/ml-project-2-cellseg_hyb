import skimage.filters as skf
import numpy as np
from scipy.ndimage.filters import generic_filter

def identity(im):
    """
        Identity function on an image.

        Args :
            - im : 2D array. The image to process.

        Return :
            - Image processed
    """
    return im

def thresholding(im,thresh):
    """
        Threshold an image.

        Args :
            - im : 2D array. The image to process.
            - thresh : float. Must be in [0,1].
        
        Return :
            - Image processed
    """
    assert 0 <= thresh <= 1, "Incorrect threshold. Choose something between 0 and 1."
    return im > thresh

class Gaussian_Filter():
    """
    Implement an object to parametrize a gaussian filter.
    """

    def __init__(self,sigma):
        """
        Initialize the filter with its parameters.

        Args :
            - sigma : float. Standard deviation for Gaussian kernel.
        """
        assert sigma > 0, "Incorrect sigma. Choose something higher than 0."
        self.sigma = sigma 

    def __call__(self,im):
        """
        Apply the filter on an image.

        Args :
            - im : 2D array. The image to process.

        Return :
            - Image processed
        """
        return skf.gaussian(im,self.sigma)

class Median_Filter():
    """
    Implement an object to parametrize a median filter.
    """

    def __init__(self,footprint):
        """
        Initialize the filter with its parameters.

        Args :
            - footprint : A 2-D array of 1 and 0.
            If None, it will be a N-D array with 3 elements for each dimension (e.g., vector, square, cube, etc.).
        """
        self.footprint = footprint

    def __call__(self,im):
        """
        Apply the filter on an image.

        Args :
            - im : 2D array. The image to process.

        Return :
            - Image processed
        """
        return skf.median(im,self.footprint)

class Gabor_Filter():
    """
    Implement an object to parametrize a gabor filter.
    """

    def __init__(self,frequency,theta):
        """
        Initialize the filter with its parameters.

        Args :
            - frequency : float. Spatial frequency of the harmonic function. Specified in pixels.
            - theta : float. Orientation in radians. If 0, the harmonic is in the x-direction.
        """
        self.frequency = frequency
        self.theta = theta 

    def __call__(self,im):
        """
        Apply the filter on an image.

        Args :
            - im : 2D array. The image to process.

        Return :
            - Image processed
        """
        real, im = skf.gabor(im,self.frequency,self.theta)
        return np.sqrt(real**2 + im**2)

class StdDev_Filter():
    """
    Implement an object to parametrize a standard deviation filter.
    """

    def __init__(self,neighbourhood):
        """
        Initialize the filter with its parameters.

        Args :
            - neighbourhood : scalar or tuple. Gives the shape that is taken from the input array, 
            at every element position, to define the input to the filter function.
        """
        self.neighbourhood = neighbourhood

    def __call__(self,im):
        """
        Apply the filter on an image.

        Args :
            - im : 2D array. The image to process.

        Return :
            - Image processed
        """
        return generic_filter(im,np.std,size=self.neighbourhood)



    