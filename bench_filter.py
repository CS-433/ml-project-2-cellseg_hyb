import skimage.filters as skf
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from tqdm import tqdm
from filters import *

class Bench_Filter():
    """
    A classifier working with the following pipeline :
    Filters -> Random Forest
    See report for more explanations.
    """

    def __init__(self, n_estimators, cpus):
        """
        Initialize Random Forest and variables of the object.

        Args :
            - n_estimators : int. The number of trees in the forest.
            - cpus : int. The number of cpus to use to train the random forest.
        """
        self.nb_filters = 0
        self.feature_names = []
        self.model = RandomForestClassifier(n_estimators=n_estimators,verbose=2,n_jobs=cpus) 
        self.filters = []
        self.trained = False

    def add_identity(self):
        """
        Add identity filter on the bench.
        """
        self.filters.append(identity)
        self.nb_filters += 1
        self.feature_names.append('Identity')

    def add_gaussian(self, sigma):
        """
        Add gaussian filter on the bench.

        Args :
            - sigma : float. Standard deviation for Gaussian kernel.
        """
        self.filters.append(Gaussian_Filter(sigma))
        self.nb_filters += 1
        self.feature_names.append(f'Gaussian sig={sigma:.2f}')

    def add_sobel(self):
        """
        Add sobel filter on the bench.
        """
        self.filters.append(skf.sobel)
        self.nb_filters += 1
        self.feature_names.append('Sobel')

    def add_prewitt(self):
        """
        Add prewitt filter on the bench.
        """
        self.filters.append(skf.prewitt)
        self.nb_filters += 1
        self.feature_names.append('Prewitt')

    def add_roberts(self):
        """
        Add roberts filter on the bench.
        """
        self.filters.append(skf.roberts)
        self.nb_filters += 1
        self.feature_names.append('Roberts')

    def add_scharr(self):
        """
        Add scharr filter on the bench.
        """
        self.filters.append(skf.scharr)
        self.nb_filters += 1
        self.feature_names.append('Scharr')

    def add_farid(self):
        """
        Add farid filter on the bench.
        """
        self.filters.append(skf.farid)
        self.nb_filters += 1
        self.feature_names.append('Farid')
    
    def add_median(self,footprint=None):
        """
        Add median filter on the bench.

        Args :
            - footprint : ndarray. If behavior=='rank', footprint is a 2-D array of 1’s and 0’s. 
            If behavior=='ndimage', footprint is a N-D array of 1’s and 0’s with the same number of dimension than image. 
            If None, footprint will be a N-D array with 3 elements for each dimension (e.g., vector, square, cube, etc.)
        """
        self.filters.append(Median_Filter(footprint))
        self.nb_filters +=1
        self.feature_names.append('Median')

    def add_gabor(self,frequency,theta):
        """
        Add gabor filter on the bench.

        Args :
            frequency : float. Spatial frequency of the harmonic function. Specified in pixels.
            theta : float. Orientation in radians. If 0, the harmonic is in the x-direction.
        """
        self.filters.append(Gabor_Filter(frequency,theta))
        self.nb_filters += 1
        self.feature_names.append(f'Gabor f,t={frequency:.2f},{theta:.2f}')

    def add_stddev(self,neighbourhood):
        """
        Add gabor filter on the bench.

        Args :
            neighbourhood : int. Gives the shape that is taken from the input array.
        """
        self.filters.append(StdDev_Filter(neighbourhood))
        self.nb_filters +=1
        self.feature_names.append(f'Std Dev neigh={neighbourhood}')

    def apply_filter_and_ravel(self, X, y=None):
        """
        Apply filter for each image of the X dataset and ravel them, ravel targets to if provided in y.

        Args :
            - X : list. List of 2D images
            - y : list. List of 2D images, groundtruth segmentation of X.

        Returns :
            - X_filter : 2D array. Lines corresponds to samples and columns to features of each filters.
            - y_ravel : 1D array. Groundtruth of each pixel of X_filter.
        """
        nb_pixels = X[0].shape[0] * X[0].shape[1]
        X_filter = np.zeros((len(X) * nb_pixels, self.nb_filters))

        for i, image in tqdm(enumerate(X),desc="Applying filters to dataset") :

            out_filters = []
            for filter in self.filters :
                out_filters.append(np.ravel(filter(image)))

            X_filter[i * nb_pixels : (i+1) * nb_pixels, :] = np.vstack(out_filters).T 
        
        y_ravel = None
        if y :
            y_ravel = np.zeros(len(X) * nb_pixels)
            for i, target in enumerate(y):
                y_ravel[i * nb_pixels : (i+1) * nb_pixels] = np.ravel(target)

        return X_filter, y_ravel

    def fit(self, X_train, y_train):
        """
        Fit the classifier.

        Args :
            - X_train : list. List of 2D images.
            - y_train : list. List of 2D images, groundtruth segmentation of X.
        """

        X_train_filtered, y_train_ravel = self.apply_filter_and_ravel(X_train, y_train)
        
        self.model.fit(X_train_filtered,y_train_ravel)
        self.trained = True

    def predict(self,X_test):
        """
        Gives predictions for testing inputs.

        Args :
            - X_test : list. List of 2D images.

        Returns :
            y_pred_ravel : 3D array (nb_images, H, W)
        """
        if not self.trained :
            raise Exception("Please train the model before trying to predict something.")

        X_test_filtered, _ = self.apply_filter_and_ravel(X_test)
        
        y_pred_ravel = self.model.predict(X_test_filtered)
        return y_pred_ravel.reshape(len(X_test), X_test[0].shape[0], X_test[0].shape[1])

    def predict_proba(self,X_test,threshold=0.5):
        """
        Gives probability predictions for testing inputs.

        Args :
            - X_test : list. List of 2D images.
            - threshold : float between ]0,1[. To threshold probability.

        Returns :
            y_pred_ravel : 3D array (nb_images, H, W)
        """
        if not self.trained :
            raise Exception("Please train the model before trying to predict something.")

        X_test_filtered, _ = self.apply_filter_and_ravel(X_test)
        
        y_pred_ravel = self.model.predict_proba(X_test_filtered)
        return (y_pred_ravel[:,1] > threshold).reshape(len(X_test), X_test[0].shape[0], X_test[0].shape[1])

    def reset_train(self):
        """
        Reset training.
        """
        self.trained = False