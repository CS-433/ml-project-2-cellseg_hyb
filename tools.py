import matplotlib.pyplot as plt
import skimage.io as skio
import skimage
import numpy as np
import pandas as pd

def load_img_tg(IM_PATH, TG_PATH):
    """
        Load images and targets from two directories where they are stored.

        Args :
            - IM_PATH : list. List of paths of images.
            - TG_PATH : list. List of paths of targets.

        Return :
            - img : list. List of 2D array of each images.
            - tg : list. List of 2D array of associated targets.
    """
    img, tg = [], []
    for im_path, tg_path in zip(IM_PATH,TG_PATH) :
        im = skio.imread(im_path,plugin='pil')
        im = skimage.img_as_float(im)
        im = (im -im.min())/(im.max() - im.min() + 1e-6)
        img.append(im)
        tg.append(skio.imread(tg_path,plugin='pil') > 0)
    return img, tg


def plot_pred_with_target(target, seg, score):
    """
        Plot predictions with their targets to compare in a 2x2 subplot :

           TARGET                     PREDICTION

        FALSE NEGATIVE              FALSE POSITIVE

        Args :
            - target : 2D array. Target.
            - seg : 2D array. Prediction of a model.
            - score : float. Segmentation score of the prediction.

        Return :
            - fig : matplotlib object. Figure.
    """
    fig, ax = plt.subplots(2,2,figsize=(15,10))
    ax[0,0].imshow(target,cmap='gray')
    ax[0,0].set_title('Target')
    ax[0,1].imshow(seg,cmap='gray')
    ax[0,1].set_title(f'Score prediction : {score:.2f}') 
    ax[1,0].imshow(target & ~seg,cmap='gray')
    ax[1,0].set_title('False Negative')
    ax[1,1].imshow(~target & seg,cmap='gray') 
    ax[1,1].set_title('False Positive')
    plt.tight_layout()
    plt.show()
    return fig
    

def crop_img_tg(list_IM, list_TG, ratio):
    """
        Crop images and target on the area where the target is the most present.

        Args :
            - list_IM : list. List of 2D array of images.
            - list_TG : list. List of 2D array of targets.
            - ratio : number to divide width and height. Has to be a common divisor of them.

        Return :
            - list_IM_cropped : list. List of 2D array of images cropped.
            - list_TG_cropped : list. List of 2D array of targets cropped.
    """
    divisors = common_divisors(list_IM[0].shape[0],list_IM[0].shape[1])
    if ratio not in divisors :
        raise ValueError(f"Please use a ratio that divide height and width. Available ratio : {divisors}")

    list_IM_cropped, list_TG_cropped = [], []
    #dimension of patchs = M*N
    M = list_IM[0].shape[0]//ratio
    N = list_IM[0].shape[1]//ratio
    print(f'Initial dimensions of images : ({list_IM[0].shape[0]},{list_IM[0].shape[1]})')
    print(f'Patch dimensions : ({M},{N})')

    for k in range(len(list_IM)):
        im = list_IM[k]
        tg = list_TG[k]
        im_tiles = [im[x:x+M, y:y+N] for x in range(0, im.shape[0], M) for y in range(0, im.shape[1], N)]
        tg_tiles = [tg[x:x+M, y:y+N] for x in range(0, im.shape[0], M) for y in range(0, im.shape[1], N)]
        scores = []

        for i in range(len(tg_tiles)):
            scores.append(np.sum(tg_tiles[i]))

        idx_best_score = np.argmax(scores) #Find the tile where the target has the most 1s.
        list_IM_cropped.append(im_tiles[idx_best_score])
        list_TG_cropped.append(tg_tiles[idx_best_score])

    return list_IM_cropped, list_TG_cropped


def common_divisors(num1, num2):
    """
        Return a list of common divisors of two numbers.

        Args :
            - num1 : int. 
            - num2 : int.

        Return :
            - divs : list. List of common divisors of num1 & num2.
    """
    divs1 = []
    for x in range (1, num1+1):
        if (num1 % x) == 0:
            divs1.append(x)
    
    divs = []
    for div in divs1 :
        if (num2 % div) == 0:
            divs.append(div)

    return divs


def RF_feature_importances(bench_model,fig_name=None):
    """
        Plot feature importances of a random forest in a bench filter model (see report).

        Args :
            - bench_model : bench filter model.
            - fig_name : str. If not None, will save the figure with this name.
    """
    importances = bench_model.model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in bench_model.model.estimators_], axis=0)

    forest_importances = pd.Series(importances, index=bench_model.feature_names)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    if fig_name :
        fig.savefig(f'output/feature_importance_{fig_name}.png')
    plt.show()
    