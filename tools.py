import matplotlib.pyplot as plt
import skimage.io as skio
import skimage
import numpy as np

def plot_pred_with_target(target, seg, score):
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
    
def load_img_tg(IM_PATH, TG_PATH):
    img, tg = [], []
    for im_path, tg_path in zip(IM_PATH,TG_PATH) :
        im = skio.imread(im_path,plugin='pil')
        im = skimage.img_as_float(im)
        im = (im -im.min())/(im.max() - im.min() + 1e-6)
        img.append(im)
        tg.append(skio.imread(tg_path,plugin='pil') > 0)
    return img, tg

def crop_img_tg(list_IM, list_TG, ratio):
    divisors = common_divisors(list_IM[0].shape[0],list_IM[0].shape[1])
    if ratio not in divisors :
        raise ValueError(f"Please use a ratio that divide height and width. Available ratio : {divisors}")

    X_train_cropped, y_train_cropped = [], []
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

        for i in range(len(im_tiles)):
            scores.append(np.sum(tg_tiles[i]))

        idx_best_score = np.argmax(scores)
        X_train_cropped.append(im_tiles[idx_best_score])
        y_train_cropped.append(tg_tiles[idx_best_score])

    return X_train_cropped, y_train_cropped

def common_divisors(num1, num2):
    divs1 = []
    for x in range (1, num1+1):
        if (num1 % x) == 0:
            divs1.append(x)
    
    divs = []
    for div in divs1 :
        if (num2 % div) == 0:
            divs.append(div)

    return divs
    