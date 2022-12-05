import matplotlib.pyplot as plt
import skimage.io as skio
import skimage

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