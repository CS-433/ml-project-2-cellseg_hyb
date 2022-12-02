import matplotlib.pyplot as plt

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