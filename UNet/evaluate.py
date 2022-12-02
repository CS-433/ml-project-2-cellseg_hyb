import torch
import torch.nn.functional as F
from torchmetrics.functional import jaccard_index
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff
from utils.IoU import iou_pytorch


def evaluate(net, dataloader, device, metric):
    assert (metric in ['IoU', 'Dice']),"Incorrect metric, choose in 'IoU' or 'Dice'."
    
    net.eval()
    num_val_batches = len(dataloader)
    score = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()
                # compute the metric
                if metric == 'IoU' :
                    score += jaccard_index(mask_pred,mask_true,task='binary') #iou_pytorch(mask_pred, mask_true) 
                else :
                    score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the metric, ignoring background
                if metric == 'IoU' :
                    score += jaccard_index(mask_pred,mask_true,task='multiclass') #iou_pytorch(mask_pred[:, 1:, ...], mask_true[:, 1:, ...]) 
                else :
                    score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)

           

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return score
    return score / num_val_batches
