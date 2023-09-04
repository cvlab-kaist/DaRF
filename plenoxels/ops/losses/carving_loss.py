import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def compute_space_carving_loss(pred_depth, target_hypothesis, is_joint=False, mask=None, norm_p=2, threshold=0.0):
    n_rays, n_points = pred_depth.shape
    num_hypothesis = target_hypothesis.shape[0]

    if target_hypothesis.shape[-1] == 1:
        ### In the case where there is no caching of quantiles
        target_hypothesis_repeated = target_hypothesis.repeat(1, 1, n_points)
    else:
        ### Each quantile here already picked a hypothesis
        target_hypothesis_repeated = target_hypothesis

    ## L2 distance
    # distances = torch.sqrt((pred_depth - target_hypothesis_repeated)**2)
    distances = torch.norm(pred_depth.unsqueeze(-1) - target_hypothesis_repeated.unsqueeze(-1), p=norm_p, dim=-1)

    if mask is not None:
        mask = mask.unsqueeze(0).repeat(distances.shape[0],1).unsqueeze(-1)
        distances = distances * mask

    if threshold > 0:
        distances = torch.where(distances < threshold, torch.tensor([0.0]).to(distances.device), distances)

    if is_joint:
        ### Take the mean for all points on all rays, hypothesis is chosen per image
        quantile_mean = torch.mean(distances, axis=1) ## mean for each quantile, averaged across all rays
        samples_min = torch.min(quantile_mean, axis=0)[0]
        loss =  torch.mean(samples_min, axis=-1)


    else:
        ### Each ray selects a hypothesis
        best_hyp = torch.min(distances, dim=0)[0]   ## for each sample pick a hypothesis
        ray_mean = torch.mean(best_hyp, dim=-1) ## average across samples
        loss = torch.mean(ray_mean)  

    return loss