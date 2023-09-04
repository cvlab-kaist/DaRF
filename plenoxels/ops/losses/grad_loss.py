import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange
import imageio

def imgrad_yx(img):
    N,C,_,_ = img.size()
    grad_y, grad_x = imgrad(img)
    return torch.cat((grad_y.view(N,C,-1), grad_x.view(N,C,-1)), dim=1)


def imgrad(img):
    # img = torch.mean(img, 1, True)
    img = rearrange(img.unsqueeze(0), 'b h w c -> b c h w') 
    fx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv1.weight = nn.Parameter(weight)
    grad_x = conv1(img)

    fy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv2.weight = nn.Parameter(weight)
    grad_y = conv2(img)
    return grad_y, grad_x


class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()
    
    def forward(self, inputs, targets):
        inputs_grad_y, inputs_grad_x = imgrad(inputs)
        targets_grad_y, targets_grad_x = imgrad(targets)
        
        # save_depth = (inputs - inputs.min()) / (inputs.max() - inputs.min())
        # save_depth2 = (targets - targets.min()) / (targets.max() - targets.min())
        # imageio.imwrite("./r_depth.png",(save_depth.detach().cpu().numpy() * 255).astype(np.uint8))
        # imageio.imwrite("./t_depth.png",(save_depth2.detach().cpu().numpy() * 255).astype(np.uint8))
        # import pdb; pdb.set_trace()
        
            
        inputs_grad = torch.cat((inputs_grad_y.view(1,-1), inputs_grad_x.view(1,-1)), dim=1)
        targets_grad = torch.cat((targets_grad_y.view(1,-1), targets_grad_x.view(1,-1)), dim=1)
        
        return torch.sum( torch.mean( torch.abs(inputs_grad-targets_grad) ) )

def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)
    
def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


class GradientLoss(nn.Module):
    def __init__(self, scales=1, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total

