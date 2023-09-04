from abc import ABC
import os
from typing import Optional, List, Union

import torch
import imageio
from torch.utils.data import Dataset
from torchvision import transforms
from plenoxels.models.dpt_depth import DPTDepthModel
from .intrinsics import Intrinsics


class BaseDataset(Dataset, ABC):
    def __init__(self,
                 datadir: str,
                 scene_bbox: torch.Tensor,
                 split: str,
                 is_ndc: bool,
                 is_contracted: bool,
                 rays_o: Optional[torch.Tensor],
                 rays_d: Optional[torch.Tensor],
                 intrinsics: Union[Intrinsics, List[Intrinsics]],
                 batch_size: Optional[int] = None,
                 imgs: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
                 depths: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
                 warp_depths: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
                 sampling_weights: Optional[torch.Tensor] = None,
                 weights_subsampled: int = 1,
                 patch_rendering : int = None,
                 rays_o_novel : Optional[torch.Tensor] = None,
                 rays_d_novel : Optional[torch.Tensor] = None,
                 rays_o_test : Optional[torch.Tensor] = None,
                 rays_d_test : Optional[torch.Tensor] = None,
                 novel_patch_size : Optional[int] = None,
                 imgs_all : Optional[torch.Tensor] = None,
                 stride : Optional[int] = 1,
                 poses_noise_novel : Optional[torch.Tensor] = None,
                 poses_noise_test : Optional[torch.Tensor] = None,
                 all_depth_hypothesis : Optional[torch.Tensor] = None,
                 colmaps: Optional[torch.Tensor] = None,
                 ):
        self.datadir = datadir
        self.name = os.path.basename(self.datadir)
        self.scene_bbox = scene_bbox
        self.split = split
        self.is_ndc = is_ndc
        self.is_contracted = is_contracted
        self.weights_subsampled = weights_subsampled
        self.batch_size = batch_size
        if self.split == 'train':
            assert self.batch_size is not None
        self.rays_o = rays_o
        self.rays_d = rays_d
        self.imgs = imgs
        self.depths = depths
        self.colmaps = colmaps

        if warp_depths is not None:
            self.warp_depths = warp_depths
        else:
            self.warp_depths = depths

        self.patch_rendering = patch_rendering

        self.rays_o_novel = rays_o_novel
        self.rays_d_novel = rays_d_novel
        self.rays_o_test = rays_o_test
        self.rays_d_test = rays_d_test
        self.noise_pose_novel = poses_noise_novel
        self.noise_pose_test = poses_noise_test

        self.imgs_all = imgs_all
        self.novel_patch_size = novel_patch_size
        self.stride = stride

        self.all_depth_hypothesis = all_depth_hypothesis

        if self.imgs is not None:
            self.num_samples = len(self.imgs) if patch_rendering is None else torch.numel(self.imgs) // 3
        elif self.rays_o is not None:
            self.num_samples = len(self.rays_o) if patch_rendering is None else torch.numel(self.rays_o) // 3
        else:
            self.num_samples = None
            #raise RuntimeError("Can't figure out num_samples.")
        if split=='train' and self.rays_o_novel is not None:
            self.rays_o_novel_test=  torch.cat((self.rays_o_novel,self.rays_o_test),dim=0)
            self.rays_d_novel_test=  torch.cat((self.rays_d_novel,self.rays_d_test),dim=0)
            self.noise_pose_novel_test = torch.cat((self.noise_pose_novel,self.noise_pose_test),dim=0)

        self.intrinsics = intrinsics
        self.sampling_weights = sampling_weights
        if self.sampling_weights is not None:
            assert len(self.sampling_weights) == self.num_samples, (
                f"Expected {self.num_samples} sampling weights but given {len(self.sampling_weights)}."
            )
        self.sampling_batch_size = 2_000_000  # Increase this?
        if self.num_samples is not None:
            self.use_permutation = self.num_samples < 100_000_000  # 64M is static
        else:
            self.use_permutation = True

        self.perm = None


        
    @property
    def img_h(self) -> Union[int, List[int]]:
        if isinstance(self.intrinsics, list):
            return [i.height for i in self.intrinsics]
        return self.intrinsics.height

    @property
    def img_w(self) -> Union[int, List[int]]:
        if isinstance(self.intrinsics, list):
            return [i.width for i in self.intrinsics]
        return self.intrinsics.width

    def reset_iter(self):
        if self.sampling_weights is None and self.use_permutation:
            self.perm = torch.randperm(self.num_samples)
        else:
            del self.perm
            self.perm = None

    def get_rand_ids(self, index):
        assert self.batch_size is not None, "Can't get rand_ids for test split"
        if self.sampling_weights is not None:
            print('You shall not see this. exiting :(')
            exit()
            batch_size = self.batch_size // (self.weights_subsampled ** 2)
            num_weights = len(self.sampling_weights)
            if num_weights > self.sampling_batch_size:
                # Take a uniform random sample first, then according to the weights
                subset = torch.randint(
                    0, num_weights, size=(self.sampling_batch_size,),
                    dtype=torch.int64, device=self.sampling_weights.device)
                samples = torch.multinomial(
                    input=self.sampling_weights[subset], num_samples=batch_size)
                return subset[samples]
            return torch.multinomial(
                input=self.sampling_weights, num_samples=batch_size)
        else:
            batch_size = self.batch_size
            if self.use_permutation:
                return self.perm[index * batch_size: (index + 1) * batch_size]
            else:
                return torch.randint(0, self.num_samples, size=(batch_size, ))

    def __len__(self):
        if self.split == 'train':
            return (self.num_samples + self.batch_size - 1) // self.batch_size
        else:
            return self.num_samples

    def __getitem__(self, index, return_idxs: bool = False):
        if self.split == 'train':
            index = self.get_rand_ids(index)            

        out = {}

        if self.rays_o_novel is not None:
            H, W = self.imgs.shape[1:3]    

            novel_id = torch.randint(self.rays_o_novel.shape[0],(1,))

            x, y = patch_index(self.novel_patch_size, W, H, 0, 0, self.imgs.device,self.stride)
            out['rays_o_novel_test'] = self.rays_o_novel_test[novel_id,y,x]
            out['rays_d_novel_test'] = self.rays_d_novel_test[novel_id,y,x]
            out['pose_novel_test'] = self.noise_pose_novel_test[novel_id]

            if self.warp_depths is not None:
                out['depth_seen'] = self.warp_depths[novel_id, y, x].unsqueeze(-1).reshape(self.novel_patch_size,self.novel_patch_size,1).permute(2,0,1)
            out['rgb_seen'] = self.seen_rgb[novel_id, y, x].unsqueeze(-1).float()
            out['pose_seen'] = self.seen_poses[novel_id]
            out['rays_o_seen'] = self.rays_o_seen[novel_id, y, x]
            out['rays_d_seen'] = self.rays_d_seen[novel_id, y, x]
            out['intrinsic'] = self.intrinsics
    
        if self.patch_rendering is not None:

            self.patch_sampling_region_width = 0
            self.patch_sampling_region_height = 0
            patch_size = self.patch_rendering
            H, W = self.imgs.shape[1:3]            
            
            rand_rays_o = self.rays_o.view(-1, 3)[index]
            rand_rays_d = self.rays_d.view(-1, 3)[index]
    
            rand_rays_imgs = self.imgs.view(-1, 3)[index]
            if self.depths is not None:
                rand_rays_depths = self.depths.view(-1, 1)[index]

            patch_rays_o, patch_rays_d, patch_rays_imgs, patch_rays_depths = [], [], [], []
            patch_rays_all_depths_hypothesis = []

            image_id = torch.randint(self.imgs.shape[0], (1,))
            x, y = patch_index(patch_size, W, H, self.patch_sampling_region_width, self.patch_sampling_region_height, self.imgs.device,1)
            patch_rays_o.append(self.rays_o[image_id, y, x])
            patch_rays_d.append(self.rays_d[image_id, y, x])
            patch_rays_imgs.append(self.imgs[image_id, y, x])
            if self.depths is not None: 
                patch_rays_depths.append(self.depths[image_id, y, x].unsqueeze(-1))

            out["patch_img_id"] = image_id
            out["patch_img_x"] = x
            out["patch_img_y"] = y
            out["patch_img"] = self.imgs[image_id]
            out["patch_mono_depth"] = self.depths[image_id]
            if self.colmaps is not None:
                out["patch_colmap"] = self.colmaps[image_id]


            if self.rays_o is not None:
                out["rays_o"] = torch.cat(([rand_rays_o] + patch_rays_o)) # [batch_size + (patch_size)^2, 3]
                
            if self.rays_d is not None:
                out["rays_d"] = torch.cat(([rand_rays_d] + patch_rays_d))

            if self.imgs is not None:
                out["imgs"] = torch.cat(([rand_rays_imgs] + patch_rays_imgs)) 
            
            if self.depths is not None:
                out["depths"] = torch.cat(([rand_rays_depths] + patch_rays_depths))
            else:
                out["imgs"] = None

            out["stride"] = self.stride
            
            if return_idxs:
                return out, index
            return out
        


        else:
            if self.rays_o is not None:
                out["rays_o"] = self.rays_o[index]

            if self.rays_d is not None:
                out["rays_d"] = self.rays_d[index]
            if self.imgs is not None:
                out["imgs"] = self.imgs[index]
            if self.depths is not None:
                out["depths"] = self.depths[index]
            else:
                out["imgs"] = None
            if return_idxs:
                return out, index
            return out

def patch_index(patch_size, W, H, sample_w, sample_h, device,stride):
    x0 = torch.randint(sample_w, W - (sample_w + (patch_size-1)*stride)-1, (1, ))
    y0 = torch.randint(sample_h, H - (sample_h + (patch_size-1)*stride)-1, (1, ))
    x, y = torch.meshgrid(
    torch.arange(start=int(x0), end=int(x0 + patch_size*stride), step=stride),
    torch.arange(start=int(y0), end=int(y0 + patch_size*stride), step=stride),
    indexing="xy")
    x_patch = x.flatten()
    y_patch = y.flatten()

    

    return x_patch, y_patch