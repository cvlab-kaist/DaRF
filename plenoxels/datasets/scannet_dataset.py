import json
import logging as log
import os
from typing import Tuple, Optional, Any

import numpy as np
import torch
import imageio

from .data_loading import parallel_load_images
from .ray_utils import get_ray_directions, generate_hemispherical_orbit, get_rays
from .intrinsics import Intrinsics
from .base_dataset import BaseDataset

import math


class ScannetDataset(BaseDataset):
    def __init__(self,
                 datadir,
                 split: str,
                 batch_size: Optional[int] = None,
                 downsample: float = 1.0,
                 max_frames: Optional[int] = None,
                 near: float = 0.0,
                 far: float = 1.0,
                 bbox_radius : float = 5.0, 
                 fewshot : int = None,
                 patch_rendering : int = None,
                 novel_patch_size: int = None,
                 stride = 1,
                 novel_depth_loss = None,                 
                 convention = "X", 
                 ):
        self.downsample = downsample
        self.max_frames = max_frames
        self.near_far = [near, far]
        self.fewshot = fewshot
        self.patch_rendering = patch_rendering


        if split == 'render':
            poses,_raw_intrinsics,scale = load_render_pose(datadir, split)
            height, width = 468,624
            intrinsics = load_scannet_intrinsics(_raw_intrinsics, height, width, downsample)
            imgs,depths,self.imgs_all = None,None,None
            colmaps=None
            poses = torch.tensor(poses)
        else:
            imgs, depths, poses, _raw_intrinsics, _all, scale, colmaps,filenames = load_scannet_images(datadir, split, fewshot)
            height, width = imgs.shape[1:3]

            intrinsics = load_scannet_intrinsics(_raw_intrinsics, height, width, downsample)
            imgs = torch.tensor(imgs) / 255.0
            poses = torch.tensor(poses)
            if colmaps is not None:
                colmaps = torch.tensor(colmaps)
            if depths is not None:
                depths = torch.tensor(depths)

            if split == 'train' and self.fewshot is not None:
                self.imgs_all = torch.tensor(_all[0])
                self.depths_all = torch.tensor(_all[1])
                self.poses_all = torch.tensor(_all[2])
            else:
                self.imgs_all, self.depths_all,self.poses_all = imgs, depths, poses

        if patch_rendering:
            rays_o, rays_d = create_scannet_rays_patch(poses, intrinsics=intrinsics)
        else:
            rays_o, rays_d, imgs, depths = create_scannet_rays(
                imgs, depths, poses, merge_all = (split == 'train'), intrinsics=intrinsics)
            
            norm_pose = poses.clone()
            norm_pose[:, :3, 3] -= scale[0]
            norm_pose[:, :3, 3] *= scale[1]

        all_depth_hypothesis = None



        if novel_depth_loss is not None:
            _,_,poses_test,_,_, scale_center, _,_ = load_scannet_images(datadir,'test')
            poses_test = torch.tensor(poses_test)
            
            noisy_poses_all = generate_noisy_pose(poses,convention=convention, scale_center=scale_center)
            noisy_poses_test = generate_noisy_pose(poses_test,convention=convention, scale_center=scale_center)

            rays_o_novel,rays_d_novel = create_scannet_rays_patch(
                noisy_poses_all, intrinsics=intrinsics)
            
            image_len = len(poses)

            unseen_idx = np.setdiff1d(np.arange(0,image_len), np.array([]))

            rays_o_novel = rays_o_novel[unseen_idx]
            rays_d_novel = rays_d_novel[unseen_idx]
            
            rays_o_test,rays_d_test = create_scannet_rays_patch(
                noisy_poses_test, intrinsics=intrinsics)
            
            self.seen_poses = self.poses_all[unseen_idx]
        
            warp_depths = self.depths_all[unseen_idx]
            self.seen_rgb = self.imgs_all[unseen_idx]
            self.rays_o_seen, self.rays_d_seen  = rays_o, rays_d

        else:
            rays_o_novel, rays_d_novel = None,None
            rays_o_test,rays_d_test = None,None
            noisy_poses_all, noisy_poses_test = None,None
            warp_depths = None

        super().__init__(
            datadir=datadir,
            split=split,
            scene_bbox=get_360_bbox(datadir, is_contracted=False, bbox_radius = bbox_radius),
            is_ndc=False,
            is_contracted=False,
            batch_size=batch_size,
            imgs=imgs,
            depths = depths,
            warp_depths = warp_depths,
            rays_o=rays_o,
            rays_d=rays_d,
            intrinsics=intrinsics,
            patch_rendering = patch_rendering,
            rays_o_novel = rays_o_novel,
            rays_d_novel = rays_d_novel,
            rays_o_test = rays_o_test,
            rays_d_test = rays_d_test,
            novel_patch_size = novel_patch_size,
            imgs_all = self.imgs_all,
            stride = stride,
            poses_noise_novel = noisy_poses_all,
            poses_noise_test = noisy_poses_test,
            all_depth_hypothesis = all_depth_hypothesis,
            colmaps = colmaps,
        )
        log.info(f"TnTDataset. Loaded {split} set from {datadir}."
                 f"{len(poses)} images of shape {self.img_h}x{self.img_w}. "
                 f"Images loaded: {imgs is not None}. "
                 f"Sampling without replacement={self.use_permutation}. {intrinsics}"
                 f"near : {near}, far : {far}")

    def __getitem__(self, index):
        out = super().__getitem__(index)
        out["bg_color"] = None
        out["near_fars"] = torch.tensor([self.near_far])
        
        return out
    
def load_render_pose(subject_dir,split,fewshot=None,load_depth=1):
    with open(os.path.join(subject_dir,'transforms_video.json'),'r') as fp:
        meta = json.load(fp)

    poses_all = []

    if '0781' in subject_dir:
        scale = np.array(0.20028356645847725) 
        center = np.array([-0.79943419,  0.27920669, -0.09046236])
    elif '0758' in subject_dir:
        scale = np.array(0.315422746016986) 
        center = np.array([ 0.08594328, -0.19489145, -0.18269538] )
    elif '0710' in subject_dir:
        scale = np.array(0.43035425417543677) 
        center = np.array([ 0.10597146, -0.07849552,  0.14555982])


    for frame in meta['frames']:
        poses_all.append(np.array(frame['transform_matrix']))
    
    poses_all = np.stack(poses_all,0)

    poses_all[:, :3, 3] -= center
    poses_all[:, :3, 3] *= scale

    for frame in meta['frames']:
        fx, fy, cx, cy = frame['fx'], frame['fy'], frame['cx'], frame['cy']
        intrinsics = np.array((fx, fy, cx, cy))
        break

    return poses_all, intrinsics, [scale,center]

def load_scannet_images(subject_dir, split, fewshot=None, load_depth = 1):
    if '0781' in subject_dir:
        colmap_scale = np.array(0.20028356645847725) 
    elif '0758' in subject_dir:
        colmap_scale = np.array(0.315422746016986) 
    elif '0710' in subject_dir:
        colmap_scale = np.array(0.43035425417543677) 

    with open(os.path.join(subject_dir, 'transforms_{}.json'.format(split)), 'r') as fp:
        meta = json.load(fp)
    
    another_split = 'test' if split=='train' else 'train'
    with open(os.path.join(subject_dir, 'transforms_{}.json'.format(another_split)), 'r') as fp:
        meta_another = json.load(fp)

    depth_scaling_factor = 10000.

    images_all = []
    depths_all = [] if load_depth is not None else None
    colmaps_all = [] if load_depth is not None else None
    poses_all = []
    poses_another = []
    filenames = []

    meta['frames'].sort(key = lambda x:int(x["file_path"].split("/")[-1].split(".")[0]))
    

    for frame in meta['frames']:
        img_dir = os.path.join(subject_dir,frame['file_path'])        

        img = imageio.imread(img_dir)
        isdepth = os.path.isdir(os.path.join(subject_dir,frame['file_path'].split('/')[0],'midas_depth'))

        if split == "test":
            depth_dir = os.path.join(subject_dir,frame['file_path'].split('/')[0],'target_depth',frame['file_path'].split('/')[-1].split('.')[0]+'.png')
            depth = imageio.imread(depth_dir).astype(np.float64)
            depth = (depth/ 1000.0).astype(np.float32)
            depths_all.append(depth)

            colmap_dir =  os.path.join(subject_dir,frame['file_path'].split('/')[0],'depth',frame['file_path'].split('/')[-1].split('.')[0]+'.png')
            colmap = imageio.imread(colmap_dir)
            colmap = (colmap / 1000.0).astype(np.float32)
            colmaps_all.append(colmap)

        elif load_depth is not None and isdepth:
            
            depth_dir = os.path.join(subject_dir,frame['file_path'].split('/')[0],'midas_depth',frame['file_path'].split('/')[-1].split('.')[0]+'_depth.png')
            depth = imageio.imread(depth_dir)

            if depth.ndim == 2:
                depth = np.expand_dims(depth, -1)

            depth = (depth / depth_scaling_factor).astype(np.float32)
            depths_all.append(depth)

            colmap_dir =  os.path.join(subject_dir,frame['file_path'].split('/')[0],'depth',frame['file_path'].split('/')[-1].split('.')[0]+'.png')
            colmap = imageio.imread(colmap_dir)
            colmap = (colmap / 1000.0).astype(np.float32) * colmap_scale
            colmaps_all.append(colmap)

        filenames.append(frame['file_path'])
        images_all.append(img)
        poses_all.append(np.array(frame['transform_matrix']))

    if len(depths_all) != len(images_all):
        depths_all = None
    
    for frame in meta_another['frames']:
        poses_another.append(np.array(frame['transform_matrix']))

    images_all = np.stack(images_all, 0)
    poses_all = np.stack(poses_all, 0)
    if depths_all is not None:
        depths_all = np.stack(depths_all,0).squeeze()
        colmaps_all = np.stack(colmaps_all,0)
    
    for frame in meta['frames']:
        fx, fy, cx, cy = frame['fx'], frame['fy'], frame['cx'], frame['cy']
        intrinsics = np.array((fx, fy, cx, cy))
        break
    
    _poses_all = np.concatenate((np.stack(poses_all,0),np.stack(poses_another,0)),axis=0)

    min_vertices = _poses_all[:,:3,3].min(axis=0)
    max_vertices = _poses_all[:,:3,3].max(axis=0)
 
    center = (min_vertices + max_vertices) / 2.0
    scale = 2.0 / (np.max(max_vertices - min_vertices) + 3.0)

    poses_all[:, :3, 3] -= center
    poses_all[:, :3, 3] *= scale



    if (fewshot is not None) and split=='train':
        if '0710' in subject_dir:
            few_gen = np.array([1,2,4,5,6,10,11,12,13])
            print('fewshot : ', few_gen)
        elif '0758' in subject_dir:
            few_gen = np.array([0,2,4,6,8,10,12,14,16,18])
            print('fewshot : ', few_gen)
        elif '0781' in subject_dir:
            few_gen = np.array([0,2,4,6,8,10,12,14,16])
            print('fewshot : ', few_gen)
            

        images = images_all[few_gen]
        camtoworlds = poses_all[few_gen]
        depths = depths_all[few_gen] if depths_all is not None else None
        
        return images, depths, camtoworlds, intrinsics, [images_all, depths_all, poses_all], [scale, center], torch.tensor(colmaps_all)

    else:
        return images_all, depths_all, poses_all, intrinsics, None, [scale, center], torch.tensor(colmaps_all),filenames


def load_scannet_intrinsics(raw_intrinsic, img_h, img_w, downsample) -> Intrinsics:
    height = img_h
    width = img_w
    
    fl_x = (raw_intrinsic[0]) / downsample
    fl_y = (raw_intrinsic[1]) / downsample

    cx = (raw_intrinsic[2] / downsample)
    cy = (raw_intrinsic[3] / downsample)
    return Intrinsics(height=height, width=width, focal_x=fl_x, focal_y=fl_y, center_x=cx, center_y=cy)

def get_360_bbox(datadir, bbox_radius, is_contracted=False):
    radius = bbox_radius

    return torch.tensor([[-radius, -radius, -radius], [radius, radius, radius]])

def create_scannet_rays_patch(
              poses: torch.Tensor,
              intrinsics: Intrinsics) -> Tuple[torch.Tensor, torch.Tensor]:
    directions = get_ray_directions(intrinsics, opengl_camera=True)  # [H, W, 3]
    num_frames = poses.shape[0]

    all_rays_o, all_rays_d = [], []
    for i in range(num_frames):
        rays_o, rays_d = get_rays(directions, poses[i], ndc=False, normalize_rd=True)  # [H*W, 3] each
        all_rays_o.append(rays_o.reshape(intrinsics.height, intrinsics.width, 3)) # [H, W, 3]
        all_rays_d.append(rays_d.reshape(intrinsics.height, intrinsics.width, 3)) # [H, W, 3]

    all_rays_o = torch.stack(all_rays_o, 0).to(dtype=torch.float32)  # [n_frames, H, W, 3]
    all_rays_d = torch.stack(all_rays_d, 0).to(dtype=torch.float32)  # [n_frames, H, W, 3]

    return all_rays_o, all_rays_d

def create_scannet_rays(
              imgs: Optional[torch.Tensor],
              depths: torch.Tensor,
              poses: torch.Tensor,
              merge_all: bool,
              intrinsics: Intrinsics) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    directions = get_ray_directions(intrinsics, opengl_camera=True)  # [H, W, 3]
    num_frames = poses.shape[0]

    all_rays_o, all_rays_d = [], []
    for i in range(num_frames):
        rays_o, rays_d = get_rays(directions, poses[i], ndc=False, normalize_rd=True)  # h*w, 3
        all_rays_o.append(rays_o)
        all_rays_d.append(rays_d)

    all_rays_o = torch.cat(all_rays_o, 0).to(dtype=torch.float32)  # [n_frames * h * w, 3]
    all_rays_d = torch.cat(all_rays_d, 0).to(dtype=torch.float32)  # [n_frames * h * w, 3]

    if imgs is not None:
        imgs = imgs.view(-1, imgs.shape[-1]).to(dtype=torch.float32)   # [N*H*W, 3/4]
    
    if depths is not None:
        depths = depths.view(-1, 1).to(dtype=torch.float32)

    if not merge_all:
        num_pixels = intrinsics.height * intrinsics.width
        
        if imgs is not None:
            imgs = imgs.view(num_frames, num_pixels, -1)  # [N, H*W, 3/4]
        if depths is not None:
            depths = depths.view(num_frames, num_pixels, -1)

        all_rays_o = all_rays_o.view(num_frames, num_pixels, 3)  # [N, H*W, 3]
        all_rays_d = all_rays_d.view(num_frames, num_pixels, 3)  # [N, H*W, 3]
    return all_rays_o, all_rays_d, imgs, depths

def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)
        
def generate_noisy_pose(input_poses, convention='X', scale_center=[1,1]):

    n_ray, _, _ = input_poses.shape
    R, t = input_poses[:,:3,:3], input_poses[:,:,-1]

    angles = matrix_to_euler_angles(input_poses[:,:3,:3],"XYZ")
    if convention =="X":
        noise = torch.rand(1)*0.2
        angles[:,0] += noise
    elif convention == "XY":
        noise = torch.rand(2)*0.2
        angles[:,:2] += noise
    elif convention == "XYZ":
        noise = torch.rand(3)*0.2
        angles[:,:3] += noise

    recon_pose = euler_angles_to_matrix(angles,"XYZ")
    noisy_pose = torch.zeros_like(input_poses)
    noisy_pose[:,:3,:3] = recon_pose

    noisy_pose[:,:,-1] = t

    return noisy_pose


###################################################################################################
###################################################################################################
 
def matrix_to_euler_angles(matrix: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.
    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.
    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.
    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])

def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
) -> torch.Tensor:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.
    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.
    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])

def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.
    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))

def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")


#####################################################################################33
def reprojecter(pose_org, intrisic, depth_GT,
                rays_novel, pose_novel, depth_novel,
                ):
    # Unknown view : points generation
    # xyz = repeat_interleave(xyz, NS)  #(SB*NS, B, 3)
    xyz = rays_novel.origins + rays_novel.direction*depth_novel 
    # Get projected depths
    pts_to_tgt_origin = xyz - pose_org[:3,-1][None, :]
    projected_depth = torch.linalg.norm(pts_to_tgt_origin, dim=-1, keepdims=True)

    # Known view pose inverse
    # Extrinsic
    rot = pose_org[:, :3, :3].transpose(1, 2)  # (B, 3, 3)
    trans = -torch.bmm(rot, pose_org[:, :3, 3:])  # (B, 3, 1)
    pose_org = torch.cat((rot, trans), dim=-1)  # (B, 3, 4)
    # Intrinsic
    
    # Scannet: Scene 781
    focal = -584.0  # focal = -focal.float()
    c = np.array([312.0, 234.0]).unsqueeze(0)  # c = (shape_novel * 0.5).unsqueeze(0)
    # Rotate and translate
    xyz_rot = torch.matmul(pose_org[:, None, :3, :3], xyz.unsqueeze(-1))[..., 0]
    xyz = xyz_rot + pose_org[:, None, :3, 3]
    # Pixel Space
    uv = -xyz[:, :, :2] / xyz[:, :, 2:]  # (SB, B, 2)
    uv *= repeat_interleave(
        focal.unsqueeze(1), NS if focal.shape[0] > 1 else 1
    )
    uv += repeat_interleave(
        c.unsqueeze(1), NS if c.shape[0] > 1 else 1
    )  # (SB*NS, B, 2)
    uv = uv.unsqueeze(2)  # (B, N, 1, 2)
    # Get corresponding depth of known view pose
    projected_depth_GT = F.grid_sample(
        depth_GT,
        uv,
        align_corners=True, #0-255 normalized, if False, 0-256 normalized
        mode=index_interp,
        padding_mode=index_padding,
    )
    return projected_depth, projected_depth_GT