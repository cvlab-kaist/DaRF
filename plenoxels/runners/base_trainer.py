import abc
import random
import logging as log
import math
import os
from copy import copy
from typing import Iterable, Optional, Union, Dict, Tuple, Sequence, MutableMapping
from torchvision import transforms
import torchvision

import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from plenoxels.utils.timer import CudaTimer
from plenoxels.utils.ema import EMA
from plenoxels.models.lowrank_model import LowrankModel

from plenoxels.models.extractor import VitExtractor
from plenoxels.utils.my_tqdm import tqdm
from plenoxels.ops.image import metrics
from plenoxels.ops.image.io import write_png
from plenoxels.runners.regularization import Regularizer
from plenoxels.ops.lr_scheduling import (
    get_cosine_schedule_with_warmup, get_step_schedule_with_warmup
)
from plenoxels.ops.losses.ranking_loss import RenderedRankingLoss
from plenoxels.ops.losses.scaleshiftloss import ScaleAndShiftInvariantLoss,ScaleAndShiftInvariantGradientLoss, GradL1Loss
from plenoxels.ops.losses.scaleshiftloss import compute_scale_and_shift
from plenoxels.ops.losses.grad_loss import GradientLoss

from plenoxels.models.dpt_depth import DPTDepthModel
from lpips import LPIPS

from pprint import pprint

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred)**2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3

def depth_stat(pred, gt = None):
    cond1 = (pred <= pred.quantile(0.98))
    cond2 = (pred >= pred.quantile(0.02))
    valid = torch.logical_and(cond1, cond2)
    gt = gt[valid]
    pred = pred[valid]
    if gt is not None:
        print(f"gt_stat : {gt.max().item()}, {gt.median().item()}, {gt.min().item()}")
        print(f"pred_corr : {pred[gt.argmax()].item()}, {pred[gt==gt.median()].mean().item()}, {pred[gt.argmin()].mean().item()}")
    
    print(f"pred_stat : {pred.max().item()}, {pred.median().item()}, {pred.min().item()}")

    if gt is not None:
        scale, shift = np.polyfit(pred, gt, 1)
        pred = pred*scale + shift
        print(f"pred_corr : {pred[gt.argmax()].item()}, {pred[gt==gt.median()].mean().item()}, {pred[gt.argmin()].mean().item()}")
        print('scale : ', scale, 'shift : ', shift)


class BaseTrainer(abc.ABC):
    def __init__(self,
                 train_data_loader: Iterable,
                 num_steps: int,
                 logdir: str,
                 expname: str,
                 train_fp16: bool,
                 save_every: int,
                 valid_every: int,
                 save_outputs: bool,
                 device: Union[str, torch.device],
                 **kwargs):
        
        self.train_data_loader = train_data_loader
        self.num_steps = num_steps
        self.train_fp16 = train_fp16
        self.save_every = save_every
        self.valid_every = valid_every
        self.save_outputs = save_outputs
        self.device = device
        self.eval_batch_size = kwargs.get('eval_batch_size', 8129)
        self.extra_args = kwargs
        self.timer = CudaTimer(enabled=False)

        self.depth_loss = kwargs['depth_loss']
        self.depth_loss_range = kwargs['depth_loss_range']
        self.depth_weight = kwargs['depth_weight']
        self.batch_size = kwargs['batch_size']
        self.patch = kwargs['patch_rendering']
        self.novel_patch_size = kwargs['novel_patch_size']
        self.fake_img = kwargs['fake_img']          
        self.ranking_step = kwargs['ranking_step']
        self.novel_depth_loss = kwargs.get('novel_depth_loss',None)
        self.novel_depth_loss_function = kwargs.get('novel_depth_loss_function',None)
        self.novel_depth_loss_range = kwargs.get('novel_depth_loss_range',None)

        self.warp = kwargs.get('warp',False)

        self.DPT_adaptor = kwargs.get('DPT_adaptor',False)
        self.adaptor_weight = kwargs.get('adaptor_weight',0.0)
        self.novel_detach = kwargs.get('novel_detach',True)

        self.ss_weight = kwargs.get('ss_weight',0.1)
        self.l1_weight = kwargs.get('l1_weight',0.1)
        self.dpt_scale = kwargs.get('dpt_scale',0.2)
        self.dpt_shift = kwargs.get('dpt_shift',0.12)
        self.dpt_weight_path = kwargs['dpt_weight_path']        
        

        self.log_dir = os.path.join(logdir, expname)
        
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.global_step: Optional[int] = None
        self.loss_info: Optional[Dict[str, EMA]] = None
        self.lpips_alex = LPIPS()

        self.model = self.init_model(**self.extra_args)

        ##### omnidata
        map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
        dpt_root_dir = self.dpt_weight_path
        pretrained_weights_path = os.path.join(dpt_root_dir,'dpt_hybrid_384.pt')

        self.depth_model = DPTDepthModel(backbone='vitb_rn50_384',inv=True)
        checkpoint = torch.load(pretrained_weights_path, map_location=map_location)

        if 'state_dict' in checkpoint:
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint

        self.depth_model.load_state_dict(state_dict)
        self.depth_model.to(self.device)
        self.trans_totensor = transforms.Compose([transforms.Normalize(mean=0.5, std=0.5)])

        if self.DPT_adaptor:
            self.adaptor_DPT = DPTDepthModel(backbone='vitb_rn50_384',inv=True)
            checkpoint = torch.load(pretrained_weights_path, map_location=map_location)

            if 'state_dict' in checkpoint:
                state_dict = {}
                for k, v in checkpoint['state_dict'].items():
                    state_dict[k[6:]] = v
            else:
                state_dict = checkpoint

            self.adaptor_DPT.load_state_dict(state_dict)
            self.adaptor_DPT.to(self.device)

            for name,para in self.adaptor_DPT.named_parameters():
                if 'refinenet' in name and "out_conv" in name:
                    para.requires_grad = True
                else:
                    para.requires_grad = False

        
        self.model.to(self.device)

        self.depth_scale = nn.Parameter(torch.tensor([self.dpt_scale],requires_grad=True, device=self.device))
        self.depth_shift = nn.Parameter(torch.tensor([self.dpt_shift],requires_grad=True, device=self.device))

        self.optimizer = self.init_optim(**self.extra_args)
        self.scheduler = self.init_lr_scheduler(**self.extra_args)
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.regularizers = self.init_regularizers(**self.extra_args)
        self.gscaler = torch.cuda.amp.GradScaler(enabled=self.train_fp16)


    @abc.abstractmethod
    def eval_step(self, data, **kwargs) -> MutableMapping[str, torch.Tensor]:
        self.model.eval()
        return None  # noqa

    def train_step(self, data, **kwargs) -> bool:
        self.model.train()
        data = self._move_data_to_device(data)
        if "timestamps" not in data:
            data["timestamps"] = None
        self.timer.check("move-to-device")

        with torch.cuda.amp.autocast(enabled=self.train_fp16):
            fwd_out = self.model(
                data['rays_o'], data['rays_d'], bg_color=data['bg_color'],
                near_far=data['near_fars'], timestamps=data['timestamps'])
            self.timer.check("model-forward")
            recon_loss = self.criterion(fwd_out['rgb'][:self.batch_size], data['imgs'][:self.batch_size]) # (batch, 3)

            # Regularization
            loss = recon_loss
            for r in self.regularizers:
                reg_loss = r.regularize(self.model, model_out=fwd_out)
                loss = loss + reg_loss

            ############### SEEN VIEW DEPTH LOSS ###############
            if self.patch is not None and self.depth_loss is not None and self.global_step < max(self.depth_loss_range):
                patch_rendered = fwd_out['depth'][-(self.patch*self.patch):]

                depth_loss = 0

                if self.DPT_adaptor:
                    data["patch_img"] = data["patch_img"].permute(0,3,1,2)
                    h_before, w_before = data["patch_img"].shape[2], data["patch_img"].shape[3]
                    h = (data["patch_img"].shape[2]//32)*32
                    w = (data["patch_img"].shape[3]//32)*32
                    
                    data["patch_img"] = F.interpolate(data["patch_img"],(h,w),mode='bilinear')
                    out_depth = self.adaptor_DPT(self.trans_totensor(data["patch_img"]))

                    with torch.no_grad():
                        out_depth_ss = self.depth_model(self.trans_totensor(data['patch_img']))
                        out_depth_ss = F.interpolate(out_depth_ss.unsqueeze(dim=1),(h_before,w_before),mode="bilinear")

                    out_depth = F.interpolate(out_depth.unsqueeze(dim=1),(h_before,w_before),mode="bilinear")
                    out_patch = out_depth[0,0,data["patch_img_y"],data["patch_img_x"]].unsqueeze(dim=1)

                    if self.adaptor_weight!=0.0:
                        ss_loss = ScaleAndShiftInvariantLoss()
                        adaptor_depth_loss,_,_ = ss_loss(out_depth.squeeze(dim=0),out_depth_ss.squeeze(dim=0),mask=None)

                        depth_loss +=self.adaptor_weight*adaptor_depth_loss

                    patch_DPT=out_patch

                else:
                    data["patch_img"] = data["patch_img"].permute(0,3,1,2)
                    h_before, w_before = data["patch_img"].shape[2], data["patch_img"].shape[3]
                    h = (data["patch_img"].shape[2]//32)*32
                    w = (data["patch_img"].shape[3]//32)*32

                    data["patch_img"] = F.interpolate(data["patch_img"],(h,w),mode='bilinear')
                    with torch.no_grad():
                        out_depth_ss = self.depth_model(self.trans_totensor(data['patch_img']))
                        out_depth_ss = F.interpolate(out_depth_ss.unsqueeze(dim=1),(h_before,w_before),mode="bilinear")

                    patch_DPT = out_depth_ss[0,0,data["patch_img_y"],data["patch_img_x"]].unsqueeze(dim=1)


                if self.depth_loss == "scale_shift":
                    ss_loss = ScaleAndShiftInvariantLoss()                                    

                    depth_loss, scale, shift = ss_loss(patch_rendered.squeeze().reshape(self.patch,self.patch),patch_DPT.squeeze().reshape(self.patch,self.patch),mask=None)
                    depth_loss = self.ss_weight * depth_loss

                if self.depth_loss == "l1_ranking":
                    ss_loss = ScaleAndShiftInvariantLoss()
                    l1_loss = nn.L1Loss()
                    ranking_loss = RenderedRankingLoss()


                    if self.global_step>=self.ranking_step:
                        ss_weight = self.ss_weight
                        ranking_weight = 0.1
                    else:
                        ss_weight = self.ss_weight
                        ranking_weight = 1


                    ss_depth_loss,scale,shift = ss_loss(patch_rendered.squeeze().reshape(self.patch,self.patch),patch_DPT.squeeze().reshape(self.patch,self.patch),mask=None)
                    l1_depth_loss = l1_loss(patch_DPT.squeeze()*self.depth_scale+self.depth_shift,patch_rendered.squeeze().detach())

                    depth_loss += ss_weight*ss_depth_loss + self.l1_weight * l1_depth_loss


                    depth_loss += ranking_weight* ranking_loss(patch_rendered,patch_DPT,random_sample_num = self.patch**2)


                loss = loss + self.depth_weight * depth_loss

            ##### NOVEL DEPTH LOSS ####
            if self.novel_depth_loss and self.novel_depth_loss_range[0]<=self.global_step and self.global_step<=self.novel_depth_loss_range[1]:
                novel_fwd_out = self.model(
                    data[f'rays_o{self.fake_img}'], data[f'rays_d{self.fake_img}'], bg_color=data['bg_color'],
                    near_far=data['near_fars'], timestamps=data['timestamps'])
                novel_rgb = novel_fwd_out['rgb'].reshape(self.novel_patch_size,self.novel_patch_size,3).permute(2,0,1).unsqueeze(dim=0)
                novel_depth = novel_fwd_out['depth'].reshape(self.novel_patch_size,self.novel_patch_size,1).permute(2,0,1)

                if self.DPT_adaptor:
                    if self.novel_detach:
                        with torch.no_grad():
                            depth_rgb = self.adaptor_DPT(self.trans_totensor(novel_rgb)).detach()
                    else:
                        depth_rgb = self.adaptor_DPT(self.trans_totensor(novel_rgb.detach()))
                else:
                    depth_rgb = self.depth_model(self.trans_totensor(novel_rgb.detach())).detach()
                
                if self.novel_depth_loss_function=="scale_shift":
                    scale_shift_loss = ScaleAndShiftInvariantLoss()
                   
                    if self.warp:
                        intrinsic = data['intrinsic']

                        with torch.no_grad():
                            seen_fwd_out = self.model(data['rays_o_seen'], data[f'rays_d_seen'], bg_color=data['bg_color'], near_far=data['near_fars'], timestamps=data['timestamps'])
                            depth_seen_render = seen_fwd_out['depth'].reshape(self.novel_patch_size,self.novel_patch_size,1).permute(2,0,1).unsqueeze(dim=0)
                            rgb_seen_render = seen_fwd_out['rgb'].reshape(self.novel_patch_size,self.novel_patch_size,3).permute(2,0,1).unsqueeze(dim=0)

                            seen_scale_, seen_shift_ = compute_scale_and_shift(data['depth_seen'].float(), depth_seen_render, torch.ones_like(depth_seen_render).long())
                            depth_seen = seen_scale_ * data['depth_seen'].float() + seen_shift_

                        ## seen to novel warping
                        _, warp_mask, warped_depth, _ = self.warper.forward_warp(frame1=rgb_seen_render, 
                                                depth1=depth_seen.unsqueeze(0),
                                                mask1=None,
                                                transformation1=data['pose_seen'].float(), transformation2=data['pose_novel_test'].float(),
                                                intrinsic1=intrinsic, intrinsic2=None,
                                                focal =intrinsic.focal_x / data['stride'],
                                                center =[self.novel_patch_size//2, self.novel_patch_size//2],
                                                depth2 = novel_depth)
                        

                        depth_error = torch.sqrt((warped_depth-novel_depth)**2)*warp_mask / novel_depth
                        novel_loss_mask = (depth_error < 2.0)


                        # warp_visual(novel_rgb,rgb_seen_render,depth_rgb, data['depth_seen'], novel_depth, warped_depth, warp_mask, novel_loss_mask,
                        #             "error50", self.global_step)
                        
                    else:
                        novel_loss_mask = torch.ones_like(novel_depth)

                    noveldepthloss, scale_, shift_ = scale_shift_loss(depth_rgb, novel_depth, novel_loss_mask.bool())

                
                _novel_depth_loss = self.novel_depth_loss
                loss = loss + _novel_depth_loss * noveldepthloss

            self.timer.check("regularizaion-forward")
        # Update weights
        self.optimizer.zero_grad(set_to_none=True)
        self.gscaler.scale(loss).backward()
        self.timer.check("backward")
        self.gscaler.step(self.optimizer)
        scale = self.gscaler.get_scale()
        self.gscaler.update()
        self.timer.check("scaler-step")

        # Report on losses
        if self.global_step % self.calc_metrics_every == 0:
            with torch.no_grad():
                recon_loss_val = recon_loss.item()
                self.loss_info[f"mse"].update(recon_loss_val)
                self.loss_info[f"psnr"].update(-10 * math.log10(recon_loss_val))
                if self.depth_loss is not None and self.global_step < max(self.depth_loss_range):
                    self.loss_info[f"depth_loss"].update(depth_loss)

                if self.novel_depth_loss and self.novel_depth_loss_range[0]<=self.global_step and self.global_step<=self.novel_depth_loss_range[1]:
                    self.loss_info['novel_depth_loss'].update(noveldepthloss)

                for r in self.regularizers:
                    r.report(self.loss_info)

        return scale <= self.gscaler.get_scale()

    def reprojecter(self, pose_seen, pose_novel,
                depth_seen, depth_novel, 
                rays_o_novel, rays_d_novel,
                intrisic, img_w, img_h, rgb_seen=None,
                ):
        # Unknown view : points generation
        # xyz = repeat_interleave(xyz, NS)  #(SB*NS, B, 3)

        pose_seen = pose_seen.float()
        xyz = rays_o_novel.unsqueeze(1) + rays_d_novel.unsqueeze(1) * depth_novel.reshape(rays_d_novel.shape[0],1,1)
        # Get projected depths
        pts_to_tgt_origin = xyz - pose_seen[:,:3,-1][None, :].to(xyz)
        projected_depth = torch.linalg.norm(pts_to_tgt_origin, dim=-1, keepdims=True)
        
        torchvision.utils.save_image(projected_depth.reshape(img_h, img_w), "./proj.png", normalize=True)
        # Known view pose inverse


        # Extrinsic
        rot = pose_seen[:, :3, :3].transpose(1, 2)  # (B, 3, 3)
        trans = -torch.bmm(rot, pose_seen[:, :3, 3:])  # (B, 3, 1)
        pose_seen = torch.cat((rot, trans), dim=-1).float()  # (B, 3, 4)
        
        # Intrinsic
        focal = torch.tensor(intrisic.focal_x).to(xyz) # focal = -focal.float()
        NS = 1
        c = torch.tensor([intrisic.center_x, intrisic.center_y]).unsqueeze(0).to(xyz)  # c = (shape_novel * 0.5).unsqueeze(0)
        # Rotate and translate
        
        xyz_rot = torch.matmul(pose_seen[:, None, :3, :3], xyz.unsqueeze(-1))[..., 0]
        xyz = xyz_rot + pose_seen[:, None, :3, 3]
        # Pixel Space
        uv = -xyz[:, :, :2] / (xyz[:, :, 2:])  # (SB, B, 2)

        uv = uv*focal + c
        # uv *= torch.repeat_interleave(
        #     focal, NS 
        # )
        # uv += torch.repeat_interleave(
        #     c, NS 
        # )  # (SB*NS, B, 2)

        uv = uv.unsqueeze(0)  # (B, N, 1, 2)
        # Get corresponding depth of known view pose

        projected_rgb = F.grid_sample(
            rgb_seen.unsqueeze(0).transpose(1,3).to(xyz),
            uv,
            align_corners=True, #0-255 normalized, if False, 0-256 normalized
            mode='nearest',
            padding_mode='zeros',
        )
        
        projected_depth_GT = F.grid_sample(
            depth_seen.unsqueeze(0).to(xyz),
            uv,
            align_corners=True, #0-255 normalized, if False, 0-256 normalized
            mode='nearest',
            padding_mode='zeros',
        )
        torchvision.utils.save_image(projected_depth_GT.reshape(1, img_h, img_w), "./proj_GT.png", normalize=True)
        torchvision.utils.save_image(rgb_seen.transpose(2,0).reshape(3, img_h, img_w), "./RGB.png", normalize=True)
        torchvision.utils.save_image(projected_rgb.reshape(3, img_h, img_w), "./proj_RGB.png", normalize=True)

        return projected_depth, projected_depth_GT

    def post_step(self, progress_bar):
        self.model.step_after_iter(self.global_step)
        if self.global_step % self.calc_metrics_every == 0:
            progress_bar.set_postfix_str(
                losses_to_postfix(self.loss_info, lr=self.lr), refresh=False)
            for loss_name, loss_val in self.loss_info.items():
                self.writer.add_scalar(f"train/loss/{loss_name}", loss_val.value, self.global_step)
                if self.timer.enabled:
                    tsum = 0.
                    tstr = "Timings: "
                    for tname, tval in self.timer.timings.items():
                        tstr += f"{tname}={tval:.1f}ms  "
                        tsum += tval
                    tstr += f"tot={tsum:.1f}ms"
                    log.info(tstr)
        progress_bar.update(1)
        if self.valid_every > -1 and self.global_step % self.valid_every == 0:
            print()
            self.validate()
        if self.save_every > -1 and self.global_step % self.save_every == 0:
            print()
            self.save_model()

    def pre_epoch(self):
        self.loss_info = self.init_epoch_info()

    def train(self):
        """Override this if some very specific training procedure is needed."""
        if self.global_step is None:
            self.global_step = 0
        log.info(f"Starting training from step {self.global_step + 1}")
        pb = tqdm(initial=self.global_step, total=self.num_steps)
        try:
            self.pre_epoch()
            batch_iter = iter(self.train_data_loader)
            while self.global_step < self.num_steps:
                self.timer.reset()
                self.model.step_before_iter(self.global_step)
                self.global_step += 1
                self.timer.check("step-before-iter")
                try:
                    data = next(batch_iter)
                    self.timer.check("dloader-next")
                except StopIteration:
                    self.pre_epoch()
                    batch_iter = iter(self.train_data_loader)
                    data = next(batch_iter)
                    log.info("Reset data-iterator")

                try:
                    step_successful = self.train_step(data, **self.extra_args)
                except StopIteration:
                    self.pre_epoch()
                    batch_iter = iter(self.train_data_loader)
                    log.info("Reset data-iterator")
                    step_successful = True

                if step_successful and self.scheduler is not None:
                    self.scheduler.step()

                for r in self.regularizers:
                    r.step(self.global_step)
                self.post_step(progress_bar=pb)
                self.timer.check("after-step")
        finally:
            pb.close()
            self.writer.close()

    def _move_data_to_device(self, data):
        data["rays_o"] = data["rays_o"].to(self.device)
        data["rays_d"] = data["rays_d"].to(self.device)
        data["imgs"] = data["imgs"].to(self.device)
        data['depths'] = data['depths'].to(self.device)
        data["near_fars"] = data["near_fars"].to(self.device)
        if "timestamps" in data:
            data["timestamps"] = data["timestamps"].to(self.device)
        bg_color = data["bg_color"]
        if isinstance(bg_color, torch.Tensor):
            bg_color = bg_color.to(self.device)
        data["bg_color"] = bg_color
        if self.novel_depth_loss is not None:
            data["rays_o_novel_test"] = data["rays_o_novel_test"].to(self.device)
            data["rays_d_novel_test"] = data["rays_d_novel_test"].to(self.device)
            data["pose_novel_test"] = data["pose_novel_test"].to(self.device)
        if self.warp:
            data["rays_o_seen"] = data["rays_o_seen"].to(self.device)
            data["rays_d_seen"] = data["rays_d_seen"].to(self.device)
            data["depth_seen"] = data["depth_seen"].to(self.device)
            data["pose_seen"] = data["pose_seen"].to(self.device)

        data["patch_img_id"] = data["patch_img_id"].to(self.device)
        data["patch_img_x"] = data["patch_img_x"].to(self.device)
        data["patch_img_y"] = data["patch_img_y"].to(self.device)
        data["patch_img"] = data["patch_img"].to(self.device)
        data["patch_mono_depth"] = data["patch_mono_depth"].to(self.device)

        return data

    def _normalize_err(self, preds: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        err = torch.abs(preds - gt)
        err = err.mean(-1, keepdim=True)  # mean over channels
        # normalize between 0, 1 where 1 corresponds to the 90th percentile
        # err = err.clamp_max(torch.quantile(err, 0.9))
        err = self._normalize_01(err)
        return err.repeat(1, 1, 3)

    @staticmethod
    def _normalize_01(t: torch.Tensor) -> torch.Tensor:
        return (t - t.min()) / (t.max()-t.min())

    def _normalize_depth(self, depth: torch.Tensor, img_h: int, img_w: int) -> torch.Tensor:
        return (
            self._normalize_01(depth)
        ).cpu().reshape(img_h, img_w)[..., None]

    def psnr_to_mse(self,psnr):
        return torch.exp(-0.1*torch.log(torch.tensor(10.))*psnr)

    def compute_avg_error(self,psnr,ssim,lpips):
        mse = self.psnr_to_mse(psnr)
        dssim = torch.sqrt(1-ssim)
        return torch.exp(torch.mean(torch.log(torch.tensor([mse,dssim,lpips]))))

    def calc_metrics(self, preds: torch.Tensor, gt: torch.Tensor):

        if gt.shape[-1] == 4:
            gt = gt[..., :3] * gt[..., 3:] + (1.0 - gt[..., 3:])

        err = (gt - preds) ** 2


        psnr = metrics.psnr(preds, gt)
        ssim = metrics.ssim(preds, gt)

        lpips = self.lpips_alex(preds.permute(2, 0, 1).unsqueeze(0), gt.permute(2, 0, 1).unsqueeze(0), normalize=True)[0]

        avg_metric = self.compute_avg_error(torch.tensor(psnr),torch.tensor(ssim),lpips[0,0,0])


        return {
            "mse": torch.mean(err),
            "psnr": metrics.psnr(preds, gt),
            "ssim": metrics.ssim(preds, gt),
            "ms-ssim": metrics.msssim(preds, gt),
            "lpips": lpips[0,0,0],
            "avg_metric": avg_metric,
        }

    def evaluate_metrics(self,
                         gt: Optional[torch.Tensor],
                         preds: MutableMapping[str, torch.Tensor],
                         dset,
                         img_idx: int,
                         name: Optional[str] = None,
                         save_outputs: bool = True,
                         is_train = False) -> Tuple[dict, np.ndarray, Optional[np.ndarray]]:
        if isinstance(dset.img_h, int):
            img_h, img_w = dset.img_h, dset.img_w
        else:
            img_h, img_w = dset.img_h[img_idx], dset.img_w[img_idx]
        preds_rgb = (
            preds["rgb"]
            .reshape(img_h, img_w, 3)
            .cpu()
            .clamp(0, 1)
        )
        if not torch.isfinite(preds_rgb).all():
            log.warning(f"Predictions have {torch.isnan(preds_rgb).sum()} NaNs, "
                        f"{torch.isinf(preds_rgb).sum()} infs.")
            preds_rgb = torch.nan_to_num(preds_rgb, nan=0.0)
        out_img = preds_rgb
        summary = dict()

        out_depth = None
        if "depth" in preds:
            out_depth = preds["depth"].cpu().reshape(img_h, img_w)[..., None]
        preds.pop("depth")


        if "scenes" in dset.datadir:
            depth_gt = dset.depths_all[img_idx].unsqueeze(dim=2)
            mask_gt = depth_gt>0
            mask_pred = out_depth>0
            mask = torch.logical_and(mask_gt,mask_pred)

            _depth_gt = depth_gt[mask]
            _out_depth = out_depth[mask]

            errors =[*compute_errors(_depth_gt.numpy(), _out_depth.numpy())]

        if gt is not None:
            gt = gt.reshape(img_h, img_w, -1).cpu()
            if gt.shape[-1] == 4:
                gt = gt[..., :3] * gt[..., 3:] + (1.0 - gt[..., 3:])
            summary.update(self.calc_metrics(preds_rgb, gt))
            out_img = torch.cat((out_img, gt), dim=0)
            out_img = torch.cat((out_img, self._normalize_err(preds_rgb, gt)), dim=0)

        out_img_np: np.ndarray = (out_img * 255.0).byte().numpy()
        out_depth_np: Optional[np.ndarray] = None
        if out_depth is not None:
            out_depth = out_depth*1000
            out_depth_np = out_depth.detach().cpu().squeeze().numpy().astype(np.uint16)
            # out_depth = self._normalize_01(out_depth)
            # out_depth_np = (out_depth * 255.0).repeat(1, 1, 3).byte().numpy()

        if save_outputs:
            out_name = f"step{self.global_step}-{img_idx}"
            if name is not None and name != "":
                out_name += "-" + name
            if is_train:
                write_png(os.path.join(self.log_dir, "train" + out_name + ".png"), out_img_np)
            else:
                write_png(os.path.join(self.log_dir, out_name + ".png"), out_img_np)
            if out_depth is not None:
                depth_name = out_name + "-depth"
                if is_train:
                    write_png(os.path.join(self.log_dir,"train" + out_name + ".png"), out_img_np)
                else:
                    write_png(os.path.join(self.log_dir, out_name + ".png"), out_img_np)

                write_png(os.path.join(self.log_dir, depth_name + ".png"), out_depth_np)
        if "scenes" in dset.datadir:
            return summary,out_img_np,out_depth_np,errors
        
        return summary, out_img_np, out_depth_np

    @abc.abstractmethod
    def validate(self):
        pass

    def report_test_metrics(self, scene_metrics: Dict[str, Sequence[float]], extra_name: Optional[str]):
        log_text = f"step {self.global_step}/{self.num_steps}"
        if extra_name is not None:
            log_text += f" | {extra_name}"
        scene_metrics_agg: Dict[str, float] = {}
        for k in scene_metrics:
            ak = f"{k}_{extra_name}" if extra_name is not None else f"        {k}         "
            scene_metrics_agg[ak] = np.mean(np.asarray(scene_metrics[k])).item()
            log_text += f" | {k}: {scene_metrics_agg[ak]:.4f}"
            self.writer.add_scalar(f"test/{ak}", scene_metrics_agg[ak], self.global_step)

        log.info(log_text)
        return scene_metrics_agg

    def get_save_dict(self):
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "global_step": self.global_step,
            "scale": self.dpt_scale,
            "shift": self.dpt_shift,
        }

    def save_model(self):
        model_fname = os.path.join(self.log_dir, f'model_{self.global_step}.pth')
        log.info(f'Saving model checkpoint to: {model_fname}')
        torch.save(self.get_save_dict(), model_fname)
        if self.DPT_adaptor:
            dpt_fname = os.path.join(self.log_dir,f'dpt_{self.global_step}.pth')
            torch.save(self.adaptor_DPT.state_dict(),dpt_fname)
    def load_model(self, checkpoint_data, training_needed: bool = True):
        self.model.load_state_dict(checkpoint_data["model"], strict=False)
        log.info("=> Loaded model state from checkpoint")

        if training_needed:
            self.optimizer.load_state_dict(checkpoint_data["optimizer"])
            log.info("=> Loaded optimizer state from checkpoint")

        if training_needed and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint_data['lr_scheduler'])

            log.info("=> Loaded scheduler state from checkpoint")

        self.global_step = checkpoint_data["global_step"]

        log.info(f"=> Loaded step {self.global_step} from checkpoints")

    def load_model_only(self, checkpoint_data, training_needed: bool = True):
        self.model.load_state_dict(checkpoint_data["model"], strict=False)
        log.info("=> Loaded model state from checkpoint")

        if training_needed:
            self.optimizer.load_state_dict(checkpoint_data["optimizer"])
            log.info("=> Loaded optimizer state from checkpoint")

    @abc.abstractmethod
    def init_epoch_info(self) -> Dict[str, EMA]:
        pass

    # noinspection PyUnresolvedReferences,PyProtectedMember
    def init_lr_scheduler(self, **kwargs) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        eta_min = 0
        lr_sched = None
        max_steps = self.num_steps
        scheduler_type = kwargs['scheduler_type']
        log.info(f"Initializing LR Scheduler of type {scheduler_type} with "
                 f"{max_steps} maximum steps.")
        if scheduler_type == "cosine":
            lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max_steps,
                eta_min=eta_min)
        elif scheduler_type == "warmup_cosine":
            lr_sched = get_cosine_schedule_with_warmup(
                self.optimizer, num_warmup_steps=512, num_training_steps=max_steps)
        elif scheduler_type == "step":
            lr_sched = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=[
                    max_steps // 2,
                    max_steps * 3 // 4,
                    max_steps * 5 // 6,
                    max_steps * 9 // 10,
                ],
                gamma=0.33)
        elif scheduler_type == "warmup_step":
            lr_sched = get_step_schedule_with_warmup(
                self.optimizer, milestones=[
                    max_steps // 2,
                    max_steps * 3 // 4,
                    max_steps * 5 // 6,
                    max_steps * 9 // 10,
                ],
                gamma=0.33,
                num_warmup_steps=512)
        return lr_sched

    def init_optim(self, **kwargs) -> torch.optim.Optimizer:
        optim_type = kwargs['optim_type']
        adaptor_lr = 1e-4 if 'adaptor_lr' not in kwargs else kwargs['adaptor_lr']
        if optim_type == 'adam':
            parameter = self.model.get_params(kwargs['lr'])
            if self.DPT_adaptor:
                dpt_params = {k:v for k,v in self.adaptor_DPT.named_parameters(prefix='scratch') if 'refinenet' in k and "out_conv" in k}
                dpt_params = list(dpt_params.values())
                parameter.append({"params":dpt_params,"lr":kwargs['lr']*adaptor_lr})
                parameter.append({"params":[self.depth_scale,self.depth_shift],"lr": kwargs['lr']*adaptor_lr})
            optim = torch.optim.Adam(params=parameter, eps=1e-15)
        else:
            raise NotImplementedError()
        return optim

    @abc.abstractmethod
    def init_model(self, **kwargs) -> torch.nn.Module:
        pass

    def get_regularizers(self, **kwargs) -> Sequence[Regularizer]:
        return ()

    def init_regularizers(self, **kwargs):
        # Keep only the regularizers with a positive weight
        regularizers = [r for r in self.get_regularizers(**kwargs) if r.weight > 0]
        return regularizers

    @property
    def lr(self):
        return self.optimizer.param_groups[0]['lr']

    @property
    def calc_metrics_every(self):
        return 1


def losses_to_postfix(loss_dict: Dict[str, EMA], lr: Optional[float]) -> str:
    pfix = [f"{lname}={lval}" for lname, lval in loss_dict.items()]
    if lr is not None:
        pfix.append(f"lr={lr:.2e}")
    return "  ".join(pfix)


def init_dloader_random(_):
    seed = torch.initial_seed() % 2**32  # worker-specific seed initialized by pytorch
    np.random.seed(seed)
    random.seed(seed)


def initialize_model(
        runner: 'StaticTrainer',
        **kwargs) -> LowrankModel:
    """Initialize a `LowrankModel` according to the **kwargs parameters.

    Args:
        runner: The runner object which will hold the model.
                Needed here to fetch dataset parameters.
        **kwargs: Extra parameters to pass to the model

    Returns:
        Initialized LowrankModel.
    """
    extra_args = copy(kwargs)
    extra_args.pop('global_scale', None)
    extra_args.pop('global_translation', None)

    dset = runner.test_dataset
    try:
        global_translation = dset.global_translation
    except AttributeError:
        global_translation = None
    try:
        global_scale = dset.global_scale
    except AttributeError:
        global_scale = None

    num_images = None
    if runner.train_dataset is not None:
        try:
            num_images = runner.train_dataset.num_images
        except AttributeError:
            num_images = None
    else:
        try:
            num_images = runner.test_dataset.num_images
        except AttributeError:
            num_images = None
    model = LowrankModel(
        grid_config=extra_args.pop("grid_config"),
        aabb=dset.scene_bbox,
        is_ndc=dset.is_ndc,
        is_contracted=dset.is_contracted,
        global_scale=global_scale,
        global_translation=global_translation,
        use_appearance_embedding=False,
        num_images=num_images,
        **extra_args)
    log.info(f"Initialized {model.__class__} model with "
             f"{sum(np.prod(p.shape) for p in model.parameters()):,} parameters, "
             f"using ndc {model.is_ndc} and contraction {model.is_contracted}. "
             f"Linear decoder: {model.linear_decoder}.")
    return model

def calculate_scale_shift(pseudo_depth, gt_depth):
    a = pseudo_depth.flatten().cpu().numpy()
    b = gt_depth.flatten().cpu().numpy()

    z = np.polyfit(a,b,1)
    scale, shift = z[0], z[1]

    return scale, shift

def warp_visual(rgb_fake,rgb_seen_render,depth_rgb,
                depth_seen, depth_fake, warped_depth, warp_mask, novel_loss_mask, dir, step):
    
    # if step > 50:
    #     exit()
    # import torchvision
    torchvision.utils.save_image(rgb_fake, f"./{dir}/{step}_novel_rgb.png")
    torchvision.utils.save_image(rgb_seen_render, f"./{dir}/{step}_seen_rgb.png")
    
    torchvision.utils.save_image(depth_rgb, f"./{dir}/{step}_novel_depth.png", normalize=True)
    torchvision.utils.save_image(depth_seen, f"./{dir}/{step}_seen_depth.png", normalize=True)
    torchvision.utils.save_image(depth_fake, f"./{dir}/{step}_novel_render_depth.png", normalize=True)
    torchvision.utils.save_image(warped_depth, f"./{dir}/{step}_warped_depth.png", normalize=True)
    torchvision.utils.save_image(((warped_depth-depth_rgb)**2)*warp_mask, f"./{dir}/{step}_error.png", normalize=True)
    torchvision.utils.save_image(novel_loss_mask.float(), f"./{dir}/{step}_novel_loss_mask.png", normalize=True)

    import pdb; pdb.set_trace()
    return None