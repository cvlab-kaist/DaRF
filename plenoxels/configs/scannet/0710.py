config = {
 'expname': '0710',
 'logdir': './output/0710',
 'device': 'cuda:0',

 'data_downsample': 1.0,
 'data_dirs': ['./data/scenes/scene0710_00'],
 'checkpoint' : '.',
 'contract': False,
 'ndc': False,

 'near' : 0.1, 
 'far' : 4.04, 
 'radius' : 1.5,
 'fewshot' : None,
 'patch_rendering' : 64,

 'DPT_adaptor': True,
 'novel_detach': True,
 'adaptor_weight': 0.1,
 'adaptor_lr': 1e-4,
  

 'depth_loss' : 'l1_ranking',
 'depth_loss_range' : [0,25001],
 'warp' : True,
 'depth_weight' : 0.1,
 'ranking_step': 1000,

 'fake_img': '_novel_test',
 'stride': 3,
 'novel_patch_size': 128,
 'dpt_weight_path': 'PRETRAINED_MiDaS_PATH',

 'novel_depth_loss': 0.01,
 'novel_depth_loss_function': 'scale_shift', 
 'novel_depth_loss_range': [5001,25000],
 'convention': "XYZ",



 # Optimization settings
 'num_steps': 25001,
 'batch_size': 4096,          # important if patch_rendering
 'scheduler_type': 'warmup_cosine',
 'optim_type': 'adam',
 'lr': 0.01,

 # Regularization
 'plane_tv_weight': 0.01,
 'plane_tv_weight_proposal_net': 0.0001,
 'histogram_loss_weight': 1.0,
 'distortion_loss_weight': 0.001,

 # Training settings
 'save_every': 5000,
 'valid_every': 5000,
 'save_outputs': True,
 'train_fp16': True,

 # Raymarching settings
 'single_jitter': False,
 'num_samples': 48,
 # proposal sampling
 'num_proposal_samples': [256, 128],
 'num_proposal_iterations': 2,
 'use_same_proposal_network': False,
 'use_proposal_weight_anneal': True,
 'proposal_net_args_list': [
   {'num_input_coords': 3, 'num_output_coords': 8, 'resolution': [64, 64, 64]},
   {'num_input_coords': 3, 'num_output_coords': 8, 'resolution': [128, 128, 128]}
 ],

 # Model settings
 'multiscale_res': [1, 2, 4, 8],
 'density_activation': 'trunc_exp',
 'concat_features_across_scales': True,
 'linear_decoder': False,
 'grid_config': [{
   'grid_dimensions': 2,
   'input_coordinate_dim': 3,
   'output_coordinate_dim': 32,
   'resolution': [64, 64, 64]
 }],
}
