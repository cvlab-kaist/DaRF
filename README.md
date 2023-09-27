# DäRF: Boosting Radiance Fields from Sparse Inputs with Monocular Depth Adaptation
<a href="https://arxiv.org/abs/2305.19201"><img src="https://img.shields.io/badge/arXiv-2305.19201-%23B31B1B"></a>
<a href="https://ku-cvlab.github.io/DaRF/ "><img src="https://img.shields.io/badge/Project%20Page-online-brightgreen"></a>
<br>

<center>
<img src="https://github.com/KU-CVLAB/DaRF/blob/gh-pages/img/Scannet_standard_qual.png" width="100%" height="100%"> 
</center>

This is official implementation of the paper "DäRF: Boosting Radiance Fields from Sparse Inputs with Monocular Depth Adaptation".

## Introduction
<center>
<img src="https://github.com/KU-CVLAB/DaRF/blob/gh-pages/img/main.png" width="100%" height="100%"> 
</center>

Unlike existing work (SCADE [CVPR'23]) that distills depths by pretrained MDE to NeRF at seen view only, our DäRF fully exploits the ability of MDE by jointly optimizing NeRF and MDE at a specific scene, and distilling the monocular depth prior to NeRF at both seen and unseen views. For more details, please visit our [project page](https://ku-cvlab.github.io/DaRF/)!

## TODO
- [ ] Reveal the pretrained-weight on Scannet
- [ ] TNT/in-the-wild datasets and dataloaders

## Installation
An example of installation is shown below:
```
git clone https://github.com/KU-CVLAB/DaRF.git
cd DaRF
conda create -n DaRF python=3.8
conda activate DaRF
pip install -r requirements.txt
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```
Also, you need to download pretrained MiDaS 3.0 weights(dpt_hybrid_384) on [here](https://github.com/isl-org/MiDaS).
And you should replace the 'dpt_pretrained_weight' part of the config file with the MiDaS pretrained weights path.

## Dataset Download
You can download Scannet Dataset on [here](https://koreaoffice-my.sharepoint.com/:f:/g/personal/seong0905_korea_ac_kr/EvJ6ob2gemdAtbUptJSzA2gBNfbgCeW2opH6DNGaLe3Odg?e=ew0TwH)
If you want to download data on a different path, you should replace the 'data_dirs' part of the config file with the donloaded dataset path.

## Training
* 18-20 view Training
```
PYTHONPATH='.' python plenoxels/main.py --config plenoxels/configs/07XX.py
```
* 9-10 view Training
```
PYTHONPATH='.' python plenoxels/main.py --config plenoxels/configs/07XX_few.py
```

## Evaluation / Rendering
If you want to Evaluation or Rendering, You need to replace the 'checkpoint' part of the config file with the trained weights path.
* 18-20 view Evalutaion
```
PYTHONPATH='.' python plenoxels/main.py --config plenoxels/configs/07XX.py --validate-only --load_model
```
* 18-20 view Rendering
```
PYTHONPATH='.' python plenoxels/main.py --config plenoxels/configs/07XX.py --render-only --load_model
```
* 9-10 view Evalutaion
```
PYTHONPATH='.' python plenoxels/main.py --config plenoxels/configs/07XX_few.py --validate-only --load_model
```
* 9-10 view Rendering
```
PYTHONPATH='.' python plenoxels/main.py --config plenoxels/configs/07XX_few.py --render-only --load_model
```

## Acknowledgements
This code heavily borrows from [K-planes](https://github.com/sarafridov/K-Planes).

## Citation
If you use this software package, please cite our paper:
```
@article{song2023d,
  title={D$\backslash$" aRF: Boosting Radiance Fields from Sparse Inputs with Monocular Depth Adaptation},
  author={Song, Jiuhn and Park, Seonghoon and An, Honggyu and Cho, Seokju and Kwak, Min-Seop and Cho, Sungjin and Kim, Seungryong},
  journal={arXiv preprint arXiv:2305.19201},
  year={2023}
}
```
