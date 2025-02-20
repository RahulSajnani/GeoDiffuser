# GeoDiffuser: Geometry-Based Image Editing with Diffusion Models
*[Rahul Sajnani](https://rahulsajnani.github.io/), [Jeroen Vanbaar](https://www.linkedin.com/in/jeroenvanbaar), [Jie Min](https://umjcs.github.io/), [Kapil Katyal](https://kdk132.github.io/), [Srinath Sridhar](https://cs.brown.edu/people/ssrinath/)*

**Note: Code is still in maintenance. Releasing initial version.**

![3D Edit](./assets/1.gif)

## Abstract
___
The success of image generative models has enabled us to build methods that can edit images based on text or other user input. However, these methods are bespoke, imprecise, require additional information, or are limited to only 2D image edits. We present GeoDiffuser, a zero-shot optimization-based method that unifies common 2D and 3D image-based object editing capabilities into a single method. Our key insight is to view image editing operations as geometric transformations. We show that these transformations can be directly incorporated into the attention layers in diffusion models to implicitly perform editing operations. Our training-free optimization method uses an objective function that seeks to preserve object style but generate plausible images, for instance with accurate lighting and shadows. It also inpaints disoccluded parts of the image where the object was originally located. Given a natural image and user input, we segment the foreground object using SAM and estimate a corresponding transform which is used by our optimization approach for editing. GeoDiffuser can perform common 2D and 3D edits like object translation, 3D rotation, and removal. We present quantitative results, including a perceptual study, that shows how our approach is better than existing methods. Visit [this https URL](https://ivl.cs.brown.edu/research/geodiffuser.html) for more information.


## New Features & ToDo
___
- [x] Speed up code
- [x] Clean Code-base
- [ ] Release Editing Dataset

## Installation
___

```
# Creating the conda environment and loading it
conda env create -f ./GeoDiffuser/envs/torch2.1_environment.yml
conda activate GeoDiffuser
```

## Running GeoDiffuser
___
```
CUDA_VISIBLE_DEVICES=0 python3 geo_diff_ui.py
```


## Citation
___
```
@InProceedings{sajnani2024geodiffuser,
  title={GeoDiffuser: Geometry-Based Image Editing with Diffusion Models}, 
  author={Rahul Sajnani and Jeroen Vanbaar and Jie Min and Kapil Katyal and Srinath Sridhar},
  booktitle = {IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  month = {March},
  year={2025},
}
```




