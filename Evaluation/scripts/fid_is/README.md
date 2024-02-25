## To calculate FID score
'''
python fid_is_score.py --mode FID path/to/dataset1 path/to/dataset2
'''

To run the evaluation on GPU, use the flag `--gpu N`, where `N` is the index of the GPU to use.

for example to use GPU 0, use `--gpu 0`

### Using different layers for feature maps

In difference to the official implementation, you can choose to use a different feature layer of the Inception network instead of the default `pool3` layer.
As the lower layer features still have spatial extent, the features are first global average pooled to a vector before estimating mean and covariance.

This might be useful if the datasets you want to compare have less than the otherwise required 2048 images.
Note that this changes the magnitude of the FID score and you can not compare them against scores calculated on another dimensionality.
The resulting scores might also no longer correlate with visual quality.

You can select the dimensionality of features to use with the flag `--dims N`, where N is the dimensionality of features.
The choices are:
- 64:   first max pooling features
- 192:  second max pooling features
- 768:  pre-aux classifier features
- 2048: final average pooling features (this is the default)

## To calculate IS score

'''
python fid_is_score.py --mode IS path/to/dataset1 path/to/dataset2
'''
