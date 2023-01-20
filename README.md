# EPiC-GAN - Equivariant Point Cloud Generation for Particle Jets

Pytorch implementation of the EPiC-GAN introduced in *EPiC-GAN - Equivariant Point Cloud Generation for Particle Jets* ([arXiv:2301:08128]( http://arxiv.org/abs/2301.08128))

## Overview:

The EPiC-GAN - equivariant point cloud generative adversarial network - is used to generate permutation equivariant point clouds of variable cardinality. The use case study presented here explores the generation of particle jets in hadronic coordinates (pt, rapidity, angle phi) with variable particle multiplicity. 
We trained the EPiC-GANs on the [JetNet30](https://doi.org/10.5281/zenodo.6975118) and the [JetNet150](https://doi.org/10.5281/zenodo.6975117) gluon, light quark, and top datasets respectively.

## Generation:

Generate particle jets using the trained models from [arXiv:2301:08128](http://arxiv.org/abs/2301.08128) via:
```bash
python generate.py --n_points 30 --dataset_type jetnet_gluon --events 100_000
```
This will generate 100k gluon jets corresponding to the JetNet30 gluon dataset. Valid options for `--n_points` are `30` or `150` for JetNet30 / JetNet150 respectively. `--datasets_type` is either `jetnet_gluon`, `jetnet_quark`, or `jetnet_top`. The `--events` flag determines the number of generated jets.

## Training:

To train the EPiC-GAN, download the JetNet datasets from the above links. Additionally a pre-processing needs to be performed, which can be run via the notebook `dataset/JetNet_dataset_processing.ipynb`. Afterwards train the EPiC-GAN with the hyperparameters from the publication for 10 epochs via:
```bash
python start_training.py --n_points 150 --dataset_type jetnet_top --epochs 10
```
The options to set the JetNet type are the same as above. Further training flags can be found in the code. 
