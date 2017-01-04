# Multi-Resolution CNNs for Large-Scale Scene Recognition
Here we provide the code and models for the following report:

    Knowledge Guided Disambiguation for Large-Scale Scene Classification with Multi-Resolution CNNs
    Limin Wang, Sheng Guo, Weilin Huang, Yuanjun Xiong, and Yu Qiao, 
    in arXiv:1610.01119, 2016

### Updates
- January 3rd, 2017
  * Release the models on the Places365 dataset and test code.

### Overview
We have made two efforts to exploit CNNs for large-scale scene recognition:
- We design a modular framework to capture multi-level visual information for scene understanding by training CNNs from different resolutions
- We propose a knowledge disambiguation strategy by using soft labels from extra networks to deal with the label ambiguity issue of scene recognition.

These two efforts are the core part of team "SIAT_MMLAB" for the following large-scale scene recogntion challenges.

|        Challenge    | Rank | Performance |
|:-------------------:|:--------------:|:--------------:|
| Places2 challenge 2015 |    2nd place   |    0.1736 top5-error   |
| Places2 challenge 2016 |    4th place   |    0.1042 top5-error   |
| LSUN challenge 2015 |    2nd place   |    0.9030 top1-accuracy   |
| LSUN challenge 2016 |    1st place   |    0.9161 top1-accuracy   |

### Places365 Models
We first release the learned models on the Places365 dataset.
- Models learned at resolution of 256 * 256

|        Model    |  Top5 Error Rate |
|:-------------------:|:--------------:|
| (A0) Normal BN-Inception |    0.143   |
| (A1) Normal BN-Inception + object networks |    0.141   |
| (A2) Normal BN-Inception + scene networks |    0.134   |

- Models learned at resolution of 384 * 384

|        Model    |  Top5 Error Rate |
|:-------------------:|:--------------:|
| (B0) Deeper BN-Inception |    0.140   |
| (B1) Deeper BN-Inception + object networks |    0.136   |
| (B2) Deeper BN-Inception + scene networks |    0.130   |

### Testing Code
We release the testing code on the Places365 validation dataset.

We also release a demo code to use our Places365 model as generic feature extraction and perform scene recognition on the MIT Indoor67 dataset.

### Training Code
The training code is based on our modified Caffe toolbox. It is a efficient parallel caffe with MPI implementation. Meanwhile, we implement a new kl-divergence loss layer for our knowledge disambiguation methods;

https://github.com/yjxiong/caffe/tree/kd

