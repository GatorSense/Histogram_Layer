# MICI:
**Multiple Instance Choquet Integral for Classifier Fusion and Regression**

_Xiaoxiao Du and Alina Zare_

Note: If this code is used, cite it: Xiaoxiao Du, & Alina Zare. (2019, April 12). GatorSense/MICI: Initial Release (Version v1.0). Zenodo. http://doi.org/10.5281/zenodo.2638378  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2638378.svg)](https://doi.org/10.5281/zenodo.2638378)

[[`IEEEXplore (MICI Classifier Fusion paper)`](https://ieeexplore.ieee.org/document/7743905)]

[[`IEEEXplore (MICI Classifier Fusion and Regression paper)`](https://ieeexplore.ieee.org/document/8528500)]

[[`arXiv`](https://arxiv.org/abs/1803.04048)]

[[`BibTeX`](#CitingMICI)]


In this repository, we provide the papers and code for the Multiple Instance Choquet Integral (MICI) Classifier Fusion and/or Regression Algorithms.

## Installation Prerequisites

This code uses MATLAB Statistics and Machine Learning Toolbox,
MATLAB Optimization Toolbox and MATLAB Parallel Computing Toolbox.

## Demo

Run `demo_main.m` in MATLAB.

## Main Functions

The MICI Classifier Fusion and Regression Algorithm runs using the following functions.

1. MICI Classifier Fusion (noisy-or model) Algorithm  

```[measure, initialMeasure,Analysis] = learnCIMeasure_noisyor(TrainBags, TrainLabels, Parameters);```

2. MICI Classifier Fusion (min-max model) Algorithm

 ```[measure, initialMeasure,Analysis] = learnCIMeasure_minmax(TrainBags, TrainLabels, Parameters);```

3. MICI Classifier Fusion (generalized-mean model) Algorithm

```[measure, initialMeasure,Analysis] = learnCIMeasure_softmax(TrainBags, TrainLabels, Parameters);```

4. MICI Regression (MICIR) Algorithm

```[measure, initialMeasure,Analysis] = learnCIMeasure_regression(TrainBags, TrainLabels, Parameters);```


## Inputs

#The *TrainBags* input is a 1xNumTrainBags cell. Inside each cell, NumPntsInBag x nSources double -- Training bags data.

#The *TrainLabels* input is a 1xNumTrainBags double vector that takes values of "1" and "0" for two-class classfication problems -- Training labels for each bag.


## Parameters
The parameters can be set in the following function:

```[Parameters] = learnCIMeasureParams();```

The parameters is a MATLAB structure with the following fields:
1. nPop: size of population
2. sigma: sigma of Gaussians in fitness function
3. maxIterations: maximum number of iterations
4. eta: percentage of time to make small-scale mutation
5. sampleVar: variance around sample mean
6. mean: mean of CI in fitness function. This value is always set to 1 (or very close to 1) if the positive label is "1".
7. analysis: if ="1", save all intermediate results
8. p: the power coefficient for the generalized-mean function. Empirically, setting p(1) to a large postive number and p(2) to a large negative number works well.

*Parameters can be modified by users in [Parameters] = learnCIMeasureParams() function.*

## Inventory

```
https://github.com/GatorSense/MICI

└── root dir
    ├── demo_main.m   //Run this. Main demo file.
    ├── demo_data_cl.mat //Demo classification data
    ├── learnCIMeasureParams.m  //parameters function
    ├── papers  //related publications
    │   ├── MICI for Classifier Fusion.pdf
    |   └── MICI Classifier Fusion and Regression for Remote Sensing Applications.pdf
    └── util  //utility functions
        ├── ChoquetIntegral_g_MultiSources.m  //compute CI for multiple sources
        ├── computeci.c    //compute CI. *Need to run "mex computeci.c"*
        ├── ismember_findrow_mex.c  //find row index if vector A is part of a row in vector B.   *Need to run "mex ismember_findrow_mex.c"*
        ├── ismember_findrow_mex_my.m  // find row index if vector A is part of a row in vector B (uses above c code).
        ├── share.h  //global variable header to be used in computeci.c
        ├── learnCIMeasure_noisyor.m  //MICI Two-Class Classifier Fusion with noisy-or objective function
        ├── learnCIMeasure_noisyor_CountME1.m  //MICI Two-Class Classifier Fusion with noisy-or objective function, using ME optimization
        ├── learnCIMeasure_minmax.m  //MICI Two-Class Classifier Fusion with min-max objective function
        ├── learnCIMeasure_softmax.m  //MICI Two-Class Classifier Fusion with generalized-mean objective function
        ├── learnCIMeasure_regression.m  //MICI Regression
        ├── evalFitness_noisyor.m  //noisy-or fitness function
        ├── evalFitness_minmax.m  //min-max fitness function
        ├── evalFitness_softmax.m  //generalized-mean fitness function
        ├── evalFitness_reg.m  //regression fitness function
        ├── invcdf_TruncatedGaussian.m //compute inverse cdf for Truncated Gaussian
        ├── sampleMeasure.m //sample new measures
        ├── sampleMeasure_Above.m  //sampling a new measure from top-down.
        ├── sampleMeasure_Bottom.m  //sampling a new measure from bottom-up.
        ├── sampleMultinomial_mat.m  //sample from a multinomial distribution.
        ├── quadLearnChoquetMeasure_3Source.m  //code for CI-QP method, hard-coded for 3 sources
        ├── quadLearnChoquetMeasure_4Source.m  //code for CI-QP method, hard-coded for 4 sources
        ├── quadLearnChoquetMeasure_5Source.m  //code for CI-QP method, hard-coded for 5 sources
        └── quadLearnChoquetMeasure_MultiSource.m  //code for the CI-QP method (learn CI measures using quadratic programming) for multiple (>=3) sources


```

## License

This source code is licensed under the license found in the [`LICENSE`](LICENSE) file in the root directory of this source tree.

This product is Copyright (c) 2018 X. Du and A. Zare. All rights reserved.

## <a name="CitingMICI"></a>Citing MICI

If you use the MICI clasifier fusion and regression algorithms, please cite the following references using the following entries.

**Plain Text:**

X. Du and A. Zare, "Multiple Instance Choquet Integral Classifier Fusion and Regression for Remote Sensing Applications," in IEEE Transactions on Geoscience and Remote Sensing, vol. 57, no. 5, pp. 2741-2753, May 2019. doi: 10.1109/TGRS.2018.2876687

X. Du, A. Zare, J. M. Keller and D. T. Anderson, "Multiple Instance Choquet integral for classifier fusion," 2016 IEEE Congress on Evolutionary Computation (CEC), Vancouver, BC, 2016, pp. 1054-1061. doi: 10.1109/CEC.2016.7743905

**BibTex:**
```
@ARTICLE{du2018multiple,
author={X. Du and A. Zare},
journal={IEEE Transactions on Geoscience and Remote Sensing},
title={Multiple Instance Choquet Integral Classifier Fusion and Regression for Remote Sensing Applications},
year={2018},
volume={57},
number={5},
pages={2741-2753}, 
month={May},
doi={10.1109/TGRS.2018.2876687}
}
```
```
@INPROCEEDINGS{du2016multiple,
author={X. Du and A. Zare and J. M. Keller and D. T. Anderson},
booktitle={IEEE Congress on Evolutionary Computation (CEC)},
title={Multiple Instance Choquet integral for classifier fusion},
year={2016},
volume={},
number={},
pages={1054-1061},
doi={10.1109/CEC.2016.7743905},
month={July}
}
```

## <a name="Related Work"></a>Related Work

Also check out our Multiple Instance Multi-Resolution Fusion (MIMRF) algorithm for multi-resolution fusion!


[[`arXiv`](https://arxiv.org/abs/1805.00930)]

[[`GitHub Code Repository`](https://github.com/GatorSense/MIMRF)]
