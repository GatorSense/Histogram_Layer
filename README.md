# Histogram Layer:
**Histogram Layers for Texture Analysis**

_Joshua Peeples, Weihuang Xu, and Alina Zare_

Note: If this code is used, cite it: Joshua Peeples, Weihuang Xu, & Alina Zare. 
(2020, March 21). GatorSense/Histogram_Layer: Initial Release (Version v1.0). 
Zendo. https://doi.org/10.5281/zenodo.3722786 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3722786.svg)](https://doi.org/10.5281/zenodo.3722786)

[[`arXiv`](https://arxiv.org/abs/2001.00215)]

[[`BibTeX`](#CitingHist)]


In this repository, we provide the paper and code for histogram layer models from "Histogram Layers for Texture Analysis."

## Installation Prerequisites

This code uses python, pytorch, and barbar. 
Please use [[`Pytorch's website`](https://pytorch.org/get-started/locally/)] to download necessary packages.
Barbar is used to show the progress of model. Please follow the instructions [[`here`](https://github.com/yusugomori/barbar)]
to download the module.

## Demo

Run `Demo.py` in Python IDE (e.g., Spyder) or command line. To evaluate performance,
run `View_Results.py` (if results are saved out).

## Main Functions

The histogram layer model (HistRes_B) runs using the following functions. 

1. Intialize model  

```model, input_size = intialize_model(**Parameters)```

2. Prepare dataset(s) for model

 ```dataloaders_dict = Prepare_Dataloaders(**Parameters)```

3. Train model 

```train_dict = train_model(**Parameters)```

4. Test model

```test_dict = test_model(**Parameters)```


## Parameters
The parameters can be set in the following script:

```Demo_Parameters.py```

## Inventory

```
https://github.com/GatorSense/Histogram_Layer

└── root dir
    ├── Demo.py   //Run this. Main demo file.
    ├── Demo_Parameters.py // Parameters file for demo.
    ├── Prepare_Data.py  // Load data for demo file.
    ├── Prepare_Data_Results.py // Load data for results file.
    ├── Texture_Information.py // Class names and directories for datasets
    ├── View_Results.py // Run this after demo to view saved results
    ├── View_Results_Parameters.py // Parameters file for results.
    ├── papers  // Related publications
    │   ├── peeples2020histogram.pdf
    └── util  //utility functions
        ├── Compute_FDR.py  // Compute Fisher Discriminant Ratio for features
        ├── Confusion_mats.py  // Generate confusion matrices
        ├── Generate_TSNE_visual.py  // Generate TSNE visualization for features
        ├── Histogram_Model.py  // Generate HistRes_B models
        ├── Network_functions.py  // Contains functions to initialize, train, and test model 
        ├── RBFHistogramPooling.py  // Create histogram layer 
        ├── Save_Results.py  // Save results from demo script
     
```

## License

This source code is licensed under the license found in the [`LICENSE`](LICENSE) file in the root directory of this source tree.

This product is Copyright (c) 2020 J. Peeples, W. Xu, and A. Zare. All rights reserved.

## <a name="CitingHist"></a>Citing Histogram Layer

If you use the histogram layer code, please cite the following reference using the following entry.

**Plain Text:**

Peeples, J., Xu, W., & Zare, A. (2020). Histogram Layers for Texture Analysis. arXiv preprint arXiv:2001.00215.

**BibTex:**
```
@article{peeples2020histogram,
  title={Histogram Layers for Texture Analysis},
  author={Peeples, Joshua and Xu, Weihuang and Zare, Alina},
  journal={arXiv preprint arXiv:2001.00215},
  year={2020}
}
```

