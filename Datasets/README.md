# Downloading datasets:

Note: Due to the size of the datasets, the images were not 
upload to the repository. Please follow the following instructions
to ensure the code works. If any of these datasets are used,
please cite the appropiate sources (papers, repositories, etc.) as mentioned
on the webpages and provided here.

##  Describable Texture Dataset (DTD) [[`BibTeX`](#CitingDTD)]

Please download the [`DTD dataset`](https://www.robots.ox.ac.uk/~vgg/data/dtd/) 
and follow these instructions:

1. Download and unzip the file
2. Name the folder `DTD`
3. The structure of the `DTD` folder is as follows:
```
└── root dir
    ├── images   // Contains folders of images for each class.
    ├── imdb // Not used.
    ├── labels  // Contains training,validation, and test splits.   
```
## <a name="CitingDTD"></a>Citing DTD

If you use the DTD dataset, please cite the following reference using the following entry.

**Plain Text:**

Cimpoi, M., Maji, S., Kokkinos, I., Mohamed, S., & Vedaldi, A. (2014). 
Describing textures in the wild. In Proceedings of the IEEE Conference on 
Computer Vision and Pattern Recognition (pp. 3606-3613).

**BibTex:**
```
@InProceedings{cimpoi14describing,
	     Author    = {M. Cimpoi and S. Maji and I. Kokkinos and S. Mohamed and and A. Vedaldi},
	     Title     = {Describing Textures in the Wild},
	     Booktitle = {Proceedings of the {IEEE} Conf. on Computer Vision and Pattern Recognition ({CVPR})},
	     Year      = {2014}}
```
## Extension of the Ground Terrain in Outdoor Scenes (GTOS-mobile) [[`BibTeX`](#CitingGTOS_m)]

Please download the 
[`GTOS-mobile dataset`](https://github.com/jiaxue1993/Deep-Encoding-Pooling-Network-DEP-) 
and follow these instructions:

1. Download and unzip the file
2. Name the folder `gtos-mobile`
3. The structure of the `gtos-mobile` folder is as follows:
```
└── root dir
    ├── test   // Contains folders of test images for each class.
    ├── train // Contains folders of training images for each class.  
```
## <a name="CitingGTOS_m"></a>Citing GTOS-mobile

If you use the GTOS-mobile dataset, please cite the following reference using the following entry.

**Plain Text:**

Xue, J., Zhang, H., & Dana, K. (2018). Deep texture manifold for ground 
terrain recognition. In Proceedings of the IEEE Conference on Computer Vision 
and Pattern Recognition (pp. 558-567).

**BibTex:**
```
@inproceedings{xue2018deep,
  title={Deep texture manifold for ground terrain recognition},
  author={Xue, Jia and Zhang, Hang and Dana, Kristin},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={558--567},
  year={2018}
}
```
## Subset of Material in Context (MINC-2500) [[`BibTeX`](#CitingMINC)]

Please download the 
[`MINC-2500 dataset`](http://opensurfaces.cs.cornell.edu/publications/minc/) 
and follow these instructions:

1. Download and unzip the file
2. Name the folder `minc-2500`
3. The structure of the `minc-2500` folder is as follows:
```
└── root dir
    ├── images   // Contains folders of images for each class.
    ├── labels // Contains training,validation, and test splits.
    ├── categories.txt  // Class names for dataset
    ├── README.txt  // README file from MINC-2500 curators
       
```
## <a name="CitingMINC"></a>Citing MINC-2500

If you use the MINC-2500 dataset, please cite the following reference using the following entry.

**Plain Text:**

Bell, S., Upchurch, P., Snavely, N., & Bala, K. (2015). Material recognition 
in the wild with the materials in context database. In Proceedings of the IEEE 
conference on computer vision and pattern recognition (pp. 3479-3487).

**BibTex:**
```
@inproceedings{bell2015material,
  title={Material recognition in the wild with the materials in context database},
  author={Bell, Sean and Upchurch, Paul and Snavely, Noah and Bala, Kavita},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={3479--3487},
  year={2015}
}
```