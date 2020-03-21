# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 09:43:09 2019
Names and data directories for results script
@author: jpeeples
"""
import numpy as np

DTD_Class_names = np.array(['banded', 'blotchy', 'braided', 'bubbly', 'bumpy',
                        'chequered', 'cobwebbed', 'cracked', 'crosshatched',
                        'crystalline', 'dotted', 'fibrous', 'flecked', 'freckled',
                        'frilly', 'gauzy', 'grid', 'grooved', 'honeycombed',
                        'interlaced', 'knitted', 'lacelike', 'lined', 'marbled',
                        'matted', 'meshed', 'paisley', 'perforated', 'pitted',
                        'pleated', 'polka−dotted', 'porous', 'potholed', 'scaly',
                        'smeared', 'spiralled', 'sprinkled', 'stained', 'stratified',
                        'striped', 'studded', 'swirly', 'veined', 'waffled', 'woven', 'wrinkled', 'zigzagged'])
MINC_Class_names = np.array(['brick','carpet','ceramic','fabric','foliage','food',
                            'glass','hair','leather','metal','mirror','other',
                            'painted','paper','plastic','polishedstone','skin',
                            'sky','stone','tile', 'wallpaper', 'water', 'wood'])
GTOS_mobile_Class_names = np.array(['Painting','aluminum','asphalt','brick','cement','cloth','dry_leaf',
                        'glass','grass','large_limestone','leaf','metal_cover',
                        'moss', 'paint_cover','painting_turf','paper',
                        'pebble','plastic','plastic_cover','root','sand','sandPaper',
                        'shale','small_limestone','soil','steel','stone_asphalt',
                        'stone_brick','stone_cement','turf','wood_chips'])

Class_names = {'DTD': DTD_Class_names, 
               'MINC_2500': MINC_Class_names, 
               'GTOS-mobile': GTOS_mobile_Class_names}
Data_dirs = {'DTD': './Datasets/DTD/', 
             'MINC_2500': './Datasets/minc-2500/',
             'GTOS-mobile': './Datasets/gtos-mobile'}


