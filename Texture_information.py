# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 09:43:09 2019

@author: jpeeples
"""
import numpy as np

KTH_Class_names = np.array(['aluminum_foil', 'brown_bread', 
                            'corduroy', 'cork', 'cotton', 'cracker', 
                            'lettuce leaf', 'linen', 'white_bread', 
                            'wood', 'wool'])
DTD_Class_names = np.array(['banded', 'blotchy', 'braided', 'bubbly', 'bumpy',
                        'chequered', 'cobwebbed', 'cracked', 'crosshatched',
                        'crystalline', 'dotted', 'fibrous', 'flecked', 'freckled',
                        'frilly', 'gauzy', 'grid', 'grooved', 'honeycombed',
                        'interlaced', 'knitted', 'lacelike', 'lined', 'marbled',
                        'matted', 'meshed', 'paisley', 'perforated', 'pitted',
                        'pleated', 'polka−dotted', 'porous', 'potholed', 'scaly',
                        'smeared', 'spiralled', 'sprinkled', 'stained', 'stratified',
                        'striped', 'studded', 'swirly', 'veined', 'waffled', 'woven', 'wrinkled', 'zigzagged'])
GTOS_Class_names = np.array(['Painting','aluminum','asphalt','asphalt_puddle',
                             'asphalt_stone','brick','cement','cloth','dry_grass', 'dry_leaf',
                             'glass','grass','ice_mud','large_limestone', 'leaf',
                             'metal_cover','moss', 'mud', 'mud_puddle','paint_cover',
                             'painting_turf', 'paper', 'pebble', 'plastic', 'plastic_cover',
                             'root','rust_cover', 'sand', 'sandPaper','shale','small_limestone',
                             'soil', 'steel', 'stone_asphalt', 'stone_brick', 'stone_cement',
                             'stone_mud', 'turf', 'wood chips'])
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
#GTOS_mobile_Class_names = np.array(['aluminum','asphalt','brick','cement','cloth','dry_leaf',
#                        'glass','grass','large_limestone','leaf','metal_cover',
#                        'moss', 'paint_cover','painting','painting_turf','paper',
#                        'pebble','plastic','plastic_cover','root','sand','sandPaper',
#                        'shale','small_limestone','soil','steel','stone_asphalt',
#                        'stone_brick','stone_cement','turf','wood_chips'])

Class_names = {'KTH': KTH_Class_names, 'DTD': DTD_Class_names, 
               'GTOS': GTOS_Class_names, 'MINC_2500': MINC_Class_names, 
               'GTOS-mobile': GTOS_mobile_Class_names}
Data_dirs = {'KTH': './Datasets/KTH-TIPS2-b', 'DTD': './Datasets/DTD/', 
             'GTOS': './Datasets/GTOS/images', 'MINC_2500': './Datasets/minc-2500/',
             'GTOS-mobile': './Datasets/gtos-mobile'}


