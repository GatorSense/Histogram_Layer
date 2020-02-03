# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:54:23 2019

@author: jpeeples
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:54:23 2019

@author: jpeeples
"""
import cv2
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
from PIL import Image
import os

import imageio
import sys

#Code sourced from: https://gist.github.com/michaelosthege/cd3e0c3c556b70a79deba6855deb2cc8
class TargetFormat(object):
    GIF = ".gif"
    MP4 = ".mp4"
    AVI = ".avi"

def convertFile(inputpath, targetFormat):
    """Reference: http://imageio.readthedocs.io/en/latest/examples.html#convert-a-movie"""
    outputpath = os.path.splitext(inputpath)[0] + targetFormat

    reader = imageio.get_reader(inputpath)
    fps = reader.get_meta_data()['fps']

    writer = imageio.get_writer(outputpath, fps=fps)
    for i,im in enumerate(reader):
        sys.stdout.write("\rframe {0}".format(i))
        sys.stdout.flush()
        writer.append_data(im)
    writer.close()

def Generate_hist_vid(directory):
    #Video parameters
    size = (640,480)
    fps = 7
    
    def fig2data ( fig ):
        """
        @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
        @param fig a matplotlib figure
        @return a numpy 3D array of RGBA values
        """
        # draw the renderer
        fig.canvas.draw ( )
     
        # Get the RGBA buffer from the figure
        w,h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = ( w, h,4 )
     
        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        buf = np.roll( buf, 3, axis = 2 )
        return buf
    
     
    def fig2img ( fig ):
        """
        @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
        @param fig a matplotlib figure
        @return a Python Imaging Library ( PIL ) image
        """
        # put the figure pixmap into a numpy array
        buf = fig2data ( fig )
        w, h, d = buf.shape
        return Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )
    # Parse through files and plot results
    sub_dir = directory
    
    # Parameters of Histogram Layer
    #Load files and initialize array for images
    saved_bins = np.load(sub_dir+'Saved_bins.npy')
    saved_widths = np.load(sub_dir+'Saved_widths.npy')
    frame_array_img = []
    for epoch in range(0,len(saved_bins[:,0])):
        Bin_centers = saved_bins[epoch,:]
        widths = saved_widths[epoch,:]
    
        # Plot histogram bins and centers with toy data
        fig3 = plt.figure()
        for ii in range(0, len(Bin_centers)):
            toy_data = np.linspace(Bin_centers[ii] - 6 * abs(widths[ii]).reshape(-1), Bin_centers[ii] + 6 * abs(widths[ii]).reshape(-1), 300)
            #Function does not square widths, so apply absolute value widths
            plt.plot(toy_data, scipy.stats.norm.pdf(toy_data, Bin_centers[ii], abs(widths[ii].reshape(-1))))
    
        title = str('Histogram Learned for Epoch %d' % epoch)
        plt.suptitle(title)
        plt.xlabel('x')
        plt.ylabel('F(x)')
#        legend_text = []
#        for histBin in range(0,len(Bin_centers)):
#            legend_text.append('Bin '+ str(histBin+1))
#        plt.legend(legend_text, loc='upper right')
        plt.close()
        #pdb.set_trace()
        temp_image = fig2img(fig3)
        frame_array_img.append(temp_image)
        plt.close()
        
    #Create video
    pathOut = sub_dir + 'Histogram_Layer_Vid'
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(f"{pathOut}.mp4",fourcc,fps,size,0)

    #pdb.set_trace()
    print('Creating Video...')
    for i in range(len(frame_array_img)):
        out.write(np.array(frame_array_img[i]))
    out.release()
    
    convertFile(pathOut+".mp4", TargetFormat.GIF)

    

    
