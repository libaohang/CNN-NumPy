import numpy as np

class MaxPoolingLayer:
    def __init__(self, filterSize):
        self.filterSize = filterSize

    def forward(self, image):
        self.image = image
        height, width, numFilters = image.shape
        outHeight = height // self.filterSize
        outWidth = width // self.filterSize

        #Organize the image into outHeight*outWidth matrix with each index being a vector of filterSize*filterSize pixels
        patches = image.reshape(outHeight, self.filterSize, outWidth, self.filterSize, numFilters)
        patches = patches.transpose(0, 2, 1, 3, 4)
        patches = patches.reshape(outHeight, outWidth, self.filterSize ** 2, numFilters)
        self.patches = patches

        #Get the max of each of each filterSize*filterSize pixel patches
        patchesMax = patches.max(axis=2)

        return patchesMax
    
    def backward(self, dE_dY):
        height, width, numFilters = self.image.shape
        outHeight = height // self.filterSize
        outWidth = width // self.filterSize

        #Shape of patchesMax is (outHeight, outWidth, 1, numFilters)
        patchesMax = self.patches.max(axis=2,keepdims=True)

        #Make a mask to select indicies where the pixel is max of that patch. 3rd dimension broadcasts to filterSize*filterSize
        mask = (patchesMax == self.patches)

        #Add axis to make it broadcast with mask: (outHeight, outWidth, numFilters) -> (outHeight, outWidth, 1, numFilters)
        dE_dY = dE_dY[:, :, None, :]

        #For each filterSize*filterSize pixel of each patch, only the ones with max values gets the corresponding gradient of that patch
        dE_dX = dE_dY * mask

        #Reshape into original shape
        dE_dX = dE_dX.reshape(outHeight, outWidth, self.filterSize, self.filterSize, numFilters)
        dE_dX = dE_dX.transpose(0, 2, 1, 3, 4)
        dE_dX = dE_dX.reshape(height, width, numFilters)

        return dE_dX
    
