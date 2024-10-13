from skimage.feature import graycomatrix, graycoprops
from BiT import bio_taxo
from mahotas.features import haralick
import cv2
import numpy as np
 
 
def haralick_feat(data):
    return haralick(data).mean(0).tolist()    
 
def haralick_feat_beta(image_path):
    data = cv2.imread(image_path, 0)
    return haralick(data).mean(0).tolist()
   
def glcm(data):
   
    co_matrix = graycomatrix(data, [1], [np.pi/4], None,symmetric=False, normed=False )
    dissimilarity = graycoprops(co_matrix, 'dissimilarity')[0, 0]
    cont = graycoprops(co_matrix, 'contrast')[0, 0]
    corr = graycoprops(co_matrix, 'correlation')[0, 0]
    ener = graycoprops(co_matrix, 'energy')[0, 0]
    asm = graycoprops(co_matrix, 'ASM')[0, 0]
    homo = graycoprops(co_matrix, 'homogeneity')[0, 0]
    return [dissimilarity, cont, corr, ener, asm, homo]
 
def glcm_beta(image_path):
    data = cv2.imread(image_path, 0)
    co_matrix = graycomatrix(data, [1], [np.pi/4], None,symmetric=False, normed=False )
    dissimilarity = graycoprops(co_matrix, 'dissimilarity')[0, 0]
    cont = graycoprops(co_matrix, 'contrast')[0, 0]
    corr = graycoprops(co_matrix, 'correlation')[0, 0]
    ener = graycoprops(co_matrix, 'energy')[0, 0]
    asm = graycoprops(co_matrix, 'ASM')[0, 0]
    homo = graycoprops(co_matrix, 'homogeneity')[0, 0]
    return [dissimilarity, cont, corr, ener, asm, homo]
 
def bitdesc(data):
    if data is None or data.size == 0:
        print("Error: Empty image data provided to bitdesc.")
        return []
    return bio_taxo(data)
 
def bitdesc_(image_path):
    data = cv2.imread(image_path, 0)
    return bio_taxo(data)
 
def bit_glcm_haralick(data):
    return bitdesc(data) + glcm(data) + haralick_feat(data)