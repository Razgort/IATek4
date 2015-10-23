import numpy as np
from skimage.color import rgb2gray
from skimage import data
from skimage import io
from skimage.transform import resize


def getImageArray(imagepath, units):
    try:
        img = io.imread(imagepath)
        img = resize(img, (units, units))
        img_gray = rgb2gray(img)       
        return img_gray
    except IOError:
        print("File "+ imagepath +" not found !")
        exit()
        return np.zeros((10, 10))


def readLines(path):
    arr = []
    try:
        f = open(path, 'r')
        for l in f:
            arr.append(l.replace("\n", ''))
        f.close()
    except IOError:
        print("File not found !")
        exit()
    return arr

def getImages(files):
    result = []
    for f in files:
        result.append(getImageArray(f, 50))
    return result