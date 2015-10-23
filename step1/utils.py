import numpy as np

from skimage.color import rgb2gray
from skimage import io
from skimage.transform import resize


def getImageArray(imagepath, units):
    try:
        img = io.imread(imagepath)
        img = resize(img, (units, units))
        img_gray = rgb2gray(img)       
        return np.array(img_gray).flatten()
    except IOError:
        print("File "+ imagepath +" not found !")
        exit()

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

def getPosition(letter, result_array):
    new_array = np.zeros(len(result_array))
    index = result_array.index(letter)
    new_array[index] = 1
    return new_array