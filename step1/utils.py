import numpy as np
import Image as im

def getImageArray(imagepath, units):
  try:
    image = im.open(imagepath)
    imagearray = np.asarray(image.crop(image.getbbox()).resize((units, units)))
    return imagearray
  except IOError:
    print("File not found !")
    exit()
    return np.zeros((10,10))


def readLines(path):
  arr = []
  try:
    f = open(path, 'r')
    for l in f:
      arr.append(l)

    f.close()
  except IOError:
    print("File not found !")
    exit()
  return arr
