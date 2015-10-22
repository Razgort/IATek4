#!/usr/bin/python

import pybrain
import Image as im
import numpy as np

def getImageArray(imagepath):
    try:
      image = im.open(imagepath)
      imagearray=np.asarray(image.crop(image.getbbox()).resize((10,10))).astype(float)
      return imagearray
    except IOError:
      print "File Not Found"
      return np.zeros((10,10))


def main():
  print("Starting Neural Net")
  print(getImageArray("data/a.bmp"))

if __name__ == "__main__":
  main()
