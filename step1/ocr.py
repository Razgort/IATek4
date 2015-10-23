#!/usr/bin/python

import utils 
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection

class nn:
  def __init__(self, input_path, output_path):
    self.n = FeedForwardNetwork()
    self.pix_size = 50
    self.input_value = utils.readLines(input_path)
    self.output_value = utils.readLines(output_path)
    self.inputUnits = self.pix_size * self.pix_size 
    self.nbHiddenLayers = 1
    self.hiddenUnits = 500
    self.outputUnits = 1

    inputLayer = LinearLayer(self.inputUnits)
    hiddenLayer = SigmoidLayer(self.hiddenUnits)
    outputLayer = LinearLayer(self.outputUnits)
  
    input_to_hidden = FullConnection(inputLayer, hiddenLayer)
    hidden_to_output = FullConnection(hiddenLayer, outputLayer) 

    self.n.addConnection(input_to_hidden)
    self.n.addConnection(hidden_to_output)


def main():
  nn("./input_data", "./output_data")

if __name__ == "__main__":
  main()
