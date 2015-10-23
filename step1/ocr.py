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
        self.outputUnits = len(self.output_value)

        inputLayer = LinearLayer(self.inputUnits)
        hiddenLayers = []
        for i in range(0, self.nbHiddenLayers):
            hiddenLayers.append(SigmoidLayer(self.hiddenUnits))
        outputLayer = LinearLayer(self.outputUnits)

        input_to_hidden = FullConnection(inputLayer, hiddenLayers[0])
        hidden_to_hidden = []
        if len(hiddenLayers) > 1:
            for i in range(1, len(hiddenLayers) - 2):
                hidden_to_hidden.append(
                    FullConnection(hiddenLayers[i], hiddenLayers[i + 1]))
        hidden_to_output = FullConnection(
            hiddenLayers[len(hiddenLayers) - 1], outputLayer)
        self.n.addConnection(input_to_hidden)
        for hidden in hidden_to_hidden:
            self.n.addConnection(hidden)
        self.n.addConnection(hidden_to_output)


def main():
    nn("./input_data", "./output_data")

if __name__ == "__main__":
    main()
