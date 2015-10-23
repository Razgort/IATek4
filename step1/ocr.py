#!/usr/bin/python

import utils
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.supervised.trainers import BackpropTrainer

results = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~"


class nn:

    def initilizeNetwork(self):
        inputLayer = LinearLayer(self.inputUnits)
        hiddenLayers = []
        for i in range(0, self.nbHiddenLayers):
            hiddenLayers.append(SigmoidLayer(self.hiddenUnits))
            self.n.addModule(hiddenLayers[i])
        outputLayer = LinearLayer(self.outputUnits)

        self.n.addInputModule(inputLayer)
        self.n.addOutputModule(outputLayer)
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
        self.n.sortModules()

    def initializeDataSet(self):
        if len(self.input_value) != len(self.output_value):
            print "Input and output arrays aren't the same size."
            exit()

        for i in range(0, len(self.input_value)):
            self.ds.addSample(
                self.input_value[i], utils.getPosition(self.output_value[i], results))

    def trainingOnDataSet(self):
        trainer = BackpropTrainer(self.n, self.ds)
        print "toto"
        result = trainer.train()
        print result

    def __init__(self, input_path, output_path):
        self.n = FeedForwardNetwork()
        self.pix_size = 50
        self.input_value = utils.getImages(utils.readLines(input_path))
        self.output_value = utils.readLines(output_path)
        self.inputUnits = self.pix_size * self.pix_size
        self.nbHiddenLayers = 1
        self.hiddenUnits = 500
        self.outputUnits = len(results)
        self.ds = SupervisedDataSet(
            self.pix_size * self.pix_size, len(results))

        self.initializeDataSet()
        self.initilizeNetwork()
        self.trainingOnDataSet()


def main():
    nn("./input_data", "./output_data")

if __name__ == "__main__":
    main()
