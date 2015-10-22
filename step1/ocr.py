#!/usr/bin/python

import utils 

class nn:
  def __init__(self, input_path, output_path):
    self.input_paths = utils.readLines(input_path)
    self.output_value = utils.readLines(output_path)
    self.nbHiddenLayers = 100
    self.hiddenUnits = 500
    self.outputUnits = 95

def main():
  print("Starting Neural Net")
  nn("./input_data", "./foutput_data")

if __name__ == "__main__":
  main()
