#ifndef __Network_HPP
#define __Network_HPP

#include "layer.hpp"

#include <vector>
#include <algorithm>
#include <fstream>
#include <string>

class Network
{
private:
    std::vector<Layer *> layers;

    double calculateLoss(std::vector<double> output);

public:
    Network();
    Network(std::vector<int> layerSize);
    ~Network();

    double activate(double);
    double activateDerivative(double);

    void train(std::vector<std::vector<double>> x, std::vector<std::vector<double>> y, int epoches, double learningRate);
    std::vector<double> predict(std::vector<double> input);
    void fprop(std::vector<double> input);
    void bprop(std::vector<double> output);
    void updateWeights(std::vector<double> input, double learningRate);

    double test(std::vector<std::vector<double>> x, std::vector<std::vector<double>> y);

    void saveModel(std::string path);
    void loadModel(std::string path);

    void printNet();
};

#endif