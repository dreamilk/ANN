#ifndef __NETWORK_HPP
#define __NETWORK_HPP

#include "layer.hpp"
#include "input_layer.hpp"
#include "output_layer.hpp"
#include "hidden_layer.hpp"

#include <vector>
#include <algorithm>

class network
{
private:
    std::vector<layer *> layers;

    double calculateLoss(std::vector<double> output);

public:
    network();
    network(std::vector<int> layerSize);
    ~network();

    double activate(double);
    double activateDerivative(double);

    void train(std::vector<std::vector<double>> x, std::vector<std::vector<double>> y, int epoches, double learningRate);
    std::vector<double> predict(std::vector<double> input);
    void fprop(std::vector<double> input);
    void bprop(std::vector<double> output);
    void updateWeights(std::vector<double> input, double learningRate);

    double test(std::vector<std::vector<double>> x, std::vector<std::vector<double>> y);

    void printNet();
};

#endif