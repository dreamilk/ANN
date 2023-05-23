#ifndef __Network_HPP
#define __Network_HPP

#include "layer.hpp"
#include "data.hpp"

#include <vector>
#include <algorithm>
#include <fstream>
#include <string>
#include <chrono>

class Network
{
private:
    std::vector<Layer *> layers;

    double calculateLoss(std::vector<double> output);

public:
    Network();
    Network(std::vector<int> layerSize);
    ~Network();

    double Sigmoid(double);
    double SigmoidDerivative(double);
    double ReLu(double);
    double ReLuDerivative(double);
    double LeakyReLu(double);
    double LeakyReLuDerivative(double);
    double Tanh(double);
    double TanhDerivative(double);

    double activate(double);
    double activateDerivative(double);

    void train(std::vector<std::vector<double>> x, std::vector<std::vector<double>> y, int epoches, double learningRate,
               int batchSize = 1, bool shuffle = false,
               std::vector<std::vector<double>> test_input = {},
               std::vector<std::vector<double>> test_ouput = {});

    std::vector<double> predict(std::vector<double> input);
    void fprop(std::vector<double> input);
    void bprop(std::vector<double> output);

    std::vector<double> collectGrad();                                 // collectGrad  but not updateweights
    void updateWeights(std::vector<double> grad, double learningRate); // update_neuron by grad

    double test(std::vector<std::vector<double>> x, std::vector<std::vector<double>> y);

    void saveModel(std::string path);
    void loadModel(std::string path);

    void printNet();
};

#endif