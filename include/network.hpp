#ifndef __Network_HPP
#define __Network_HPP

#include "layer.hpp"

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

    void saveLogs(std::string, std::vector<double>);
    void shuffleData(std::vector<std::vector<double>> &train_input, std::vector<std::vector<double>> &train_output);

public:
    Network();
    Network(std::vector<int> layerSize);
    ~Network();

    double Sigmoid(double);
    double SigmoidDerivative(double);
    double ReLu(double);
    double ReLuDerivative(double);

    double activate(double);
    double activateDerivative(double);

    void train(std::vector<std::vector<double>> x, std::vector<std::vector<double>> y, int epoches, double learningRate, int batchSize = 1, bool shuffle = false);

    std::vector<double> predict(std::vector<double> input);
    void fprop(std::vector<double> input);
    void bprop(std::vector<double> output);
    
    std::vector<double> collectGrad();                                              // collectGrad  but not updateweights
    void updateWeights(std::vector<double> grad, double learningRate);             // update_neuron by grad

    double test(std::vector<std::vector<double>> x, std::vector<std::vector<double>> y);

    void saveModel(std::string path);
    void loadModel(std::string path);

    void printNet();
};

#endif