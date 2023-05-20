#ifndef __Neuron_HPP
#define __Neuron_HPP

#include <vector>
#include <random>
#include <ctime>

class Neuron
{
private:
    std::default_random_engine engine;
    std::normal_distribution<double> distribution;

public:
    double generateRandom(double min, double max);

    std::vector<double> weights;
    double bias;
    double output;
    double delta;

public:
    Neuron();
    Neuron(int preSize);
    ~Neuron();
};

#endif