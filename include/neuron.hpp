#ifndef __NEURON_HPP
#define __NEURON_HPP

#include <vector>
#include <random>
#include <ctime>

class neuron
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
    neuron();
    neuron(int preSize);
    ~neuron();
};

#endif