#ifndef __NEURON_HPP
#define __NEURON_HPP

#include <vector>
#include <random>
#include <ctime>

class neuron
{
private:
    std::default_random_engine engine{time(nullptr)};
    std::normal_distribution<> dis;

public:
    std::vector<double> w;
    double b;
    double o;
    double error;
    
public:
    neuron();
    neuron(int preSize);
    ~neuron();
};

#endif