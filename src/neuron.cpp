#include "neuron.hpp"

neuron::neuron()
{
    bias = 0;
    delta = 0;
    output = 0;
}

double neuron::generateRandom(double min, double max)
{
    double rd = (double)rand() / RAND_MAX;
    return min + rd * (max - min);
}

neuron::neuron(int preSize)
{
    bias = generateRandom(-1, 1);
    for (int i = 0; i < preSize; ++i)
    {
        weights.push_back(generateRandom(-1, 1));
    }
    delta = 0.0;
    output = 0.0;
}

neuron::~neuron()
{
}
