#include "Layer.hpp"

Layer::Layer()
{
}

Layer::~Layer()
{
    for (int i = 0; i < neurons.size(); ++i)
    {
        Neuron *n = neurons.at(i);
        delete n;
    }
    neurons.clear();
}

Layer::Layer(int preSize, int size)
{
    for (int i = 0; i < size; ++i)
    {
        neurons.push_back(new Neuron(preSize));
    }
}