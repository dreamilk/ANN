#include "layer.hpp"

Layer::Layer()
{
}

Layer::~Layer()
{
    for (size_t i = 0; i < neurons.size(); ++i)
    {
        Neuron *n = neurons.at(i);
        delete n;
    }
    //neurons.clear();
}

Layer::Layer(int preSize, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        neurons.push_back(new Neuron(preSize));
    }
}
