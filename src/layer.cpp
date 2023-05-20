#include "Layer.hpp"

Layer::Layer()
{
}

Layer::~Layer()
{
}

Layer::Layer(int preSize, int size)
{
    for (int i = 0; i < size; ++i)
    {
        neurons.push_back(new Neuron(preSize));
    }
}