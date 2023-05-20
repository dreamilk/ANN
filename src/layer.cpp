#include "layer.hpp"

layer::layer()
{
}

layer::~layer()
{
}

layer::layer(int size) : preLayer(nullptr)
{
    for (int i = 0; i < size; ++i)
    {
        neurons.push_back(new neuron(1));
    }
}

layer::layer(layer *preLayer, int size) : preLayer(preLayer)
{
    for (int i = 0; i < size; ++i)
    {
        neurons.push_back(new neuron(preLayer->neurons.size()));
    }
}