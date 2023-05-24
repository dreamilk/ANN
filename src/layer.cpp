#include "layer.hpp"

Layer::Layer()
{
}

Layer::~Layer()
{
}

Layer::Layer(size_t preSize, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        neurons.push_back(std::make_shared<Neuron>(preSize));
    }
}
