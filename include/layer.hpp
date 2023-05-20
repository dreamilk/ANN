#ifndef __Layer_HPP
#define __Layer_HPP

#include "neuron.hpp"

#include <vector>

class Layer
{
public:
    std::vector<Neuron *> neurons;

public:
    Layer();
    Layer(int preSize, int size);
    ~Layer();
};

#endif