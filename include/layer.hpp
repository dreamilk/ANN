#ifndef __Layer_HPP
#define __Layer_HPP

#include "neuron.hpp"

#include <memory>
#include <vector>

class Layer
{
public:
    std::vector<std::shared_ptr<Neuron>> neurons;

public:
    Layer();
    Layer(size_t preSize, size_t size);
    ~Layer();
};

#endif