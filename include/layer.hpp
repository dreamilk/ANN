#ifndef __LAYER_HPP
#define __LAYER_HPP

#include "neuron.hpp"

#include <vector>

class layer
{
public:
    std::vector<neuron*> neurons;
    layer *preLayer;

public:
    layer();
    layer(int size);
    layer(layer* preLayer,int size);
    ~layer();
};

#endif