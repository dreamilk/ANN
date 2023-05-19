#ifndef __OUTPUT_LAYER_HPP
#define __OUTPUT_LAYER_HPP

#include "neuron.hpp"
#include "layer.hpp"

#include <vector>

class input_layer:layer
{

public:
    input_layer(int size);
    ~input_layer();
};

#endif