#ifndef __OUTPUT_LAYER_HPP
#define __OUTPUT_LAYER_HPP

#include "neuron.hpp"
#include "layer.hpp"

#include <vector>

class output_layer:public layer
{

public:
    output_layer(int size);
    output_layer(layer* preLayer,int size);
};

#endif