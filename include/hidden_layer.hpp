#ifndef __HIDDEN_LAYER_HPP
#define __HIDDEN_LAYER_HPP

#include "layer.hpp"

class hidden_layer:public layer
{
private:
    /* data */
public:
    hidden_layer(layer* preLayer,int size);
    ~hidden_layer();
};



#endif