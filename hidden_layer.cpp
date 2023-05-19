#include "hidden_layer.hpp"

#include<vector>

hidden_layer::hidden_layer(layer* preLayer,int size){
    this->preLayer = preLayer;
    for(int i = 0;i<size;++i){
        neurons.push_back(new neuron(preLayer->neurons.size()));
    }
}

hidden_layer::~hidden_layer()
{

}