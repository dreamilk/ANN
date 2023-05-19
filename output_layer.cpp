#include "output_layer.hpp"

output_layer::output_layer(int size){
    
}

output_layer::output_layer(layer* preLayer,int size){
    this->preLayer = preLayer;
    for(int i = 0;i<size;++i){
        neurons.push_back(new neuron(preLayer->neurons.size()));
    }
}