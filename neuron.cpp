#include "neuron.hpp"

neuron::neuron()
{

}

neuron::neuron(int preSize)
{
    b = dis(engine);
    for(int i = 0;i<preSize;++i){
        w.push_back(dis(engine));
    }
}

neuron::~neuron()
{

}
