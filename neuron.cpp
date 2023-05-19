#include "neuron.hpp"

neuron::neuron()
{
    
}

neuron::~neuron()
{

}

neuron::neuron(double w,double b):w(w),b(b){
    error = 0;
}