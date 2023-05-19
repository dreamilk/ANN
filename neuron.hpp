#ifndef __NEURON_HPP
#define __NEURON_HPP

class neuron
{
private:

public:
    double w,b;
    double o;
    double error;
    
public:
    neuron(double w,double b);
    neuron();
    ~neuron();
};

#endif