#include "network.hpp"

network::network()
{
}

network::network(std::vector<int> layerSize)
{
    for (int i = 0; i < layerSize.size(); ++i)
    {
        if (i == 0)
        {
            layers.push_back(new input_layer(layerSize[i]));
        }
        else if (i == layerSize.size() - 1)
        {
            layers.push_back(new output_layer(layers.back(), layerSize[i]));
        }
        else
        {
            layers.push_back(new hidden_layer(layers.back(), layerSize[i]));
        }
    }
}

double network::calculateLoss(std::vector<double> output)
{
    double loss = 0;
    for (int i = 0; i < layers.back()->neurons.size(); ++i)
    {
        loss += pow((layers.back()->neurons.at(i)->o - output[i]), 2);
    }
    return loss;
}

void network::bprop(double loss)
{
}

void network::train(std::vector<std::vector<double>> x, std::vector<std::vector<double>> y, int epoches)
{
    printf("Training Epoches = %d \n", epoches);
    for (int epo = 0; epo < epoches; ++epo)
    {
        double totalLoss = 0;
        for (int i = 0; i < x.size(); ++i)
        {
            auto input = x[i];
            auto output = y[i];

            fprop(input);
            double loss = calculateLoss(output);
            totalLoss += loss;
            bprop(loss);
        }
        printf("[%d|%d] TotalLoss %f \n", epo, epoches, totalLoss);
    }
    printf("End Training \n");
}

void network::fprop(std::vector<double> input)
{
    for (int i = 0; i < layers.size(); ++i)
    {
        layer *preLayer = layers[i]->preLayer;
        for (int j = 0; j < layers[i]->neurons.size(); ++j)
        {
            neuron *n = layers[i]->neurons[j];
            if (i == 0)
            {
                n->o = input[j];
            }
            else
            {
                double sum = n->b;
                for (int k = 0; k < preLayer->neurons.size(); ++k)
                {
                    sum += preLayer->neurons[k]->o * n->w[k];
                }
                n->o = sum;
            }
        }
    }
}

std::vector<double> network::predict(std::vector<double> input)
{
    fprop(input);

    std::vector<double> output;
    for (int i = 0; i < layers.back()->neurons.size(); ++i)
    {
        output.push_back(layers.back()->neurons.at(i)->o);
    }
    return output;
}

network::~network()
{
}