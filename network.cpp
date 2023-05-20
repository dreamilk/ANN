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
    double loss = 0.0;
    for (int i = 0; i < layers.back()->neurons.size(); ++i)
    {
        loss += pow((layers.back()->neurons.at(i)->output - output[i]), 2);
    }
    return loss / (2 * output.size());
}

double network::activate(double x)
{
    return 1 / (1 + exp(-x));
}

double network::activateDerivative(double y)
{
    return y * (1 - y);
}

void network::bprop(std::vector<double> output)
{
    for (int i = layers.size() - 1; i >= 0; i--)
    {
        layer *l = layers.at(i);
        std::vector<double> errors;
        if (i != layers.size() - 1)
        {
            for (int j = 0; j < l->neurons.size(); j++)
            {
                double error = 0;
                for (neuron *n : layers.at(i + 1)->neurons)
                {
                    error += n->weights.at(j) * n->delta;
                }
                errors.push_back(error);
            }
        }
        else
        {
            for (int j = 0; j < l->neurons.size(); j++)
            {
                errors.push_back(output.at(j) - l->neurons.at(j)->output);
            }
        }
        for (int j = 0; j < l->neurons.size(); j++)
        {
            neuron *n = l->neurons.at(j);
            n->delta = errors.at(j) * activateDerivative(n->output);
        }
    }
}

void network::updateWeights(std::vector<double> input, double learingRate)
{
    for (int i = 1; i < layers.size(); ++i)
    {
        layer *l = layers.at(i);
        if (i != 0)
        {
            input.clear();
            for (neuron *n : layers.at(i - 1)->neurons)
            {
                input.push_back(n->output);
            }
        }

        for (neuron *n : l->neurons)
        {
            for (int j = 0; j < n->weights.size(); ++j)
            {
                n->weights[j] += learingRate * n->delta * input.at(j);
            }
            n->bias += learingRate * n->delta;
        }
    }
}

void network::train(std::vector<std::vector<double>> x, std::vector<std::vector<double>> y, int epoches, double learningRate)
{
    printf("Training Epoches = %d  LearningRate = %f\n ", epoches, learningRate);
    for (int epo = 1; epo <= epoches; ++epo)
    {
        double totalLoss = 0;
        for (int i = 0; i < x.size(); ++i)
        {
            auto input = x[i];
            auto output = y[i];

            fprop(input);
            double loss = calculateLoss(output);
            totalLoss += loss;
            bprop(output);
            updateWeights(input, learningRate);
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
                n->output = input[j];
            }
            else
            {
                double sum = n->bias;
                for (int k = 0; k < preLayer->neurons.size(); ++k)
                {
                    sum += preLayer->neurons[k]->output * n->weights[k];
                }
                n->output = activate(sum);
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
        output.push_back(layers.back()->neurons.at(i)->output);
    }
    return output;
}

void network::printNet()
{
    printf("Network\n");
    for (int i = 0; i < layers.size(); i++)
    {
        printf("{");
        for (int j = 0; j < layers.at(i)->neurons.size(); ++j)
        {
            printf(" [");
            neuron *n = layers[i]->neurons[j];
            for (int k = 0; k < n->weights.size(); ++k)
            {
                printf(" %f ", n->weights[k]);
            }
            printf(" %f ] ", n->bias);
        }
        printf("}\n");
    }
}

network::~network()
{
}