#include "layer.hpp"
#include "network.hpp"
#include "dataset.hpp"

#include <cstdio>
#include <vector>

int main()
{
    printf("Welcome my mechine learning\n");
    DataSet ds;
    ds.readIrisData();
    std::vector<std::vector<double>> x = ds.getInput();
    std::vector<std::vector<double>> y = ds.getOutput();

    // std::vector<int> spec = {4, 5, 3}; // the num of neuron in every layer
    // Network net(spec);
    // net.train(x, y, 10000, 0.1);
    // net.saveModel("./data.model");

    Network net;
    net.loadModel("./data.model");

    std::vector<double> input = x[0];
    std::vector<double> output = net.predict(input);
    printf("Predict output is: \n");
    for (auto o : output)
    {
        printf("%f ", o);
    }
    printf("\n");

    printf("accuracy %f\n", net.test(x, y));

    // net.printNet();

    return 0;
}