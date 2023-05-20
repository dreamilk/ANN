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

    // for (auto a : x[0])
    // {
    //     printf("%f ", a);
    // }
    // printf("\n");
    // for (auto b : y[0])
    // {
    //     printf("%f ", b);
    // }
    // printf("\n");

    std::vector<int> spec = {4, 4, 5, 6, 5, 4, 3}; // the num of neuron in every layer

    network net(spec);
    net.train(x, y, 10000, 0.01);

    std::vector<double> input = x[0];
    std::vector<double> output = net.predict(input);
    printf("Predict output is: \n");
    for (auto o : output)
    {
        printf("%f ", o);
    }
    printf("\n");

    // net.printNet();
    printf("accuracy %f\n", net.test(x, y));

    return 0;
}