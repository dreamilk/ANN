#include "layer.hpp"
#include "network.hpp"
#include "dataset.hpp"

#include "data.hpp"

#include <cstdio>
#include <vector>

int main()
{
    printf("Welcome my mechine learning\n");
    DataSet ds;
    ds.readMnistData();
    std::vector<std::vector<double>> x = ds.getInput();
    std::vector<std::vector<double>> y = ds.getOutput();
    std::vector<std::vector<double>> x_test = ds.getInput();
    std::vector<std::vector<double>> y_test = ds.getOutput();

    std::vector<int> spec = {28 * 28, 256, 64, 10}; // the num of neuron in every layer
    Network net(spec);
    net.train(x, y, 10, 0.01);
    net.saveModel("./net.model");

    // Network net;
    // net.loadModel("./net.model");

    printf("accuracy %f\n", net.test(x_test, y_test));
    // net.printNet();

    std::vector<double> input = x_test[0];
    std::vector<double> expect = y_test[0];

    ds.printDigit(input);
    std::vector<double> output = net.predict(input);
    printf("Predict expect is: \n");
    printData(expect);
    printf("Predict output is: \n");
    printData(output);

    return 0;
}