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

    std::vector<std::vector<double>> x_test = ds.getTestInput();
    std::vector<std::vector<double>> y_test = ds.getTestOutput();

    std::vector<std::vector<double>> x_n = ds.getNormalizedData(x);
    std::vector<std::vector<double>> y_n = ds.getNormalizedData(y);

    std::vector<std::vector<double>> x_test_n = ds.getNormalizedData(x_test);
    std::vector<std::vector<double>> y_test_n = ds.getNormalizedData(y_test);

    std::vector<int> spec = {28 * 28, 256, 64, 10}; // the num of neuron in every layer
    Network net(spec);
    net.train(x_n, y_n, 10, 0.01, 16, true, x_test_n, y_test_n);
    net.saveModel("./models/net.model");

    // Network net;
    // net.loadModel("./models/net.model");

    // printf("accuracy %f\n", net.test(x_n, y_n));
    // net.printNet();

    std::vector<double> input = x_n[0];
    std::vector<double> expect = y_n[0];

    ds.printDigit(input, 0.6);

    std::vector<double> output = net.predict(input);
    printf("Predict expect is: \n");
    printData(expect);
    printf("result is %d\n", maxIndex(expect));
    printf("Predict output is: \n");
    printData(output);
    printf("result is %d\n", maxIndex(output));

    return 0;
}