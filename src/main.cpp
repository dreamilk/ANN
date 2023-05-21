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
    ds.readIrisData();

    std::vector<std::vector<double>> x = ds.getInput();
    std::vector<std::vector<double>> y = ds.getOutput();

    std::vector<std::vector<double>> x_test = ds.getTestInput();
    std::vector<std::vector<double>> y_test = ds.getTestOutput();

    // make data normalized
    std::vector<std::vector<double>> x_n = ds.getNormalizedData(x);
    std::vector<std::vector<double>> y_n = ds.getNormalizedData(y);

    std::vector<std::vector<double>> x_test_n = ds.getNormalizedData(x_test);
    std::vector<std::vector<double>> y_test_n = ds.getNormalizedData(y_test);

    std::vector<int> spec = {4, 5, 3}; // the num of neuron in every layer
    Network net(spec);
    net.train(x_n, y_n, 300, 0.1, 3, true, x_n, y_n);
    net.saveModel("./models/net.model");

    // Network net;
    // net.loadModel("./models/net.model");

    printf("accuracy %f\n", net.test(x_n, y_n));
    // net.printNet();

    // predict test
    std::vector<double> input = x_n[0];
    std::vector<double> expect = y_n[0];

    // ds.printDigit(input, 0.6);

    std::vector<double> output = net.predict(input);
    printf("Predict expect is: \n");
    printData(expect);
    printf("expect is %d\n", maxIndex(expect));
    printf("Predict output is: \n");
    printData(output);
    printf("result is %d\n", maxIndex(output));

    return 0;
}