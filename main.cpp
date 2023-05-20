#include "layer.hpp"
#include "network.hpp"

#include <cstdio>
#include <vector>

int main()
{
    printf("Welcome my mechine learning\n");
    std::vector<std::vector<double>> x;
    std::vector<std::vector<double>> y;

    for (int i = 1; i <= 10000; i++)
    {
        if (i % 2 == 0)
        {
            x.push_back({2, 1});
            y.push_back({1, 0});
        }
        else
        {
            x.push_back({1, 3});
            y.push_back({0, 1});
        }
    }

    std::vector<int> spec = {1, 2, 2}; // the num of neuron in every layer

    network net(spec);
    net.train(x, y, 100, 0.1);

    std::vector<double> input = {2, 1};
    std::vector<double> output = net.predict(input);
    net.printNet();

    printf("Predict output is: \n");
    for (auto o : output)
    {
        printf("%f ", o);
    }
    printf("\n");

    return 0;
}