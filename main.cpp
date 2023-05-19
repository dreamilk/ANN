#include "layer.hpp"
#include "network.hpp"

#include <cstdio>
#include <vector>

int main()
{
    printf("Welcome my mechine learning\n");
    std::vector<std::vector<double>> x = {{1}, {2}};
    std::vector<std::vector<double>> y = {{2}, {4}};

    network net({1, 2, 2, 1});
    net.train(x, y, 3);

    std::vector<double> input = {3};
    std::vector<double> output = net.predict(input);

    printf("Predict output is: \n");
    for (auto o : output)
    {
        printf("%f ", o);
    }
    printf("\n");

    return 0;
}