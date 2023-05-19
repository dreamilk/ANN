#include "network.hpp"

network::network()
{
}

network::network(std::vector<int> layerSize)
{
    for(int i = 0;i<layerSize.size();++i){
        
    }
}

void network::train(std::vector<std::vector<double>> x, std::vector<std::vector<double>> y, int epoches)
{
}

std::vector<double> network::predict(std::vector<double> input)
{
    return std::vector<double>{};
}

network::~network()
{
}