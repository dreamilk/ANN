#include "dataset.hpp"

DataSet::DataSet()
{
}

void DataSet::readIrisData()
{
    this->label = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"};
    std::ifstream f;
    f.open("./datasets/IRIS_data/iris.data", std::ios::in);
    std::string line;
    while (std::getline(f, line))
    {
        std::vector<double> x;
        std::vector<double> y;
        std::string s;
        for (int i = 0; i < line.size(); ++i)
        {
            if (line[i] == ',')
            {
                x.push_back(std::stof(s));
                s.clear();
            }
            else
            {
                s += line[i];
            }
        }
        for (int i = 0; i < label.size(); ++i)
        {
            if (label[i] == s)
            {
                y.push_back(1.0);
            }
            else
            {
                y.push_back(0.0);
            }
        }

        input.push_back(x);
        output.push_back(y);
    }
    f.close();
}

DataSet::~DataSet()
{
}

std::vector<std::vector<double>> DataSet::getInput()
{
    return input;
}

std::vector<std::vector<double>> DataSet::getOutput()
{
    return output;
}

double DataSet::getNormalized(double d, double min, double max)
{
    double t = (d - min) / (max - min);
    return t;
}

std::vector<std::vector<double>> DataSet::getNormalizedInput()
{
    std::vector<double> maxVec = input[0];
    std::vector<double> minVec = input[0];
    for (int i = 0; i < input.size(); ++i)
    {
        for (int j = 0; j < maxVec.size(); ++j)
        {
            maxVec[j] = std::max(maxVec[j], input[i][j]);
            minVec[j] = std::min(minVec[j], input[i][j]);
        }
    }
    std::vector<std::vector<double>> x;

    for (int i = 0; i < input.size(); ++i)
    {
        std::vector<double> item;
        for (int j = 0; j < maxVec.size(); ++j)
        {
            item.push_back(getNormalized(input[i][j], minVec[j], maxVec[j]));
        }
        x.push_back(item);
    }

    return x;
}
