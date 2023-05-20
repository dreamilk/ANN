#ifndef __DATASET_HPP
#define __DATASET_HPP

#include <vector>
#include <fstream>
#include <string>

class DataSet
{
private:
    std::vector<std::vector<double>> input;
    std::vector<std::vector<double>> output;
    std::vector<std::string> label;

    double getNormalized(double d,double min,double max);

public:
    DataSet();

    void readIrisData();

    std::vector<std::vector<double>> getNormalizedInput();
    std::vector<std::vector<double>> getNormalizedOutput();

    std::vector<std::vector<double>> getInput();
    std::vector<std::vector<double>> getOutput();

    ~DataSet();
};

#endif