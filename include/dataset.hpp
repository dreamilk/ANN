#ifndef __DATASET_HPP
#define __DATASET_HPP

#include <vector>
#include <fstream>
#include <string>

class DataSet
{
private:
    std::vector<std::vector<double>> train_input;
    std::vector<std::vector<double>> train_output;
    std::vector<std::vector<double>> test_input;
    std::vector<std::vector<double>> test_output;
    std::vector<std::string> label;

    double getNormalized(double d, double min, double max);

    void readMnistTrainImage();
    void readMnistTrainLable();
    void readMnistTestImage();
    void readMnistTestLable();

public:
    DataSet();

    void readIrisData();
    void readMnistData();

    void printDigit(std::vector<double>, double mask);

    std::vector<std::vector<double>> getNormalizedData(std::vector<std::vector<double>>);

    std::vector<std::vector<double>> getInput();
    std::vector<std::vector<double>> getOutput();

    std::vector<std::vector<double>> getTestInput();
    std::vector<std::vector<double>> getTestOutput();

    ~DataSet();
};

#endif