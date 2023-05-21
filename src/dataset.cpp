#include "dataset.hpp"

DataSet::DataSet()
{
}

void DataSet::readMnistTrainLable()
{
    label = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
    std::ifstream ifsLable;
    ifsLable.open("./datasets/MNIST_data/train-labels.idx1-ubyte", std::ios::in | std::ios::binary);
    unsigned char bytes[8];
    ifsLable.read((char *)bytes, 8);
    uint32_t magic = (uint32_t)((bytes[0] << 24) |
                                (bytes[1] << 16) |
                                (bytes[2] << 8) |
                                bytes[3]);
    uint32_t num = (uint32_t)((bytes[4] << 24) |
                              (bytes[5] << 16) |
                              (bytes[6] << 8) |
                              bytes[7]);
    // printf("MnistTrainLable %d %d \n", magic, num);
    while (!ifsLable.eof())
    {
        unsigned char byte;
        ifsLable.read((char *)&byte, 1);
        if (ifsLable.fail())
        {
            break;
        }
        int pos = (unsigned int)byte;
        std::vector<double> y(10, 0.0);
        y[pos] = 1.0;
        train_output.push_back(y);
    }
    ifsLable.close();
}

void DataSet::readMnistTestLable()
{
    label = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
    std::ifstream ifsLable;
    ifsLable.open("./datasets/MNIST_data/t10k-labels.idx1-ubyte", std::ios::in | std::ios::binary);
    unsigned char bytes[8];
    ifsLable.read((char *)bytes, 8);
    uint32_t magic = (uint32_t)((bytes[0] << 24) |
                                (bytes[1] << 16) |
                                (bytes[2] << 8) |
                                bytes[3]);
    uint32_t num = (uint32_t)((bytes[4] << 24) |
                              (bytes[5] << 16) |
                              (bytes[6] << 8) |
                              bytes[7]);
    // printf("MnistTestLable %d %d \n", magic, num);
    while (!ifsLable.eof())
    {
        unsigned char byte;
        ifsLable.read((char *)&byte, 1);
        if (ifsLable.fail())
        {
            break;
        }
        int pos = (unsigned int)byte;
        std::vector<double> y(10, 0.0);
        y[pos] = 1.0;
        test_output.push_back(y);
    }
    ifsLable.close();
}

void DataSet::readMnistTrainImage()
{
    std::ifstream ifsLable;
    ifsLable.open("./datasets/MNIST_data/train-images.idx3-ubyte", std::ios::in | std::ios::binary);
    unsigned char bytes[16];
    ifsLable.read((char *)bytes, 16);
    uint32_t magic = (uint32_t)((bytes[0] << 24) |
                                (bytes[1] << 16) |
                                (bytes[2] << 8) |
                                bytes[3]);
    uint32_t num = (uint32_t)((bytes[4] << 24) |
                              (bytes[5] << 16) |
                              (bytes[6] << 8) |
                              bytes[7]);
    uint32_t rows = (uint32_t)((bytes[8] << 24) |
                               (bytes[9] << 16) |
                               (bytes[10] << 8) |
                               bytes[11]);
    uint32_t cols = (uint32_t)((bytes[12] << 24) |
                               (bytes[13] << 16) |
                               (bytes[14] << 8) |
                               bytes[15]);
    // printf("MnistTrainImage %d %d %d %d\n", magic, num, rows, cols);
    while (!ifsLable.eof())
    {
        int cnt = 0;
        std::vector<double> x;
        while (cnt < rows * cols && !ifsLable.fail())
        {
            unsigned char byte;
            ifsLable.read((char *)&byte, 1);
            int pix = (unsigned int)byte;
            x.push_back(pix);
            ++cnt;
        }
        if (x.size() == rows * cols)
            train_input.push_back(x);
    }
    ifsLable.close();
}

void DataSet::readMnistTestImage()
{
    std::ifstream ifsLable;
    ifsLable.open("./datasets/MNIST_data/t10k-images.idx3-ubyte", std::ios::in | std::ios::binary);
    unsigned char bytes[16];
    ifsLable.read((char *)bytes, 16);
    uint32_t magic = (uint32_t)((bytes[0] << 24) |
                                (bytes[1] << 16) |
                                (bytes[2] << 8) |
                                bytes[3]);
    uint32_t num = (uint32_t)((bytes[4] << 24) |
                              (bytes[5] << 16) |
                              (bytes[6] << 8) |
                              bytes[7]);
    uint32_t rows = (uint32_t)((bytes[8] << 24) |
                               (bytes[9] << 16) |
                               (bytes[10] << 8) |
                               bytes[11]);
    uint32_t cols = (uint32_t)((bytes[12] << 24) |
                               (bytes[13] << 16) |
                               (bytes[14] << 8) |
                               bytes[15]);
    // printf("MnistTestImage %d %d %d %d\n", magic, num, rows, cols);
    while (!ifsLable.eof())
    {
        int cnt = 0;
        std::vector<double> x;
        while (cnt < rows * cols && !ifsLable.fail())
        {
            unsigned char byte;
            ifsLable.read((char *)&byte, 1);
            int pix = (unsigned int)byte;
            x.push_back(pix);
            ++cnt;
        }
        if (x.size() == rows * cols)
            test_input.push_back(x);
    }
    ifsLable.close();
}

void DataSet::printDigit(std::vector<double> x, double mask)
{
    if (x.size() != 28 * 28)
    {
        printf("printDigit Error\n");
        return;
    }
    for (int i = 0; i < 28; ++i)
    {
        for (int j = 0; j < 28; ++j)
        {
            if (x[i * 28 + j] > mask)
            {
                printf("##");
            }
            else
            {
                printf("  ");
            }
        }
        printf("\n");
    }
}

void DataSet::readMnistData()
{
    readMnistTrainLable();
    readMnistTrainImage();
    readMnistTestImage();
    readMnistTestLable();
    printf("train_image = %d train_lable = %d \n", train_input.size(), train_output.size());
    printf("test_image = %d test_lable = %d \n", test_input.size(), test_output.size());
}

void DataSet::readIrisData()
{
    label = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"};
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

        train_input.push_back(x);
        train_output.push_back(y);
    }
    f.close();
}

DataSet::~DataSet()
{
}

std::vector<std::vector<double>> DataSet::getInput()
{
    return train_input;
}

std::vector<std::vector<double>> DataSet::getOutput()
{
    return train_output;
}

std::vector<std::vector<double>> DataSet::getTestInput()
{
    return test_input;
}

std::vector<std::vector<double>> DataSet::getTestOutput()
{
    return test_output;
}

double DataSet::getNormalized(double d, double min, double max)
{
    double t = (d - min) / (max - min);
    return t;
}

std::vector<std::vector<double>> DataSet::getNormalizedData(std::vector<std::vector<double>> data)
{
    std::vector<double> maxVec = data[0];
    std::vector<double> minVec = data[0];
    for (int i = 0; i < data.size(); ++i)
    {
        for (int j = 0; j < maxVec.size(); ++j)
        {
            maxVec[j] = std::max(maxVec[j], data[i][j]);
            minVec[j] = std::min(minVec[j], data[i][j]);
        }
    }
    std::vector<std::vector<double>> x;

    for (int i = 0; i < data.size(); ++i)
    {
        std::vector<double> item;
        for (int j = 0; j < maxVec.size(); ++j)
        {
            item.push_back(getNormalized(data[i][j], minVec[j], maxVec[j]));
        }
        x.push_back(item);
    }

    return x;
}
