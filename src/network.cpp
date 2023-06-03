#include "network.hpp"

Network::Network()
{
}

Network::Network(std::vector<int> layerSize)
{
    for (int i = 0; i < layerSize.size(); ++i)
    {
        if (i == 0)
        {
            layers.push_back(std::make_shared<Layer>(0, layerSize[i]));
        }
        else
        {
            layers.push_back(std::make_shared<Layer>(layerSize[i - 1], layerSize[i]));
        }
    }
}

double Network::calculateLoss(std::vector<double> output)
{
    double loss = 0.0;
    for (int i = 0; i < layers.back()->neurons.size(); ++i)
    {
        loss += pow((layers.back()->neurons.at(i)->output - output[i]), 2);
    }
    return loss / (2 * output.size());
}

double Network::Sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

double Network::SigmoidDerivative(double y)
{
    return y * (1 - y);
}

double Network::ReLu(double x)
{
    return std::max(x, 0.0);
}

double Network::ReLuDerivative(double y)
{
    return y > 0 ? 1.0 : 0.0;
}

double Network::LeakyReLu(double d)
{
    return std::max(0.01 * d, d);
}
double Network::LeakyReLuDerivative(double d)
{
    return d > 0 ? 1 : 0.01;
}

double Network::Tanh(double x)
{
    // double a = exp(x);
    // double b = exp(-x);
    // return (a - b) / (a + b);
    return tanh(x);
}

double Network::TanhDerivative(double x)
{
    // double a = exp(x);
    // double b = exp(-x);
    // return 1 - pow((a - b) / (a + b), 2);
    return 1 - pow(tanh(x), 2);
}

double Network::activate(double x)
{
    return Sigmoid(x);
}

double Network::activateDerivative(double y)
{
    return SigmoidDerivative(y);
}

void Network::bprop(std::vector<double> output)
{
    for (int i = layers.size() - 1; i >= 0; i--)
    {
        auto l = layers.at(i);
        std::vector<double> errors;
        if (i != layers.size() - 1)
        {
            for (int j = 0; j < l->neurons.size(); j++)
            {
                double error = 0;
                for (auto n : layers.at(i + 1)->neurons)
                {
                    error += n->weights.at(j) * n->delta;
                }
                errors.push_back(error);
            }
        }
        else
        {
            for (int j = 0; j < l->neurons.size(); j++)
            {
                errors.push_back(output.at(j) - l->neurons.at(j)->output);
            }
        }
        for (int j = 0; j < l->neurons.size(); j++)
        {
            auto n = l->neurons.at(j);
            n->delta = errors.at(j) * activateDerivative(n->output);
        }
    }
}

std::vector<double> Network::collectGrad()
{
    std::vector<double> grad;
    std::vector<double> input;
    for (int i = 1; i < layers.size(); ++i)
    {
        auto l = layers.at(i);

        for (auto n : layers.at(i - 1)->neurons)
        {
            input.push_back(n->output);
        }

        for (auto n : l->neurons)
        {
            for (int j = 0; j < n->weights.size(); ++j)
            {
                grad.push_back(n->delta * input.at(j));
            }
            grad.push_back(n->delta);
        }
        input.clear();
    }
    return grad;
}

void Network::updateWeights(std::vector<double> grad, double learningRate)
{
    int p = 0;
    for (int i = 1; i < layers.size(); ++i)
    {
        auto l = layers.at(i);

        for (auto n : l->neurons)
        {
            for (int j = 0; j < n->weights.size(); ++j)
            {
                n->weights[j] += learningRate * grad[p++];
            }
            n->bias += learningRate * grad[p++];
        }
    }
}

// void Network::updateWeights(std::vector<double> input, double learingRate)
// {
//     for (int i = 0; i < layers.size(); ++i)
//     {
//         Layer *l = layers.at(i);
//         if (i != 0)
//         {
//             input.clear();
//             for (Neuron *n : layers.at(i - 1)->neurons)
//             {
//                 input.push_back(n->output);
//             }
//         }

//         for (Neuron *n : l->neurons)
//         {
//             for (int j = 0; j < n->weights.size(); ++j)
//             {
//                 n->weights[j] += learingRate * n->delta * input.at(j);
//             }
//             n->bias += learingRate * n->delta;
//         }
//     }
// }

void Network::train(std::vector<std::vector<double>> x, std::vector<std::vector<double>> y,
                    int epoches, double learningRate,
                    int batchSize, bool shuffle,
                    std::vector<std::vector<double>> test_input,
                    std::vector<std::vector<double>> test_ouput)
{
    printf("Training Epoches = %d LearningRate = %f BatchSize = %d Shuffle = %d \n", epoches, learningRate, batchSize, shuffle);
    printf("TrainDataSize = %d TestDataSize = %d \n", x.size(), test_input.size());
    if (x.size() != y.size() || test_input.size() != test_ouput.size() || x.size() % batchSize != 0)
    {
        printf("There are some errors in the dataset or parameters\n");
        return;
    }

    std::vector<std::string> log;
    auto start = std::chrono::system_clock::now();

    int train_step = 0;

    for (int epo = 1; epo <= epoches; ++epo)
    {
        // shuffle train_datasets
        if (shuffle)
        {
            shuffleData(x, y);
        }
        int iteration = x.size() / batchSize;

        double totalLoss = 0.0;

        for (int j = 0; j < iteration; ++j)
        {
            std::vector<std::vector<double>> totalGrad;
            for (int i = 0; i < batchSize; ++i)
            {
                auto input = x[j * batchSize + i];
                auto expect = y[j * batchSize + i];

                fprop(input);
                // calculate loss
                double loss = calculateLoss(expect);
                totalLoss += loss;
                bprop(expect);
                // update after batchsize
                std::vector<double> grad = collectGrad();
                totalGrad.push_back(grad);
            }
            std::vector<double> avgGrad;
            for (int i = 0; i < totalGrad[0].size(); ++i)
            {
                double sum = 0.0;
                for (int j = 0; j < totalGrad.size(); ++j)
                {
                    sum += totalGrad[j][i];
                }
                avgGrad.push_back(sum / totalGrad.size());
            }
            updateWeights(avgGrad, learningRate);
            ++train_step;
        }

        double accuracy = -1.0;
        if (test_input.size() > 0 && test_input.size() == test_ouput.size())
        {
            accuracy = test(test_input, test_ouput);
        }

        printf("[%d|%d] TotalLoss %f Accuracy %f \n", epo, epoches, totalLoss, accuracy);
        log.push_back(std::to_string(totalLoss) + " " + std::to_string(accuracy));
    }
    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::string path = "./logs/train_logs.txt";
    saveLogs(path, log);

    printf("End Training, Train Step %ld , Time Cost %.2f s, Save Logs %s \n", train_step, 0.000001 * duration, path.c_str());
}

void Network::fprop(std::vector<double> input)
{
    for (int i = 0; i < layers.size(); ++i)
    {
        for (int j = 0; j < layers[i]->neurons.size(); ++j)
        {
            auto n = layers[i]->neurons[j];
            if (i == 0)
            {
                n->output = input[j];
            }
            else
            {
                auto preLayer = layers[i - 1];
                double sum = n->bias;
                for (int k = 0; k < preLayer->neurons.size(); ++k)
                {
                    sum += preLayer->neurons[k]->output * n->weights[k];
                }
                n->output = activate(sum);
            }
        }
    }
}

std::vector<double> Network::predict(std::vector<double> input)
{
    fprop(input);

    std::vector<double> output;
    for (int i = 0; i < layers.back()->neurons.size(); ++i)
    {
        output.push_back(layers.back()->neurons.at(i)->output);
    }
    return output;
}

void Network::printNet()
{
    printf("Network Structure:\n");
    printf("--------------------------------------------\n");
    long long int sum = 0;
    for (int i = 0; i < layers.size(); ++i)
    {
        std::string s;
        auto l = layers.at(i);
        if (i == 0)
        {
            printf("[input_layer] input_size = %d \n", l->neurons.size());
        }
        else if (i == layers.size() - 1)
        {
            printf("[output_layer] output_size = %d \n", l->neurons.size());
        }
        else
        {
            printf("[hidden_layer] neuron_size = %d \n", l->neurons.size());
        }
        sum += l->neurons.size() * (l->neurons[0]->weights.size() + 1);
    }
    printf("--------------------------------------------\n");
    printf("Number of parameters: %lld\n", sum);
    printf("--------------------------------------------\n");
}

double Network::test(std::vector<std::vector<double>> input, std::vector<std::vector<double>> output)
{
    int totalNum = input.size();
    int correctNum = 0;
    for (int i = 0; i < totalNum; ++i)
    {
        auto x = input.at(i);
        auto y = output.at(i);
        std::vector<double> z = predict(x);

        int a = std::max_element(y.begin(), y.end()) - y.begin();
        int b = std::max_element(z.begin(), z.end()) - z.begin();
        correctNum += (a == b);
    }
    return 1.0 * correctNum / totalNum;
}

void Network::saveModel(std::string path)
{
    std::vector<std::string> data;
    for (int i = 0; i < layers.size(); ++i)
    {
        auto l = layers.at(i);
        data.push_back("L" + std::to_string(l->neurons.size()));
        for (auto n : l->neurons)
        {
            std::string nData = "N" + std::to_string(n->weights.size()) + " ";
            for (int j = 0; j < n->weights.size(); ++j)
            {
                nData += std::to_string(n->weights[j]) + ",";
            }
            nData += std::to_string(n->bias);
            data.push_back(nData);
        }
    }
    std::ofstream ofs;
    ofs.open(path, std::ios::out);
    if (!ofs.is_open())
    {
        printf("saveModel Error\n");
        return;
    }
    for (std::string s : data)
    {
        ofs << s << std::endl;
    }
    ofs.close();
}

void Network::loadModel(std::string path)
{
    printf("LoadModel from %s \n", path.c_str());
    std::ifstream ifs;
    ifs.open(path, std::ios::in);
    if (!ifs.is_open())
    {
        printf("LoadModel Error\n");
        return;
    }
    layers.clear();

    std::string line;
    while (std::getline(ifs, line))
    {
        if (line[0] == 'L')
        {
            int size = std::stoi(line.substr(1, line.size() - 1));
            layers.push_back(std::make_shared<Layer>());
        }
        else if (line[0] == 'N')
        {
            auto l = layers.back();
            int num = 0;
            std::string s;
            std::shared_ptr<Neuron> neu = std::make_shared<Neuron>();
            for (int i = 1; i < line.size(); ++i)
            {
                if (line[i] == ' ')
                {
                    num = std::stoi(s);
                    s.clear();
                }
                else if (line[i] == ',')
                {
                    neu->weights.push_back(std::stod(s));
                    s.clear();
                }
                else
                {
                    s += line[i];
                }
            }
            neu->bias = std::stod(s);
            l->neurons.push_back(neu);
        }
    }
    ifs.close();
    printf("LoadModel Success \n");
}

Network::~Network()
{
}