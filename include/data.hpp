#ifndef __DATA_HPP
#define __DATA_HPP

#include <iostream>
#include <algorithm>
#include <fstream>

template <typename T>
void printData(T t)
{
    for (int i = 0; i < t.size(); ++i)
    {
        printf("%f ", t[i]);
    }
    printf("\n");
}

template <typename T>
int maxIndex(T t)
{
    return std::max_element(t.begin(), t.end()) - t.begin();
}

template <typename T>
void saveLogs(std::string path, std::vector<T> logs)
{
    std::ofstream ofs(path, std::ios::out);
    if (!ofs.is_open())
    {
        printf("save logs Error\n");
    }
    else
    {
        for (auto a : logs)
        {
            ofs << a << std::endl;
        }
        ofs.close();
    }
}

template <typename T>
void shuffleData(T &train_input, T &train_output)
{
    int len = train_input.size();
    int num = len / 16;
    while (num--)
    {
        int p1 = ((double)rand() / RAND_MAX) * len;
        int p2 = ((double)rand() / RAND_MAX) * len;
        if (p1 >= len || p2 >= len)
        {
            continue;
        }
        std::swap(train_input[p1], train_input[p2]);
        std::swap(train_output[p1], train_output[p2]);
    }
}

#endif