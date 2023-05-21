#ifndef __DATA_HPP
#define __DATA_HPP

#include <iostream>
#include <algorithm>

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

#endif