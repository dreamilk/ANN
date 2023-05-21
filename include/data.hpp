#ifndef __DATA_HPP
#define __DATA_HPP

#include <iostream>

template <typename T>
void printData(T t)
{
    for (int i = 0; i < t.size(); ++i)
    {
        printf("%f ", t[i]);
    }
    printf("\n");
}

#endif