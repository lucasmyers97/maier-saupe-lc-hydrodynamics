#include "Numerics/NumericalTools.hpp"

#include <iostream>
#include <algorithm>

int main()
{
    double begin;
    double end;
    unsigned int num;

    std::cout << "input the beginning" << std::endl;
    std::cin >> begin;
    std::cout << "input the end" << std::endl;
    std::cin >> end;
    std::cout << "input the number of steps" << std::endl;
    std::cin >> num;

    std::vector<double> range = NumericalTools::linspace(begin, end, num);
    std::for_each(range.begin(), range.end(),
                  [](double d){std::cout << d << " ";});
    std::cout << std::endl;

    return 0;
}
