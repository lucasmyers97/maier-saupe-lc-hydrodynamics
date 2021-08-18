#include <cmath>
#include <iostream>
#include <bitset>
#include <cmath>

namespace{
    constexpr unsigned long n_bits{5};
}

int main()
{
    for (unsigned long j = 0; j < std::pow(2, n_bits); ++j)
    {
        unsigned long x{j};
        std::bitset<n_bits> x_bits(x);

        for (int i = 0; i < n_bits; ++i)
        {
            std::cout << x_bits[i] << std::endl;
        }
        std::cout << std::endl;
    }

    return 0;
}