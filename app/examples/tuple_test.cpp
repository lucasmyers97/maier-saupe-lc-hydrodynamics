#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>
#include <iostream>

int main()
{
    constexpr int D_dim = 3;
    using permutation = std::tuple<int, int, int>;

    std::vector<permutation> 
        pos_perms = {{0, 1, 2},
                     {1, 2, 0},
                     {2, 0, 1}};
    std::vector<permutation> 
        neg_perms = {{2, 1, 0},
                     {0, 2, 1},
                     {1, 0, 2}};
    for (std::size_t i = 0; i < pos_perms.size(); ++i)
    {
        std::cout << std::get<0>(pos_perms[i]);
        std::cout << std::get<1>(pos_perms[i]);
        std::cout << std::get<2>(pos_perms[i]);
        std::cout << "\n\n";

        std::cout << std::get<0>(neg_perms[i]);
        std::cout << std::get<1>(neg_perms[i]);
        std::cout << std::get<2>(neg_perms[i]);
        std::cout << "\n\n";
    }


    std::tuple<int, int, int> perm;
    for (int i = 0; i < D_dim; ++i)
        for (int j = 0; j < D_dim; ++j)
            for (int k = 0; k < D_dim; ++k)
            {
                perm = {i, j, k};
                if ((perm == pos_perms[0]) ||
                    (perm == pos_perms[1]) ||
                    (perm == pos_perms[2]))
                {
                    std::cout << "+1 for: " << i << j << k << std::endl;
                }
                else if ((perm == neg_perms[0]) ||
                         (perm == neg_perms[1]) ||
                         (perm == neg_perms[2]))
                {
                    std::cout << "+1 for: " << i << j << k << std::endl;
                }
            }

    return 0;
}
