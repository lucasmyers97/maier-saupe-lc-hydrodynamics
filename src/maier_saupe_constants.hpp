#ifndef MAIER_SAUPE_CONSTANTS_HPP
#define MAIER_SAUPE_CONSTANTS_HPP

#include <array> 

namespace maier_saupe_constants{

    // dim of (i) vector representing Q-tensor, and (ii) matrix of Q-tensor
    template <int space_dim>
    constexpr int mat_dim{3};
    template <int space_dim>
    constexpr int vec_dim{};

    // Q-tensor degrees of freedom for 3- and 2-dimensional systems
    template<>
    constexpr int vec_dim<3> = 5;
    template<>
    constexpr int vec_dim<2> = 3;

    // alias Q-tensor vector and matrix classes
    template <int space_dim>
    using vec = std::array<int, vec_dim<space_dim>>;
    template <int space_dim>
    using mat = std::array< std::array<int, mat_dim<space_dim>>, 
                            mat_dim<space_dim> >;

    // (i, j) indices in Q-tensor given Q-vector index
    template <int space_dim>
    constexpr vec<space_dim> Q_row = {{}};
    template <int space_dim>
    constexpr vec<space_dim> Q_col = {{}};
    
    // explicit indices for 3- and 2-spatial dimensions
    template <>
    constexpr vec<3> Q_row<3> = { {0, 0, 0, 1, 1} };
    template <>
    constexpr vec<3> Q_col<3> = { {0, 1, 2, 1, 2} };
    template <>
    constexpr vec<2> Q_row<2> = { {0, 0, 1} };
    template <>
    constexpr vec<2> Q_col<2> = { {0, 1, 1} };

    // Q-vector indices given Q-tensor (i, j) indices (except (3, 3) entry)
    template <int space_dim>
    constexpr mat<space_dim> Q_idx = {{{0, 1, 2}, {1, 3, 4}, {2, 4, 0}}};

    // Kronecker delta
    template <int space_dim>
    constexpr mat<space_dim> delta = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};

    // Vector delta
    template <int space_dim>
    constexpr vec<space_dim> delta_vec = {{}};
    template <>
    constexpr vec<3> delta_vec<3> = { {1, 0, 0, 1, 0} };
    template <>
    constexpr vec<2> delta_vec<2> = { {1, 0, 1} };
}

#endif