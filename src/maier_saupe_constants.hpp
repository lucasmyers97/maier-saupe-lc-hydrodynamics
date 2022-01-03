#ifndef MAIER_SAUPE_CONSTANTS_HPP
#define MAIER_SAUPE_CONSTANTS_HPP

#include <array> 

namespace maier_saupe_constants{

    template <int space_dim>
    struct backend {};

    template <>
    struct backend<2>
    {
        // TODO: when everything genuinely works in 2D, uncomment this line
        static constexpr int vec_dim = 5;
        static constexpr int mat_dim = 3;

        using vec = std::array<int, vec_dim>;
        using mat = std::array< std::array<int, mat_dim>, mat_dim>;

        static constexpr vec Q_row = { {0, 0, 1} };
        static constexpr vec Q_col = { {0, 1, 1} };

        static constexpr mat Q_idx = {{{0, 1, 2}, {1, 3, 4}, {2, 4, 0}}};
        static constexpr mat delta = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
        static constexpr vec delta_vec = { {1, 0, 1} };
    };

    template <>
    struct backend<3>
    {
        static constexpr int vec_dim = 5;
        static constexpr int mat_dim = 3;

        using vec = std::array<int, vec_dim>;
        using mat = std::array< std::array<int, mat_dim>, mat_dim>;

        static constexpr vec Q_row = { {0, 0, 0, 1, 1} };
        static constexpr vec Q_col = { {0, 1, 2, 1, 2} };

        static constexpr mat Q_idx = {{{0, 1, 2}, {1, 3, 4}, {2, 4, 0}}};
        static constexpr mat delta = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
        static constexpr vec delta_vec = { {1, 0, 0, 1, 0} };
    };

    // dim of (i) vector representing Q-tensor, and (ii) matrix of Q-tensor
    template <int space_dim>
    constexpr int vec_dim = backend<space_dim>::vec_dim;
    template <int space_dim>
    constexpr int mat_dim = backend<space_dim>::mat_dim;

    // alias Q-tensor vector and matrix classes
    template <int space_dim>
    using vec = std::array<int, vec_dim<space_dim>>;
    template <int space_dim>
    using mat = std::array< std::array<int, mat_dim<space_dim>>,
                            mat_dim<space_dim> >;

    // (i, j) indices in Q-tensor given Q-vector index
    template <int space_dim>
    vec<space_dim> Q_row = backend<space_dim>::Q_row;
    template <int space_dim>
    vec<space_dim> Q_col = backend<space_dim>::Q_col;
    
    // Q-vector indices given Q-tensor (i, j) indices (except (3, 3) entry)
    template <int space_dim>
    mat<space_dim> Q_idx = backend<space_dim>::Q_idx;
    // Kronecker delta
    template <int space_dim>
    mat<space_dim> delta = backend<space_dim>::delta;
    // Vector delta
    template <int space_dim>
    vec<space_dim> delta_vec = backend<space_dim>::delta_vec;
}

#endif
