#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "sphere_lebedev_rule.hpp"
#define private public
#include "LagrangeMultiplierGPU.hpp"

namespace{
    constexpr int order{590};
    constexpr int vec_dim{5};
    constexpr int space_dim{3};
}

__global__
void setLebedev(double *lebedev_coords, double *lebedev_weights,
                double *out_lebedev_coords, double *out_lebedev_weights)
{
    extern __shared__ LagrangeMultiplierGPU<double, order, vec_dim> lm[];

    lm[0].setLebedevData(lebedev_coords, lebedev_weights);
    
    int x_coord{};
    int y_coord{};
    int z_coord{};
    for (int i = 0; i < order; ++i)
    {
        x_coord = i*space_dim;
        y_coord = i*space_dim + 1;
        z_coord = i*space_dim + 2;

        out_lebedev_coords[x_coord] = lm[0].lebedev_coords[x_coord];
        out_lebedev_coords[y_coord] = lm[0].lebedev_coords[y_coord];
        out_lebedev_coords[z_coord] = lm[0].lebedev_coords[z_coord];
        out_lebedev_weights[i] = lm[0].lebedev_weights[i];
    }
}

BOOST_AUTO_TEST_CASE(set_lebedev_test)
{
    double *x, *y, *z, *w;
    x = new double[order];
    y = new double[order];
    z = new double[order];
    w = new double[order];

    ld_by_order(order, x, y, z, w);

    double *lebedev_coords = new double[3*order];
    double *lebedev_weights = new double[order];

    for (int i = 0; i < order; ++i)
    {
        lebedev_coords[space_dim*i] = x[i];
        lebedev_coords[space_dim*i + 1] = y[i];
        lebedev_coords[space_dim*i + 2] = z[i];
        lebedev_weights[i] = w[i];
    }

    double *d_lebedev_coords;
    double *d_lebedev_weights;
    cudaMalloc(&d_lebedev_coords, space_dim*order*sizeof(double));
    cudaMalloc(&d_lebedev_weights, order*sizeof(double));

    cudaMemcpy(d_lebedev_coords, lebedev_coords,
               space_dim*order*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lebedev_weights, lebedev_weights,
               order*sizeof(double), cudaMemcpyHostToDevice);

    for (int i = 0; i < order; ++i)
    {
        lebedev_coords[space_dim*i] = 0;
        lebedev_coords[space_dim*i + 1] = 0;
        lebedev_coords[space_dim*i + 2] = 0;
        lebedev_weights[i] = 0;
    }

    double *d_out_lebedev_coords;
    double *d_out_lebedev_weights;
    cudaMalloc(&d_out_lebedev_coords, space_dim*order*sizeof(double));
    cudaMalloc(&d_out_lebedev_weights, order*sizeof(double));

    setLebedev
		<<<1, 1, sizeof(LagrangeMultiplierGPU<double, order, vec_dim>)>>>
		(d_lebedev_coords, d_lebedev_weights,
        d_out_lebedev_coords, d_out_lebedev_weights);

    cudaFree(d_lebedev_coords);
    cudaFree(d_lebedev_weights);

    cudaMemcpy(lebedev_coords, d_out_lebedev_coords, 
               space_dim*order*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(lebedev_weights, d_out_lebedev_weights, 
               space_dim*order*sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_out_lebedev_coords);
    cudaFree(d_out_lebedev_weights);
            
    for (int i = 0; i < order; ++i)
    {
        BOOST_TEST(lebedev_coords[space_dim*i] == x[i]);
        BOOST_TEST(lebedev_coords[space_dim*i + 1] == y[i]);
        BOOST_TEST(lebedev_coords[space_dim*i + 2] == z[i]);
        BOOST_TEST(lebedev_weights[i] == w[i]);
    }

    delete[] lebedev_coords;
    delete[] lebedev_weights;
    delete[] x;
    delete[] y;
    delete[] z;
    delete[] w;
}