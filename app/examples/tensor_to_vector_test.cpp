/**
 * This example quickly tests the TensorToVector class, which takes as a
 * template parameter a class derived from dealii::TensorFunction, and then
 * is a function which gives back a vector corresponding to the Q-tensor.
 * That is, the vector has entries (0, 0), (0, 1), (0, 2), (1, 1), and (1, 2)
 * in that order of the tensor outputted from the TensorFunction class.
 */

#include <deal.II/base/tensor_function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>

#include <memory>
#include <utility>
#include <iostream>

#include "ExampleFunctions/PeriodicQTensor.hpp"
#include "ExampleFunctions/TensorToVector.hpp"

int main()
{
    const int dim = 2;

    const double k = 3.14;
    const double eps = 0.1;

    std::unique_ptr<dealii::TensorFunction<2, dim, double>>
        tensor_function = std::make_unique<PeriodicQTensor<dim>>(k, eps);

    TensorToVector<dim, PeriodicQTensor<dim>> vector_function(k, eps);

    dealii::Point<dim> p(0.3, 0.8);
    dealii::Tensor<2, dim, double> tensor = tensor_function->value(p);

    dealii::Vector<double> vector(msc::vec_dim<dim>);
    vector_function.vector_value(p, vector);

    std::cout << tensor << "\n";
    std::cout << vector << std::endl;

    return 0;
}
