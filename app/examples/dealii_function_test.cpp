#include <iostream>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

using namespace dealii;

template <int dim>
class TestFunction : public Function<dim>
{
public:
    TestFunction()
        : Function <dim>()
    {}

    virtual double value(const Point <dim> &p,
                         const unsigned int component = 0) const override;
    virtual void vector_value(const Point<dim> &p,
                              Vector<double> & value) const override;
};

template <int dim>
double TestFunction<dim>::value(const Point <dim> &p,
                                const unsigned int /*component*/) const
{
    double return_value = 0.0;
    for (unsigned int i = 0; i < dim; ++i)
        return_value += 4.0 * std::pow(p(i), 4.0);

    return return_value;
}

template <int dim>
void TestFunction<dim>::vector_value(const Point<dim> &p,
        Vector<double> & value) const
{
    for (unsigned int i = 0; i < dim; ++i)
        value[i] = 4.0 * std::pow(p(i), 4.0);
}

int main()
{
    Point<3, double> x{1.0, 2.0, 3.0};
    std::cout << "Point is: " << x << std::endl;

    double y;
    TestFunction<3> test_function;
    y = test_function.value(x);

    std::cout << "Value is: " << y << std::endl;

    double *z_ar = new double[3];
    z_ar[0] = 5;
    z_ar[1] = 2;
    z_ar[2] = 3;

    Vector<double> z(z_ar, z_ar + 3);
    std::cout << z << std::endl;
    Vector<double> w(3);
    std::cout << w << std::endl;
    test_function.vector_value(x, z);
    test_function.vector_value(x, w);
    std::cout << "Vector is: " << z << std::endl;
    std::cout << "Vector is: " << w << std::endl;

    FullMatrix<double> m(3, 1, z_ar);
    m.print(std::cout);

    return 0;
}
