#include <deal.II/lac/vector.h>
#include <algorithm>
#include <iostream>

int main()
{
  double x[] = {5.0, 3.7, 2.9, 7.8, 1.0};
  dealii::Vector<double> v(&x[0], &x[4]);

  auto max_element_ptr = std::max_element(v.begin(), v.end());
  auto max_idx = std::distance(v.begin(), max_element_ptr);
  std::cout << max_idx << std::endl;
}
