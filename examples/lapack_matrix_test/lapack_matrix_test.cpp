#include <deal.II/lac/lapack_full_matrix.h>

int main()
{
  dealii::LAPACKFullMatrix<double> mat(3, 3);

  return 0;
}
