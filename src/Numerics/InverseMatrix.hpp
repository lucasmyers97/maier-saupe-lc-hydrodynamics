#ifndef INVERSE_MATRIX_HPP
#define INVERSE_MATRIX_HPP

#include <deal.II/base/subscriptor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_cg.h>

template <class MatrixType, class PreconditionerType>
class InverseMatrix : public dealii::Subscriptor
{
public:
    InverseMatrix(const MatrixType & m,
                  const PreconditionerType &preconditioner);

    void vmult(dealii::Vector<double> &dst,
               const dealii::Vector<double> &src) const;

private:
    const dealii::SmartPointer<const MatrixType>         matrix;
    const dealii::SmartPointer<const PreconditionerType> preconditioner;
};


template <class MatrixType, class PreconditionerType>
InverseMatrix<MatrixType, PreconditionerType>::InverseMatrix
    (const MatrixType &        m,
     const PreconditionerType &preconditioner)
    : matrix(&m)
    , preconditioner(&preconditioner)
{}

template <class MatrixType, class PreconditionerType>
void InverseMatrix<MatrixType, PreconditionerType>::vmult
    (dealii::Vector<double> &      dst,
     const dealii::Vector<double> &src) const
{
    dealii::SolverControl solver_control(src.size(), 1e-6 * src.l2_norm());
    dealii::SolverCG<dealii::Vector<double>> cg(solver_control);

    dst = 0;

    cg.solve(*matrix, dst, src, *preconditioner);
}

#endif
