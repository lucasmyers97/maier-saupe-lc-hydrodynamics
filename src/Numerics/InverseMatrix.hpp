#ifndef INVERSE_MATRIX_HPP
#define INVERSE_MATRIX_HPP

#include <deal.II/base/subscriptor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_cg.h>

#include <exception>
#include <cassert>

template <class MatrixType, class PreconditionerType>
class InverseMatrix : public dealii::Subscriptor
{
public:
    InverseMatrix(const MatrixType & m,
                  const PreconditionerType &preconditioner);

    template <typename VectorType = dealii::Vector<double>>
    void vmult(VectorType &dst, const VectorType &src) const;

private:
    const dealii::SmartPointer<const MatrixType>         matrix;
    const dealii::SmartPointer<const PreconditionerType> preconditioner;
};


template <class MatrixType, class PreconditionerType>
InverseMatrix<MatrixType, PreconditionerType>::
InverseMatrix(const MatrixType &m, const PreconditionerType &preconditioner)
    : matrix(&m)
    , preconditioner(&preconditioner)
{}

template <class MatrixType, class PreconditionerType>
template <typename VectorType>
void InverseMatrix<MatrixType, PreconditionerType>::
vmult(VectorType &dst, const VectorType &src) const
{
    dealii::SolverControl solver_control(src.size(), 1e-8 * src.l2_norm());
    dealii::SolverCG<VectorType> cg(solver_control);

    dst = 0;

    try
    {
        cg.solve(*matrix, dst, src, *preconditioner);
    }
    catch (std::exception &e)
    {
        assert(false && e.what());
    }
}

#endif
