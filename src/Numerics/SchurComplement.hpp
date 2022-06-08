#ifndef SCHUR_COMPLEMENT_HPP
#define SCHUR_COMPLEMENT_HPP

#include <deal.II/base/subscriptor.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/lac/vector.h>

#include "Numerics/InverseMatrix.hpp"

template <class PreconditionerType>
class SchurComplement : public dealii::Subscriptor
{
public:
    SchurComplement
    (const dealii::BlockSparseMatrix<double> &system_matrix,
     const InverseMatrix<dealii::SparseMatrix<double>, PreconditionerType> &A_inverse);

    void vmult(dealii::Vector<double> &dst,
               const dealii::Vector<double> &src) const;

private:
    const dealii::SmartPointer<const dealii::BlockSparseMatrix<double>> system_matrix;
    const dealii::SmartPointer<
        const InverseMatrix<dealii::SparseMatrix<double>, PreconditionerType>>
    A_inverse;

    mutable dealii::Vector<double> tmp1, tmp2;
};



template <class PreconditionerType>
SchurComplement<PreconditionerType>::SchurComplement
    (const dealii::BlockSparseMatrix<double> &system_matrix,
     const InverseMatrix<dealii::SparseMatrix<double>, PreconditionerType> &A_inverse)
    : system_matrix(&system_matrix)
    , A_inverse(&A_inverse)
    , tmp1(system_matrix.block(0, 0).m())
    , tmp2(system_matrix.block(0, 0).m())
{}


template <class PreconditionerType>
void SchurComplement<PreconditionerType>::vmult
    (dealii::Vector<double> &      dst,
     const dealii::Vector<double> &src) const
{
    system_matrix->block(0, 1).vmult(tmp1, src);
    A_inverse->vmult(tmp2, tmp1);
    system_matrix->block(1, 0).vmult(dst, tmp2);
}

#endif
