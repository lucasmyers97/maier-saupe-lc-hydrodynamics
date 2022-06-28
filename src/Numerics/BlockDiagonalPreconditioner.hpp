#ifndef BLOCK_DIAGONAL_PRECONDITIONER_HPP
#define BLOCK_DIAGONAL_PRECONDITIONER_HPP

#include <deal.II/base/subscriptor.h>

template <class PreconditionerA, class PreconditionerS, class BlockVector>
class BlockDiagonalPreconditioner : public dealii::Subscriptor
{
public:
    BlockDiagonalPreconditioner(const PreconditionerA &preconditioner_A,
                                const PreconditionerS &preconditioner_S);

    void vmult(BlockVector &dst, const BlockVector &src) const;

private:
    const PreconditionerA &preconditioner_A;
    const PreconditionerS &preconditioner_S;
};



template <class PreconditionerA, class PreconditionerS, class BlockVector>
BlockDiagonalPreconditioner<PreconditionerA, PreconditionerS, BlockVector>::
BlockDiagonalPreconditioner(const PreconditionerA &preconditioner_A,
                            const PreconditionerS &preconditioner_S)
    : preconditioner_A(preconditioner_A)
    , preconditioner_S(preconditioner_S)
{}



template <class PreconditionerA, class PreconditionerS, class BlockVector>
void BlockDiagonalPreconditioner<PreconditionerA, PreconditionerS, BlockVector>::
vmult(BlockVector &dst, const BlockVector &src) const
{
    preconditioner_A.vmult(dst.block(0), src.block(0));
    preconditioner_S.vmult(dst.block(1), src.block(1));
}

#endif
