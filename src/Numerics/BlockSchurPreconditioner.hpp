#ifndef BLOCK_SCHUR_PRECONDITIONER_HPP
#define BLOCK_SCHUR_PRECONDITIONER_HPP

#include <deal.II/base/subscriptor.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_cg.h>

template <class PreconditionerTypeA, class PreconditionerTypeMp,
          class BlockMatrix, class BlockVector, class Vector>
class BlockSchurPreconditioner : public dealii::Subscriptor
{
public:
    BlockSchurPreconditioner(const BlockMatrix &S,
                             const BlockMatrix &Spre,
                             const PreconditionerTypeMp &Mppreconditioner,
                             const PreconditionerTypeA & Apreconditioner,
                             const bool                  do_solve_A);

    void vmult(BlockVector &      dst, const BlockVector &src) const;

private:
    const dealii::SmartPointer<const BlockMatrix> stokes_matrix;
    const dealii::SmartPointer<const BlockMatrix>
    stokes_preconditioner_matrix;
    const PreconditionerTypeMp &mp_preconditioner;
    const PreconditionerTypeA &a_preconditioner;
    const bool do_solve_A;
};



template <class PreconditionerTypeA, class PreconditionerTypeMp,
          class BlockMatrix, class BlockVector, class Vector>
BlockSchurPreconditioner<PreconditionerTypeA, PreconditionerTypeMp,
                         BlockMatrix, BlockVector, Vector>::
BlockSchurPreconditioner(const BlockMatrix &S,
                         const BlockMatrix &Spre,
                         const PreconditionerTypeMp &Mppreconditioner,
                         const PreconditionerTypeA & Apreconditioner,
                         const bool                  do_solve_A)
    : stokes_matrix(&S)
    , stokes_preconditioner_matrix(&Spre)
    , mp_preconditioner(Mppreconditioner)
    , a_preconditioner(Apreconditioner)
    , do_solve_A(do_solve_A)
{}



template <class PreconditionerTypeA, class PreconditionerTypeMp,
          class BlockMatrix, class BlockVector, class Vector>
void BlockSchurPreconditioner<PreconditionerTypeA, PreconditionerTypeMp,
                              BlockMatrix, BlockVector, Vector>::
vmult(BlockVector &dst, const BlockVector &src) const
{
    Vector utmp(src.block(0));
    {
        dealii::SolverControl solver_control(5000, 1e-6 * src.block(1).l2_norm());

        dealii::SolverCG<Vector> solver(solver_control);

        solver.solve(stokes_preconditioner_matrix->block(1, 1),
                     dst.block(1),
                     src.block(1),
                     mp_preconditioner);

        dst.block(1) *= -1.0;
    }

    {
        stokes_matrix->block(0, 1).vmult(utmp, dst.block(1));
        utmp *= -1.0;
        utmp.add(src.block(0));
    }

    if (do_solve_A == true)
    {
        dealii::SolverControl solver_control(5000, utmp.l2_norm() * 1e-2);
        dealii::SolverCG<Vector> solver(solver_control);
        solver.solve(stokes_matrix->block(0, 0),
                     dst.block(0),
                     utmp,
                     a_preconditioner);
    }
    else
        a_preconditioner.vmult(dst.block(0), utmp);
}

#endif
