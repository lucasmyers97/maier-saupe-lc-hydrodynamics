#ifndef PERIODIC_BOUNDARIES_HPP
#define PERIODIC_BOUNDARIES_HPP

#include <deal.II/base/types.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_tools.h>

template <int space_dim>
class PeriodicBoundaries 
{
public:
    PeriodicBoundaries(const dealii::types::boundary_id b_id1,
                       const dealii::types::boundary_id b_id2,
                       const unsigned int direction,
                       const dealii::Tensor< 1, space_dim >& 
                       offset =  dealii::Tensor<1, space_dim>(),
                       const dealii::FullMatrix<double>& 
                       rotation = dealii::FullMatrix<double>() )

        : b_id1(b_id1)
        , b_id2(b_id2)
        , direction(direction)
        , offset(offset)
        , rotation(rotation)
    {};

    template <typename MeshType>
    void apply_to_triangulation(MeshType& triangulation) const
    {
        using PeriodicFaces
            = std::vector<dealii::GridTools::PeriodicFacePair<
                typename MeshType::cell_iterator
                    >
                >;

        PeriodicFaces periodic_faces;
        dealii::GridTools::collect_periodic_faces(triangulation,
                                                  b_id1,
                                                  b_id2,
                                                  direction,
                                                  periodic_faces,
                                                  offset,
                                                  rotation);

        triangulation.add_periodicity(periodic_faces);
    }

    template <typename MeshType>
    void apply_to_constraints(const MeshType& dof_handler, 
                              dealii::AffineConstraints<double> &constraints) const
    {
        using PeriodicFaces
            = std::vector<dealii::GridTools::PeriodicFacePair<
                typename MeshType::cell_iterator
                    >
                >;

        PeriodicFaces periodic_faces;
        dealii::GridTools::collect_periodic_faces(dof_handler,
                                                  b_id1,
                                                  b_id2,
                                                  direction,
                                                  periodic_faces);

        dealii::DoFTools::
            make_periodicity_constraints<MeshType::dimension, space_dim>(periodic_faces,
                                                                         constraints);
    }

private:
    const dealii::types::boundary_id b_id1;
    const dealii::types::boundary_id b_id2;
    const unsigned int direction;
    const dealii::Tensor<1, space_dim> offset;
 	const dealii::FullMatrix<double> rotation;
};

#endif
