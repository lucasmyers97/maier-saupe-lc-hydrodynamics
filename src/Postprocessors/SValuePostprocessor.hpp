#ifndef S_VALUE_POSTPROCESSOR_HPP
#define S_VALUE_POSTPROCESSOR_HPP

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_postprocessor.h>

#include "maier_saupe_constants.hpp"

namespace msc = maier_saupe_constants;

template <int dim>
class SValuePostprocessor : public dealii::DataPostprocessorScalar<dim>
{
public:
    SValuePostprocessor(std::string suffix)
        :
        dealii::DataPostprocessorScalar<dim> ("S_" + suffix,
                                              dealii::update_values)
    {}

    virtual void evaluate_vector_field
    (const dealii::DataPostprocessorInputs::Vector<dim> &input_data,
     std::vector<dealii::Vector<double>> &computed_quantities) const override
    {
        AssertDimension(input_data.solution_values.size(),
                        computed_quantities.size());

        const double lower_bound = -5.0;
        const double upper_bound = 5.0;
        const double abs_accuracy = 1e-8;

        dealii::LAPACKFullMatrix<double> Q(msc::mat_dim<dim>,
                                           msc::mat_dim<dim>);
        dealii::FullMatrix<double> eigenvecs(msc::mat_dim<dim>,
                                             msc::mat_dim<dim>);
        dealii::Vector<double> eigenvals(msc::mat_dim<dim>);
        for (unsigned int p=0; p<input_data.solution_values.size(); ++p)
        {
            AssertDimension(computed_quantities[p].size(), 1);

            Q.reinit(msc::mat_dim<dim>, msc::mat_dim<dim>);
            eigenvecs.reinit(msc::mat_dim<dim>, msc::mat_dim<dim>);
            eigenvals.reinit(msc::mat_dim<dim>);

            // generate Q-tensor
            Q(0, 0) = input_data.solution_values[p][0];
            Q(0, 1) = input_data.solution_values[p][1];
            Q(0, 2) = input_data.solution_values[p][2];
            Q(1, 1) = input_data.solution_values[p][3];
            Q(1, 2) = input_data.solution_values[p][4];
            Q(1, 0) = Q(0, 1);
            Q(2, 0) = Q(0, 2);
            Q(2, 1) = Q(1, 2);
            Q(2, 2) = -(Q(0, 0) + Q(1, 1));

            Q.compute_eigenvalues_symmetric(lower_bound, upper_bound,
                                            abs_accuracy, eigenvals,
                                            eigenvecs);

            if (eigenvals.size() != 0) {
              // Find index of maximal eigenvalue
              auto max_element_iterator =
                  std::max_element(eigenvals.begin(), eigenvals.end());
              long int max_entry{
                  std::distance(eigenvals.begin(), max_element_iterator)};
              computed_quantities[p][0] = (3.0 / 2.0) * eigenvals(max_entry);
            }
            else {
                computed_quantities[p][0] = 0;
            }
        }
    }
};

#endif
