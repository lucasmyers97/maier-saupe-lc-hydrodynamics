#ifndef LIQUID_CRYSTAL_POSTPROCESSOR
#define LIQUID_CRYSTAL_POSTPROCESSOR

#include <deal.II/numerics/data_postprocessor.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <string>
#include <vector>

template <int dim>
class DirectorPostprocessor : public dealii::DataPostprocessorVector<dim>
{
public:
	DirectorPostprocessor(std::string suffix);

	virtual void evaluate_vector_field
	(const dealii::DataPostprocessorInputs::Vector<dim> &input_data,
	 std::vector<dealii::Vector<double>> &computed_quantities) const override;

private:
	int mat_dim = 3;
	int vec_dim = 5;
};



template <int dim>
DirectorPostprocessor<dim>::DirectorPostprocessor(std::string suffix)
 : dealii::DataPostprocessorVector<dim>("n_" + suffix, dealii::update_values)
{}



template <int dim>
void DirectorPostprocessor<dim>::evaluate_vector_field
(const dealii::DataPostprocessorInputs::Vector<dim> &input_data, 
 std::vector<dealii::Vector<double>> &computed_quantities) const
{
	// dealii::AssertDimension(input_data.solution_values.size(),
	// 						computed_quantities.size());

	const double lower_bound = -5.0;
	const double upper_bound = 5.0;
	const double abs_accuracy = 1e-8;

	dealii::LAPACKFullMatrix<double> Q(mat_dim, mat_dim);
	dealii::FullMatrix<double> eigenvecs(mat_dim, mat_dim);
	dealii::Vector<double> eigenvals(mat_dim);
	for (unsigned int p=0; p<input_data.solution_values.size(); ++p)
	{
		// dealii::AssertDimension(computed_quantities[p].size(), dim);

		Q.reinit(mat_dim, mat_dim);
		eigenvecs.reinit(mat_dim, mat_dim);
		eigenvals.reinit(mat_dim);

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

		// Find index of maximal eigenvalue
		auto max_element_iterator = std::max_element(eigenvals.begin(),
													 eigenvals.end());
		long int max_entry{std::distance(eigenvals.begin(),
											max_element_iterator)};
		computed_quantities[p][0] = eigenvecs(0, max_entry);
		computed_quantities[p][1] = eigenvecs(1, max_entry);
		if (dim == 3) { computed_quantities[p][2] = eigenvecs(2, max_entry); }
	}

}



template <int dim>
class SValuePostprocessor : public dealii::DataPostprocessorScalar<dim>
{
public:
	SValuePostprocessor(std::string suffix);

	virtual void evaluate_vector_field
	(const dealii::DataPostprocessorInputs::Vector<dim> &input_data,
	 std::vector<dealii::Vector<double>> &computed_quantities) const override;

private:
	static constexpr int mat_dim = 3;
};



template <int dim>
SValuePostprocessor<dim>::SValuePostprocessor(std::string suffix)
	:
	dealii::DataPostprocessorScalar<dim> ("S_" + suffix, dealii::update_values)
{}



template <int dim>
void SValuePostprocessor<dim>::evaluate_vector_field
(const dealii::DataPostprocessorInputs::Vector<dim> &input_data,
	std::vector<dealii::Vector<double>> &computed_quantities) const
{
	// AssertDimension(input_data.solution_values.size(),
	// 				computed_quantities.size());

	const double lower_bound = -5.0;
	const double upper_bound = 5.0;
	const double abs_accuracy = 1e-8;

	dealii::LAPACKFullMatrix<double> Q(mat_dim, mat_dim);
	dealii::FullMatrix<double> eigenvecs(mat_dim, mat_dim);
	dealii::Vector<double> eigenvals(mat_dim);
	for (unsigned int p=0; p<input_data.solution_values.size(); ++p)
	{
		// AssertDimension(computed_quantities[p].size(), 1);

		Q.reinit(mat_dim, mat_dim);
		eigenvecs.reinit(mat_dim, mat_dim);
		eigenvals.reinit(mat_dim);

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

		// Find index of maximal eigenvalue
		auto max_element_iterator = std::max_element(eigenvals.begin(),
														eigenvals.end());
		long int max_entry{std::distance(eigenvals.begin(),
											max_element_iterator)};
		computed_quantities[p][0] = (3.0 / 2.0) * eigenvals(max_entry);
	}
}

#endif