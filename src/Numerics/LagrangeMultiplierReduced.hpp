#ifndef LAGRANGE_MULTIPLIER_REDUCED_HPP
#define LAGRANGE_MULTIPLIER_REDUCED_HPP

#include <vector>
#include <array>
#include <fstream>

#include <boost/serialization/access.hpp>

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>

#include "Utilities/maier_saupe_constants.hpp"

/**
 * \brief Takes in a \f$Q\f$-tensor, outputs corresponding \f$\Lambda\f$ value
 * using the Newton-Rhapson method, where \f$\Lambda\f$ is given by:
 *
 * \f{equation}{
 *   Q_{ij} 
 *   = \frac{\partial \log Z}{\partial \Lambda_{ij}} 
 *   - \frac13 \delta_{ij}
 * \f}
 * with
 * \f{equation}{
 *   Z[\Lambda] = \int_{S^2} \exp[\Lambda_{ij} \xi_i \xi_j] \: d \xi
 * \f}
 * A typical use-case of this class is as follows:
 * @code
 * #include <deal.II/lac/vector.h>
 * #include <iostream>
 * 
 * const int order = 590;
 * 
 * double alpha = 1.0;
 * double tol = 1e-12;
 * int max_iters = 12;
 *
 * dealii::Vector<double> Q({6.0 / 10.0, 0.0, 0.0, -3.0 / 10.0, 0.0});
 * dealii::Vector<double> Lambda();
 * Lambda.reinit(5);
 *
 * LagrangeMultiplierReduced<order> lm(alpha, tol, max_iters);
 * lm.invertQ(Q);
 * lm.returnLambda(Lambda);
 * std::cout << Lambda << std::endl;
 * @endcode
 */
template <int space_dim>
class LagrangeMultiplierReduced
{
public:

	/**
	 * \brief Constructor specifies Newton-Rhapson parameters (step size, error
     * tolerance for convergence, max number of iterations before failure).
     *
     * @param[in] alpha_ Gives step size for Newton-Rhapson method:
     *                   \f$\Lambda_i^{n + 1}
     *                      = \Lambda_i^n + \alpha \delta \Lambda_i^n\f$
     * @param[in] tol_ Gives error tolerance for Newton-Rhapson method to
     *                 terminate:
     *                 \f$|R_m(\Lambda^n)| < \epsilon \f$ where \f$R_m\f$ is the
     *                 residual, and \f$ \epsilon\f$ is the tolerance.
     * @param[in] max_iter_ Maximum number of iterations for Newton-Rhapson
     *                      method -- after this many iterations, the method
     *                      terminates and `inverted` flag is set to false.
	 */
    LagrangeMultiplierReduced(const int order_,
                              const double alpha_,
                              const double tol_,
                              const unsigned int max_iter_);

    /** \brief Takes in Q-vector, and iteratively inverts the equation to solve
      * for `Lambda`.
      * Produces `Lambda`, `Z` (partition function), and Jacobian `Jac` in the
      * process of the inversion.
      *
      * @param[in] Q_in Q-vector which is to be inverted.
      *
      * @return Number of iterations done in Newton's method
      */
    unsigned int invertQ(const dealii::Tensor<1, 2, double> &Q_in);
    /** \brief Returns Z (partition function) corresponding to calculated Lambda
      * value.
      * Throws an exception if Q has not been inverted
      *
      * @return Calculated value of Z (partition function);
      */
    double returnZ() const;
    /** \brief Returns the calculated Lambda
      *
      * @param[out] outLambda Vector to which the value of Lambda will be copied
      */
    dealii::Tensor<1, 2, double> returnLambda() const;
    /** \brief Returns Jacobian corresponding to calculated Lambda
      *
      * @param[out] outJac Matrix to which the Jacobian will be copied
      */
    dealii::Tensor<2, 2, double> returnJac();

private:
    // For implementing Newton's method
	/**
	 * \brief Initializes the Newton's method inversion scheme. 
	 *
	 * Initializing the Newton's method inversion scheme involves setting 
     * default (i.e. Lambda = 0) values for the residual and Jacobian.
	 *
	 * @param[in] Q_in Vector holding degrees of freedom of Q which is to be
     *                 inverted.
	 */
    void initializeInversion(const dealii::Tensor<1, 2, double> &Q_in);
	/**
	 * \brief Updates the residual and Jacobian for the current values of Lambda
     * and Q.
	 */
    void updateResJac();
	/**
	 * \brief caluclate Newton update `dLambda` given a particular residual and
	 * Jacobian. Need to have called `updateResJac` after applying the last
     * Newton update for this function to make sense, otherwise the Jacobian
     * will be in LU-factorized form (and therefore unusable).
	 */
    void updateVariation();

    // Helpers for computations
	/**
	 * \brief Calculate \f$\Lambda_{kl} \xi_k \xi_l\f$ for some particular
	 * \f$\Lambda\f$ where repeated indices are summed over. Here \f$\xi\f$ is
	 * a point in 3-space
	 *
	 * @param[in] x Point in 3-space with which this sum is being calculated
	 */
    double lambdaSum
            (const dealii::Point<maier_saupe_constants::mat_dim<space_dim>> &x)
            const;

    // Add ability to serialize object
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & inverted;
        ar & Jac_updated;
        ar & alpha;
        ar & tol;
        ar & max_iter;
        ar & Q;
        ar & Lambda;
        ar & Res;
        ar & Jac;
        ar & dLambda;
        ar & Z;
    }

    bool inverted;
    /** \brief Flag indicating whether `Jac` has been updated to current
      * `Lambda` value.
      * Turned false when `Jac` is LU-factorized, because it is unusable.
      */
    bool Jac_updated;

    // Newton's method parameters
    double alpha;
    double tol;
    unsigned int max_iter;

    // Vector to invert, and inversion solution
    /** \brief Vector holding Q degrees of freedom */
    dealii::Tensor<1, 2, double> Q;
    /** \brief Vector holding Lambda degrees of freedom */
    dealii::Tensor<1, 2, double> Lambda;

    // Interim variables for Newton's method
    /** \brief Vector holding Residual for Newton's method */
    dealii::Tensor<1, 2, double> Res;
    /** \brief Matrix holding Jacobian -- can be inverted by LU-factorization */
    dealii::Tensor<2, 2, double> Jac;
    /** \brief Vector holding variation (step) in Newton's method */
    dealii::Tensor<1, 2, double> dLambda;
    /** \brief Z-value (partition function) corresponding to current Lambda */
    double Z;

    struct ReducedLebedevCoords
    {
    public:
        std::vector<double> x;
        std::vector<double> y;
        std::vector<double> w;
    };

    ReducedLebedevCoords makeLebedevCoords(const int order);
    const ReducedLebedevCoords leb;
};

#endif
