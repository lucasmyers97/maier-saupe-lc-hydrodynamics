#ifndef LAGRANGEMULTIPLIER_HPP
#define LAGRANGEMULTIPLIER_HPP

#include <functional>
#include <vector>
#include <array>
#include <fstream>

#include <boost/serialization/access.hpp>

#include <deal.II/base/point.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/base/function.h>

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
 * LagrangeMultiplier<order> lm(alpha, tol, max_iters);
 * lm.invertQ(Q);
 * lm.returnLambda(Lambda);
 * std::cout << Lambda << std::endl;
 * @endcode
 */
template <int order, int space_dim = 3>
class LagrangeMultiplier
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
    LagrangeMultiplier(const double alpha_,
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
    unsigned int invertQ(const dealii::Vector<double> &Q_in);
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
    void returnLambda(dealii::Vector<double> &outLambda) const;
    /** \brief Returns Jacobian corresponding to calculated Lambda
      *
      * @param[out] outJac Matrix to which the Jacobian will be copied
      */
    void returnJac(dealii::LAPACKFullMatrix<double> &outJac);

    // void lagrangeTest();

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
    void initializeInversion(const dealii::Vector<double> &Q_in);
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
            (const dealii::Point<maier_saupe_constants::mat_dim<space_dim>> x) 
            const;
	/**
	 * \brief Calculate one term in the lebedev quadrature sum corresponding to
	 * the following integral: 
	 * \f$\int_{S^2} \xi_{i(m)} \xi_{j(m)} 
	 * \exp[\Lambda_{kl} \xi_k \xi_l] d\xi\f$
	 * where \f$i(m)\f$ is the row index in the Q-tensor corresponding to the
	 * \f$m\f$th entry of the Q-vector, and \f$j(m)\f$ is the column index in
	 * the Q-tensor corresponding to the \f$m\f$th entry of the Q-vector. 
	 * The quadrature sum is calculated one term at a time to avoid having to
	 * calculate the same \f$\exp[\Lambda_{kl} \xi_k \xi_l]\f$ multiple times.
	 *
	 * @param[in] exp_lambda Holds the value of 
	 						 \f$\exp[\Lambda_{kl} \xi_k \xi_l]\f$
	 *					 	 for a particular \f$\Lambda\f$ and Lebedev point.
	 * @param[in] quad_idx Indexes which term in the quadrature sum we are
	 *					   calculating.
	 * @param[in] i_m Row index in Q-tensor corresponding to m-th entry of 
	 				  Q-vector
	 * @param[in] j_m Column index in Q-tensor corresponding to m-th entry of
	 *			  	  Q-vector.
	 */
    double calcInt1Term(const double exp_lambda, 
                        const int quad_idx, const int i_m, const int j_m) const;
	/**
	 * \brief Calculate one term in the lebedev quadrature sum corresponding to
	 * the following integral: 
	 * \f$\int_{S^2} \xi_{i(m)} \xi_{j(m)} \xi_{i(n)} \xi_{j(n)}
	 * \exp[\Lambda_{kl} \xi_k \xi_l] d\xi\f$
	 * where \f$i(m)\f$ and \f$j(m)\f$ are as in calcInt1Term().
     * See calcInt1Term() for an explanation of parameters.
	 */
    double calcInt2Term(const double exp_lambda, 
                        const int quad_idx, const int i_m, const int j_m, 
                        const int i_n, const int j_n) const;
	/**
	 * \brief Calculate one term in the lebedev quadrature sum corresponding to
	 * the following integral: 
	 * \f$\int_{S^2} \xi_{i(m)} \xi_{j(m)} 
	 * \left(\xi_{i(n)}^2 -  \xi_{3}^2\right)
	 * \exp[\Lambda_{kl} \xi_k \xi_l] d\xi\f$
	 * where \f$i(m)\f$ and \f$j(m)\f$ are as in calcInt1Term().
     * See calcInt1Term() for an explanation of parameters.
	 */
    double calcInt3Term(const double exp_lambda,
                        const int quad_idx,
                        const int i_m, const int j_m,
                        const int i_n, const int j_n) const;
	/**
	 * \brief Calculate one term in the lebedev quadrature sum corresponding to
	 * the following integral: 
	 * \f$\int_{S^2} \left(\xi_{i(n)}^2 - \xi_{3}\right)
	 * \exp[\Lambda_{kl} \xi_k \xi_l] d\xi\f$
	 * where \f$i(m)\f$ and \f$j(m)\f$ are as in calcInt1Term().
     * See calcInt1Term() for an explanation of parameters.
	 */
    double calcInt4Term(const double exp_lambda,
                        const int quad_idx, const int i_m, const int j_m) const;

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
        ar & int1;
        ar & int2;
        ar & int3;
        ar & int4;
    }

    // Flags
    /** \brief Flag indicating whether `Lambda` has been inverted yet */
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
    dealii::Vector<double> Q;
    /** \brief Vector holding Lambda degrees of freedom */
    dealii::Vector<double> Lambda;

    // Interim variables for Newton's method
    /** \brief Vector holding Residual for Newton's method */
    dealii::Vector<double> Res;
    /** \brief Matrix holding Jacobian -- can be inverted by LU-factorization */
    dealii::LAPACKFullMatrix<double> Jac;
    /** \brief Vector holding variation (step) in Newton's method */
    dealii::Vector<double> dLambda;
    /** \brief Z-value (partition function) corresponding to current Lambda */
    double Z;

    // Arrays for storing integrals
    using int_vec 
        = std::array<double, maier_saupe_constants::vec_dim<space_dim>>;
    using int_mat
        = std::array<int_vec, maier_saupe_constants::vec_dim<space_dim>>; 
    /** \brief Array holding partial sum of integral 1 */
    int_vec int1 = {0};
    /** \brief Array holding partial sum of integral 2 */
    int_mat int2 = {{0}};
    /** \brief Array holding partial sum of integral 3 */
    int_mat int3 = {{0}};
    /** \brief Array holding partial sum of integral 4 */
    int_vec int4 = {0};

    using coords =
        std::vector<dealii::Point<maier_saupe_constants::mat_dim<space_dim>>>;

    struct LebedevCoords
    {
        coords x;
        std::vector<double> w;
    };

    LebedevCoords makeLebedevCoords(const int order_);
    const LebedevCoords leb;

};

#endif
