#include <boost/test/tools/interface.hpp>
#include <boost/test/unit_test_suite.hpp>
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <deal.II/base/point.h>
#include <deal.II/lac/vector.h>

#include <iostream>
#include <fstream>
#include <cstdio>
#include "Utilities/maier_saupe_constants.hpp"

#define private public
#include "BoundaryValues/UniformConfiguration.hpp"
#include "BoundaryValues/DefectConfiguration.hpp"

namespace utf = boost::unit_test;

BOOST_AUTO_TEST_CASE(uniform_configuration_test, *utf::tolerance(1e-9))
{
    namespace msc = maier_saupe_constants;

    constexpr int n_pts = 100;
    constexpr int dim = 3;
    double S = 1.0;
    double phi = 0.0;
    UniformConfiguration<dim> uniform_configuration(S, phi);
    dealii::Point<dim> p({1.0, 0.0, 0.0});

    // Set up the correct vector for the uniform configuration
    dealii::Vector<double> correct_v(msc::vec_dim<dim>);
    correct_v[0] = 2.0/3.0;
    correct_v[1] = 0.0;
    correct_v[2] = 0.0;
    correct_v[3] = -1.0/3.0;
    correct_v[4] = -1.0/3.0;

    // Test single return with components 
    for (int i = 0; i < correct_v.size(); ++i)
        BOOST_TEST(uniform_configuration.value(p, /*component = */i) 
                   == correct_v[i]);


    // Test single return whole vector
    dealii::Vector<double> vec(msc::vec_dim<dim>);
    uniform_configuration.vector_value(p, vec);
    for (int i = 0; i < correct_v.size(); ++i)
        BOOST_TEST(vec[i] == correct_v[i]);

    // Test multiple return components
    std::vector<double> value_list(n_pts);
    std::vector<dealii::Point<dim>> point_list(n_pts);
    for (int i = 0; i < n_pts; ++i)
    {
        point_list[i][0] = i;
        point_list[i][1] = i / 2.0;
        point_list[i][2] = i / 3.0;
    }
    for (int component = 0; component < msc::vec_dim<dim>; ++component)
    {
        uniform_configuration.value_list(point_list, value_list, component);
        for (int pt = 0; pt < n_pts; pt++)
            BOOST_TEST(value_list[pt] == correct_v[component]);
    }

    // Test multiple return whole vectors
    std::vector<dealii::Vector<double>> 
        vector_list(n_pts, dealii::Vector<double>(msc::vec_dim<dim>));
    uniform_configuration.vector_value_list(point_list, vector_list);
    for (auto& vector : vector_list)
        for (int i = 0; i < msc::vec_dim<dim>; ++i)
            BOOST_TEST(vector[i] == correct_v[i]);
}



BOOST_AUTO_TEST_CASE(uniform_serialization_test, *utf::tolerance(1e-9))
{
    std::string filename = "uniform-configuration-archive.dat";

    constexpr int dim = 2;
    double S = 0.72;
    double phi = 2.31;
    UniformConfiguration<dim> uc(S, phi);
    UniformConfiguration<dim> new_uc;

    std::ofstream ofs(filename);
    {
        boost::archive::text_oarchive oa(ofs);
        oa << uc;
    }
    {
        std::ifstream ifs(filename);
        boost::archive::text_iarchive ia(ifs);
        ia >> new_uc;
    }

    BOOST_TEST(new_uc.S == uc.S);
    BOOST_TEST(new_uc.phi == uc.phi);

    BOOST_TEST(!remove(filename.c_str()));
}



BOOST_AUTO_TEST_CASE(defect_configuration_test, *utf::tolerance(1e-9))
{
    namespace msc = maier_saupe_constants;

    constexpr int n_pts = 100;
    constexpr int dim = 3;
    double S = 1.0;
    DefectConfiguration<dim> defect_configuration(S, DefectCharge::plus_half);
    dealii::Point<dim> p({1.0, 0.0, 0.0});

    // set up vector corresponding to single point on x axis
    dealii::Vector<double> correct_v(msc::vec_dim<dim>);
    correct_v[0] = 2.0/3.0;
    correct_v[1] = 0.0;
    correct_v[2] = 0.0;
    correct_v[3] = -1.0/3.0;
    correct_v[4] = 0.0;

    // Test single return with components 
    for (int i = 0; i < correct_v.size(); ++i)
        BOOST_TEST(defect_configuration.value(p, /*component = */i) 
                   == correct_v[i]);
}



BOOST_AUTO_TEST_CASE(defect_archive_test, *utf::tolerance(1e-9))
{
    std::string filename = "defect-configuration-archive.dat";

    constexpr int dim = 3;
    double S = 0.721;
    DefectConfiguration<dim> dc(S, DefectCharge::minus_half);
    dc.psi = 3.14;
    dc.center(0) = 22;
    dc.center(1) = 1.2;

    DefectConfiguration<dim> new_dc;

    std::ofstream ofs(filename);
    {
        boost::archive::text_oarchive oa(ofs);
        oa << dc;
    }
    {
        std::ifstream ifs(filename);
        boost::archive::text_iarchive ia(ifs);
        ia >> new_dc;
    }

    BOOST_TEST(new_dc.S0 == dc.S0);
    BOOST_TEST((new_dc.charge == dc.charge));
    BOOST_TEST(new_dc.k == dc.k);
    BOOST_TEST(new_dc.psi == dc.psi);
    BOOST_TEST(new_dc.center(0) == dc.center(0));
    BOOST_TEST(new_dc.center(1) == dc.center(1));
    BOOST_TEST(new_dc.center(2) == dc.center(2));

    BOOST_TEST(!remove(filename.c_str()));
}
