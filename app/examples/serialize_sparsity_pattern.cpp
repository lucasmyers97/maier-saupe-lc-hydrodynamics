#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/array.hpp>

#include <boost/serialization/array_wrapper.hpp>
#include <fstream>
#include <iostream>
#include <memory>

#define protected public
#include <deal.II/lac/sparsity_pattern.h>

int main()
{
    std::unique_ptr<std::size_t[]> rowstart;
    std::cout << rowstart.get() << std::endl;

    std::ofstream ofs("sparsity_test.dat");
    {
        boost::archive::text_oarchive oa(ofs);
        oa << boost::serialization::make_array(rowstart.get(), 1);
    }

    // dealii::SparsityPattern sp;

    // std::cout << sp.rowstart.get() << std::endl;

    // std::ofstream ofs("sparsity_test.dat");
    // {
    //     boost::archive::text_oarchive oa(ofs);
    //     oa << sp;
    // }

    return 0;
}
