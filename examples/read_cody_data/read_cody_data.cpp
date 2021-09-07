#include <highfive/H5DataSet.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <highfive/H5File.hpp>
#include <highfive/H5Easy.hpp>

constexpr int vec_dim = 5;

int main()
{
    std::string filename = DATA_FILE_LOCATION;
    
    HighFive::File f(filename);
    size_t num_objects = f.getNumberObjects();
    std::vector<std::string> object_names = f.listObjectNames();

    for (int i = 0; i < num_objects; ++i)
        std::cout << object_names[i] << std::endl;


    using mat =  std::vector<std::vector<double>>;
    std::vector<mat> Q_vec(vec_dim);
    mat X;
    mat Y;

    std::vector<HighFive::DataSet> dset(num_objects);

    for (int i = 0; i < vec_dim; ++i)
    {
        dset[i] = f.getDataSet(object_names[i]);
        dset[i].read(Q_vec[i]);
    }

    dset[num_objects - 2] = f.getDataSet(object_names[num_objects - 2]);
    dset[num_objects - 2].read(X);
    dset[num_objects - 1] = f.getDataSet(object_names[num_objects - 1]);
    dset[num_objects - 1].read(Y);

    return 0;
}