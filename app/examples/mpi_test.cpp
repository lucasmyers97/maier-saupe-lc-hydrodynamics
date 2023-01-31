#include <mpi.h>

#include <exception>
#include <iostream>

int main()
{
    try
    {  
        int rank;
        int size;
        MPI_Comm comm;
        comm = MPI_COMM_WORLD;

        MPI_Init(NULL, NULL);

        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        std::cout << "Hello from rank " << rank << std::endl;
        MPI_Finalize();

        return 0;
    }
    catch (std::exception &exc)
    {
        std::cout << exc.what() << std::endl;
        MPI_Finalize();
        return -1;
    }
    catch (...)
    {
        std::cout << "Got exception which wasn't caught" << std::endl;
        MPI_Finalize();
        return -1;
    }
}
