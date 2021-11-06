#include <iostream>
#define N 5

class data_class
{
public:
	double data[N]{};
	data_class(){};
	inline __host__ __device__ double& operator()(int i)
	{
		return data[i];
	}
};

__global__
void read_array(data_class *input, double *output)
{
	int thread_idx = threadIdx.x;
	if (thread_idx < N)
		output[thread_idx] = input[0](thread_idx) / 2.0;
}

int main()
{
	data_class *input = new data_class;
	for (int i = 0; i < N; ++i)
	{
		input[0](i) = i;
		std::cout << input[0](i) << std::endl;
	}
	std::cout << std::endl;
	
	data_class *d_input;
	cudaMalloc(&d_input, sizeof(data_class));
	cudaMemcpy(d_input, input, sizeof(data_class), cudaMemcpyHostToDevice);
	
	double *d_output;
	cudaMalloc(&d_output, N*sizeof(double));
	read_array<<<1, N>>> (d_input, d_output);
	
	double *output = new double[N];
	cudaMemcpy(output, d_output, N*sizeof(double), cudaMemcpyDeviceToHost);
	
	for (int i = 0; i < N; ++i)
		std::cout << output[i] << std::endl;
		
	cudaFree(d_input);
	cudaFree(d_input);
	delete input;
	delete output;
	
	return 0;
}