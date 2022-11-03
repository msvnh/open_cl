#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

#include <iostream>
#include "OpenCLHelper.h"
#include <array>

int main()
{
	cl::Program program = CreateProgram("ProcessMultidimensionalArray.cl");
	auto context = program.getInfo<CL_PROGRAM_CONTEXT>();
	auto devices = program.getInfo<CL_PROGRAM_DEVICES>();

	auto device = devices.front();

	cl_int err = 0;

	const int numRows = 5;
	const int numCols = 3;
	const int count = numRows * numCols;
	std::array<std::array<int, numCols>, numRows> arr = { {{80, 80, 80}, {22, 22, 22}, {31, 31, 31}, {4, 4, 4}, {67, 67, 67}} };

	cl::Buffer buf(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * count, arr.data());
	cl::Kernel kernel(program, "ProcessMultidimensionalArray");
	kernel.setArg(0, buf);

	cl::CommandQueue queue(context, device);
	err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(numCols, numRows));
	err = queue.enqueueReadBuffer(buf, CL_TRUE, 0, sizeof(int) * count, arr.data());

	std::cin.get();
}