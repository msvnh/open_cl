#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

#include <iostream>
#include "OpenCLHelper.h"
#include <array>

int main()
{
	cl::Program program = CreateProgram("RotateMatrix.cl");
	auto context = program.getInfo<CL_PROGRAM_CONTEXT>();
	auto devices = program.getInfo<CL_PROGRAM_DEVICES>();

	auto device = devices.front();

	cl_int err = 0;

	const int numRows = 3;
	const int numCols = 3;
	const int count = numRows * numCols;
	const int max = max(numCols, numRows);
	std::array<std::array<int, numCols>, numRows> arr = { {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}} };

	cl::Buffer inBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(int) * count, arr.data(), &err);
	cl::Buffer outBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(int) * arr.size());

	cl::Kernel kernel(program, "RotateMatrix", &err);
	err = kernel.setArg(0, inBuf);
	err = kernel.setArg(1, outBuf);
	err = kernel.setArg(2, (size_t)count);

	cl::CommandQueue queue(context, device);
	err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(max, max));
	err = queue.enqueueReadBuffer(outBuf, CL_FALSE, 0, sizeof(int) * count, arr.data());
	cl::finish();

	std::cin.get();
}