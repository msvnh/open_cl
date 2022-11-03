#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

#include <iostream>
#include "OpenCLHelper.h"

int main()
{
	cl::Program program = CreateProgram("ProcessArray.cl");
	auto context = program.getInfo<CL_PROGRAM_CONTEXT>();
	auto devices = program.getInfo<CL_PROGRAM_DEVICES>();

	auto device = devices.front();

	std::vector<int> vec(1024);

	cl_int err = 0;

	cl::Buffer inBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(int) * vec.size(), vec.data(), &err);
	cl::Buffer outBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(int) * vec.size());

	cl::Kernel kernel(program, "ProcessArray", &err);
	err = kernel.setArg(0, inBuf);
	err = kernel.setArg(1, outBuf);

	cl::CommandQueue queue(context, device);

	err = queue.enqueueFillBuffer(inBuf, 3, sizeof(int) * 10, sizeof(int) * vec.size() - 10);

	err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(vec.size()));
	err = queue.enqueueReadBuffer(outBuf, CL_FALSE, 0, sizeof(int) * vec.size(), vec.data());

	cl::finish();

	std::cin.get();
}