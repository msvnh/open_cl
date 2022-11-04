#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

#include <iostream>
#include "OpenCLHelper.h"
#include <array>

int main()
{
	cl::Program program = CreateProgram("NumericalReduction.cl");
	auto context = program.getInfo<CL_PROGRAM_CONTEXT>();
	auto devices = program.getInfo<CL_PROGRAM_DEVICES>();

	auto device = devices.front();

	cl_int err = 0;

	std::vector<int> vec(1024);

	for (int i = 0; i < vec.size(); i++)
		vec[i] = i;

	cl::Kernel kernel(program, "NumericalReduction");

	auto workGroupSize = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device, &err);

	auto numWorkGroups = vec.size() / workGroupSize;

	cl::Buffer inBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(int) * vec.size(), vec.data());
	cl::Buffer outBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, numWorkGroups);

	kernel.setArg(0, inBuf);
	kernel.setArg(1, sizeof(int) * workGroupSize, nullptr);
	kernel.setArg(2, outBuf);

	std::vector<int> outVec(numWorkGroups);

	cl::CommandQueue queue(context, device);
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(vec.size()), cl::NDRange(workGroupSize));
	queue.enqueueReadBuffer(outBuf, CL_TRUE, 0, sizeof(int) * outVec.size(), outVec.data());

	std::cin.get();
}