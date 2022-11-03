
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

#include <iostream>
#include <fstream>
#include "OpenCLHelper.h"


int main()
{
	auto program = CreateProgram("HelloWorld.cl");
	auto context = program.getInfo<CL_PROGRAM_CONTEXT>();
	auto devices = program.getInfo<CL_PROGRAM_DEVICES>();

	auto device = devices.front();

	char buf[16];
	cl::Buffer memBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(buf));
	cl::Kernel kernel(program, "ByeWorld");
	kernel.setArg(0, memBuf);

	cl::CommandQueue queue(context, device);
	queue.enqueueTask(kernel);
	queue.enqueueReadBuffer(memBuf, CL_TRUE, 0, sizeof(buf), buf);

	std::cout << buf;
	std::cin.get();
}