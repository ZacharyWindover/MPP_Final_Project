#include <iostream>
#include <vector>
#include <CL/cl.h>

double degrees_to_radians_host(double degrees) {

    // Initialize OpenCL platform and device
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem input_buffer, output_buffer;
    cl_int err;

    // Get the first available platform and device
    err = clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    // Create a context and command queue
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);

    // convert data to vectors
    std::vector<double> degrees_vector = { degrees };
    std::vector<double> radians_vector = { 0.0 };

    // Create input and output buffers
    input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * degrees_vector.size(), degrees_vector.data(), &err);
    output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double) * radians_vector.size(), NULL, &err);

    // Create the kernel program
    const char* source = "#define M_PI 3.14159265358979323846\n"
        "__kernel void degrees_to_radians(__global double* degrees, __global double* radians) {\n"
        "    int gid = get_global_id(0);\n"
        "    radians[gid] = degrees[gid] * M_PI / 180.0;\n"
        "}\n";
    size_t source_size = strlen(source);
    program = clCreateProgramWithSource(context, 1, &source, &source_size, &err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    // Create the kernel
    kernel = clCreateKernel(program, "degrees_to_radians", &err);

    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer);

    // Execute the kernel
    size_t global_size = 1;
    size_t local_size = 1;

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);

    // Read the result from the output buffer
    double output;
    err = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, sizeof(double), &output, 0, NULL, NULL);

    // Release resources
    clReleaseMemObject(input_buffer);
    clReleaseMemObject(output_buffer);

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return output;

}

