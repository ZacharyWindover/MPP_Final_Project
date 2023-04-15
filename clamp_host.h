#include <iostream>
#include <vector>
#include <CL/cl.h>

double clamp_host(double x, double min, double max) {

    // Initialize OpenCL platform and device
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem input_buffer_x, output_buffer, input_buffer_min, input_buffer_max;
    cl_int err;

    // Get first available platform and device
    err = clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    // Create context and command queue
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);

    // Convert data to vectors
    std::vector<double> x_vector = { x };
    std::vector<double> output_vector = { 0.0 };
    std::vector<double> min_vector = { min };
    std::vector<double> max_vector = { max };

    // Create input and output buffers
    input_buffer_x = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * x_vector.size(), x_vector.data(), &err);
    input_buffer_min = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * min_vector.size(), min_vector.data(), &err);
    input_buffer_max = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * max_vector.size(), max_vector.data(), &err);
    output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double) * output_vector.size(), NULL, &err);

    // Create kernel program
    const char* source = "__kernel void clamp(__global double* x,  __global double* output, __global double* min, __global double* max) {\n"
        "    int gid = get_global_id(0);\n"
        "    if (x[gid] < min[gid]) {\n"
        "        output[gid] = min[gid];\n"
        "    } else if (x[gid] > max[gid]) {\n"
        "        output[gid] = max[gid];\n"
        "    } else {\n"
        "        output[gid] = x[gid];\n"
        "    }\n"
        "}\n";
    size_t source_size = strlen(source);
    program = clCreateProgramWithSource(context, 1, &source, &source_size, &err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    // Create kernel
    kernel = clCreateKernel(program, "clamp", &err);

    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer_x);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &input_buffer_min);
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &input_buffer_max);

    // Execute kernel
    size_t global_size = 1;
    size_t local_size = 1;

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);

    // Read the result from the output buffer
    double output;
    err = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, sizeof(double), &output, 0, NULL, NULL);

    // Release resources
    clReleaseMemObject(input_buffer_x);
    clReleaseMemObject(output_buffer);
    clReleaseMemObject(input_buffer_min);
    clReleaseMemObject(input_buffer_max);

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return output;

}
