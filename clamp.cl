__kernel void clamp(__global double* x,  __global double* output, __global double* min, __global double* max) 
{

    int gid = get_global_id(0);

    if (x[gid] < min[gid]) {
        output[gid] = min[gid];
    } else if (x[gid] > max[gid]) {
        output[gid] = max[gid];
    } else {
        output[gid] = x[gid];
    }

}