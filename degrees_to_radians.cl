#define M_PI 3.1415926535897932

__kernel void degrees_to_radians(__global double* degrees, __global double* radians) {

    int gid = get_global_id(0);
    radians[gid] = degrees[gid] * M_PI / 180;

}