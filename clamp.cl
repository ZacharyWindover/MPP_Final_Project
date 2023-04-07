__kernel void clamp(__global double x, __global double min, __global double max) {
    double return_value = get_global_id(0);
    if (x < min) return_value = min;
    if (x > max) return_value = max;
    return_value = x;
}