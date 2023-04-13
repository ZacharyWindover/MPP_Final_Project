#define M_PI 3.1415926535897932

__kernel void degrees_to_radians(__global double degrees) {
    double radians;
    radians = degrees * M_PI / 180;
}
