__kernel void degrees_to_radians(__global double degrees) {
    double radians;
    radians = degrees * 3.1415926535897932 / 180;
}