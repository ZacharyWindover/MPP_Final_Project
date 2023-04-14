#ifndef utilities_package_h
#define utilities_package_h

#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>

#include <CL/cl.h>

#include "clamp_host.h"
#include "degrees_to_radians_host.h"

// Using
using std::shared_ptr;
using std::make_shared;
using std::sqrt;

// Constants
const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

// Utility Functions

// put this function into an opencl kernel
inline double degrees_to_radians(double degrees) {
    //return degrees * m_pi / 180.0;
    double radians = degrees_to_radians_host(degrees);
    return radians;
}

// put this function into an opencl kernel
inline double clamp(double x, double min, double max) {
    //if (x < min) return min;
    //if (x > max) return max;
    //return x;
    double clamp_value = clamp_host(x, min, max);
    return clamp_value;
}

inline double random_double() {
    // Returns a random real in [0,1).
    return rand() / (RAND_MAX + 1.0);
}

inline double random_double(double min, double max) {
    // Returns a random real in [min,max).
    return min + (max - min) * random_double();
}

inline int random_int(int min, int max) {
    // Returns a random integer in [min,max].
    return static_cast<int>(random_double(min, max + 1));
}

inline bool random_bool() {
    return rand() > (RAND_MAX / 2);
}



#include "ray.h"
#include "vec3.h"


#endif





