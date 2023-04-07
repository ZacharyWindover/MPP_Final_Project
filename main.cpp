#include "utilities_package.h"

//#include "aabb.h"
//#include "aarect.h"
#include "aabb_minmax_box.h"
#include "colour.h"
#include "collision.h"
#include "collision_list.h"
#include "eye.h"
#include "material.h"
#include "plane_rects.h"
#include "sphere.h"
#include "texture.h"
#include "vec3.h"

#include <iostream>
#include <cstdlib>
#include <thread>
#include <vector>
#include <functional>
#include <chrono>
#include <omp.h>
#include <time.h>
#include <cmath>

using namespace std::chrono;

int image_width;
int samples_per_pixel;
int max_depth;
int image_height;

int global_x, global_k, global_z;
int light_x0, light_x1, light_y0, light_y1, light_k;

collision_list world;
colour background;
//colour pixel_colour(0, 0, 0);
eye camera;

//std::vector<int> rgb_file_values;
//std::vector<std::vector<int>> rgb_values (1, std::vector<int>(3, 0));


colour ray_colour(const ray& r, const colour& background, const collision& world, int depth) {
    
    collision_record record;

    // If reached max depth, won't branch out and get more ray light
    if (depth <= 0)
        return colour(0, 0, 0);

    // If ray hits nothing, return background colour;
    if (!world.hit(r, 0.001, infinity, record)) 
        return background;

    ray scattered;
    colour attenuation;
    colour emitted = record.mat_ptr->emitted(record.u, record.v, record.p);

    if (!record.mat_ptr->scatter(r, record, attenuation, scattered))
        return emitted;

    return emitted + attenuation * ray_colour(scattered, background, world, depth - 1);

}



collision_list preset_scene() {

    collision_list objects;

    auto red = make_shared<lambertian>(colour(.65, .05, .05));
    auto white = make_shared<lambertian>(colour(.73, .73, .73));
    auto green = make_shared<lambertian>(colour(.12, .45, .15));
    auto blue = make_shared<lambertian>(colour(0.196, 0.196, 0.658));
    auto light = make_shared<diffuse_light>(colour(25, 25, 25));
    auto metal_material = make_shared<metal>(colour(0.7, 0.6, 0.5), 0.0);

    light_x0 = -1;
    light_x1 =  1;
    light_y0 = -1;
    light_y1 =  1;
    global_k = light_k = 5;

    // make the light source
    objects.add(make_shared<xz_rect>(light_x0, light_x1, light_y0, light_y1, light_k, light));

    // make the ground plane
    objects.add(make_shared<xz_rect>(-555, 555, -555, 555, 0, green));

    // add three spheres
    objects.add(make_shared<sphere>(coordinate(0, 1, -1), 1.0, metal_material));

    objects.add(make_shared<sphere>(coordinate(2, 0.75, -2), 0.75, red));
    objects.add(make_shared<sphere>(coordinate(3, 0.75, 2), 0.75, blue));

    return objects;

}

/*
// base ray trace function
void base_ray_trace(int i, int j) {

    auto u = (i + random_double()) / (image_width - 1);
    auto v = (j + random_double()) / (image_height - 1);
    ray r = camera.get_ray(u, v);
    pixel_colour += ray_colour(r, background, world, max_depth); 

}
*/


int main(int argc, char** argv) {

    // recording starting point
    auto start = high_resolution_clock::now();

    // obtain values from cmd line inputs
    image_width = atoi(argv[1]);
    samples_per_pixel = atoi(argv[2]);
    max_depth = atoi(argv[3]);
    int trace_type = atoi(argv[4]);

    // Image
    auto aspect_ratio = 16.0 / 9.0;
    image_height = static_cast<int>(image_width / aspect_ratio);

    // presets (testing)
    //image_width = 1280;
    //samples_per_pixel = 16384;
    //max_depth = 16;

    // World
    world = preset_scene();
    coordinate look_from = coordinate(13, 2, 3);
    coordinate look_at = coordinate(0, 0, 0);
    auto v_fov = 40.0;

    // choose either blue or black
    //background = colour(0, 0, 0);
    background = colour(0.0823, 0.6745, 0.9294);
    
    // Eye
    const vec3 v_up(0, 1, 0);
    const auto distance_to_focus = 10.0;
    auto aperture = 0.0;

    eye camera(look_from, look_at, v_up, v_fov, aspect_ratio, aperture, distance_to_focus, 0.0, 1.0);

    // Render type
    enum render_type {
        base_ray_trace,
        original_omp_ray_trace,
        new_omp_ray_trace,
        new_opencl_ray_trace,
        new_omp_opencl_ray_trace
    };

    render_type render = new_omp_ray_trace;

    // change the render type based on the input
    if (trace_type == 1) render = base_ray_trace;
    else if (trace_type == 2) render = original_omp_ray_trace;
    else if (trace_type == 3) render = new_omp_ray_trace;
    else if (trace_type == 4) render = new_opencl_ray_trace;
    else if (trace_type == 5) render = new_omp_opencl_ray_trace;
    else render = new_omp_ray_trace;
   
    if (render == base_ray_trace) { // if base_ray_trace, performs ray trace without any parallelization

        // ppm file header
        std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

        for (int j = image_height - 1; j >= 0; --j) {

            std::cerr << "\rLines remaining: " << j << ' ' << std::flush;

            for (int i = 0; i < image_width; ++i) {

                colour pixel_colour(0, 0, 0);

                for (int s = 0; s < samples_per_pixel; ++s) {

                    auto u = (i + random_double()) / (image_width - 1);
                    auto v = (j + random_double()) / (image_height - 1);
                    ray r = camera.get_ray(u, v);
                    pixel_colour += ray_colour(r, background, world, max_depth);

                }

                write_colour(std::cout, pixel_colour, samples_per_pixel);

            }

        }

    }

    else if (render == original_omp_ray_trace) {    // if using original_omp_ray_trace, ray trace computed with original parallelization

        // ppm file header
        std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

        for (int j = image_height - 1; j >= 0; --j) {

            std::cerr << "\rLines remaining: " << j << ' ' << std::flush;

            for (int i = 0; i < image_width; ++i) {

                colour pixel_colour(0, 0, 0);

                #pragma omp parallel for schedule(dynamic, 4)
                for (int s = 0; s < samples_per_pixel; ++s) {

                    auto u = (i + random_double()) / (image_width - 1);
                    auto v = (j + random_double()) / (image_height - 1);
                    ray r = camera.get_ray(u, v);
                    pixel_colour += ray_colour(r, background, world, max_depth);

                }

                write_colour(std::cout, pixel_colour, samples_per_pixel);

            }

        }

    }

    else if (render == new_omp_ray_trace) {
        
        // ppm file header
        std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

        for (int j = image_height - 1; j >= 0; --j) {

            std::cerr << "\rLines remaining: " << j << ' ' << std::flush;

            for (int i = 0; i < image_width; ++i) {

                colour pixel_colour(0, 0, 0);

                #pragma omp parallel for ordered schedule(guided) // 657 seconds
                //#pragma omp parallel for ordered schedule(dynamic, 12) // 678 seconds
                //#pragma omp parallel for schedule(dynamic, 12) // 668s for 1024 9586s for 16384
                //#pragma omp parallel for schedule(guided) //651s for 1024, 9767s for 16384
                for (int s = 0; s < samples_per_pixel; ++s) {

                    auto u = (i + random_double()) / (image_width - 1);
                    auto v = (j + random_double()) / (image_height - 1);
                    ray r = camera.get_ray(u, v);
                    pixel_colour += ray_colour(r, background, world, max_depth);

                }

                write_colour(std::cout, pixel_colour, samples_per_pixel);

            }

        }
        

        /*
        // ppm file header
        std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

        omp_set_num_threads(12);

        for (int j = image_height - 1; j >= 0; --j) {

            std::cerr << "\rLines remaining: " << j << ' ' << std::flush;

            for (int i = 0; i < image_width; ++i) {

                colour pixel_colour(0, 0, 0);
                
                #pragma omp parallel for ordered schedule(dynamic, 8)
                for (int s = 0; s < samples_per_pixel; ++s) {

                    auto u = (i + random_double()) / (image_width - 1);
                    auto v = (j + random_double()) / (image_height - 1);
                    ray r = camera.get_ray(u, v);
                    pixel_colour += ray_colour(r, background, world, max_depth);

                }

                write_colour(std::cout, pixel_colour, samples_per_pixel);

            }

        }
        */

        /*
        // ppm file header
        std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

        omp_set_num_threads(12);

        //#pragma omp parallel for ordered
        for (int j = image_height - 1; j >= 0; --j) {

            std::cerr << "\rLines remaining: " << j << ' ' << std::flush;

            for (int i = 0; i < image_width; ++i) {

                colour pixel_colour(0, 0, 0);

                #pragma omp parallel for ordered 
                for (int s = 0; s < samples_per_pixel; ++s) {

                    auto u = (i + random_double()) / (image_width - 1);
                    auto v = (j + random_double()) / (image_height - 1);
                    ray r = camera.get_ray(u, v);
                    pixel_colour += ray_colour(r, background, world, max_depth);

                }

                write_colour(std::cout, pixel_colour, samples_per_pixel);

            }

        }
        */

    }

    else if (render == new_opencl_ray_trace) {

        // ppm file header
        std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

        for (int j = image_height - 1; j >= 0; --j) {

            std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;

            for (int i = 0; i < image_width; ++i) {

                colour pixel_colour(0, 0, 0);

                for (int s = 0; s < samples_per_pixel; ++s) {

                    auto u = (i + random_double()) / (image_width - 1);
                    auto v = (j + random_double()) / (image_height - 1);
                    ray r = camera.get_ray(u, v);
                    pixel_colour += ray_colour(r, background, world, max_depth);

                }

                write_colour(std::cout, pixel_colour, samples_per_pixel);

            }

        }

    }

    else if (render == new_omp_opencl_ray_trace) {

        // ppm file header
        std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

        for (int j = image_height - 1; j >= 0; --j) {

            std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;

            for (int i = 0; i < image_width; ++i) {

                colour pixel_colour(0, 0, 0);

                for (int s = 0; s < samples_per_pixel; ++s) {

                    auto u = (i + random_double()) / (image_width - 1);
                    auto v = (j + random_double()) / (image_height - 1);
                    ray r = camera.get_ray(u, v);
                    pixel_colour += ray_colour(r, background, world, max_depth);

                }

                write_colour(std::cout, pixel_colour, samples_per_pixel);

            }

        }

    } 

    else { // Automatically does new omp ray trace



    }

    std::cerr << "\nDone.\n";

    // get the end time of the program
    auto stop = high_resolution_clock::now();

    // calculate time taken
    auto runtime = duration_cast<microseconds>(stop - start);

    // output time
    std::cerr << runtime.count() << " microseconds " << std::endl;

}