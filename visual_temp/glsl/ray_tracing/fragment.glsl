#version 430 core

#include figure_trace.glsl
#include vec3_utils.glsl

out vec4 FragColor;


uniform int width;
uniform int height;
uniform int triangles_len;

uniform vec3 camera_center;

uniform vec2 mousePos;

uniform float current_time;

uniform sampler2D texture_diffuse1;

struct Triangle {
    vec4 v1;
    vec4 v2;
    vec4 v3;
    vec4 texCoords;

};


layout(std430, binding=3) buffer TrianglesBuffer {
    Triangle triangles[];
};


/// Ray struct

struct Ray {
    vec3 orig;
    vec3 dir;
};


struct Sphere {
    vec3 center;
    float radius;
};

Ray createRay(vec3 origin, vec3 direction) {
    Ray ray;
    ray.orig = origin;
    ray.dir = direction;
    return ray;
}

vec3 rayAt(Ray ray, float t) {
    return ray.orig + t * ray.dir;
}


// GLSL doesn't support recursion 


vec3 inter_ray_color(Ray r, Sphere[2] t_list, float upper_seed) {
    vec3 final_color = vec3(0, 0, 0); // null

    for (int i = 0; i < 2; i++) {
        Sphere t_sphere = t_list[i];
        float t_d = calculateSphere(r.orig, r.dir, t_sphere.center, t_sphere.radius);

        int depth = 0;

        while (t_d > 0.0) {

            if (depth > 3) {
                break;
            }

            depth += 1;

            vec3 p = rayAt(r, t_d);


            vec3 N = (p - t_sphere.center) / t_sphere.radius;

            vec3 current_r_color = 0.5 * (N + vec3(1, 1, 1));

            final_color = (0.5 / depth) * current_r_color;

      
            final_color /= depth; // <-??

            vec2 seed = gl_FragCoord.xy + gl_FragCoord.yx + 1 + upper_seed + depth;

            vec3 direction = random_on_hemisphere(N, seed); // N + random_unit_vector(seed); //
            //final_color = direction;


            //t_d = 0;
            for (int j = 0; j < 2; j++) {
                if (i == j) {
                    continue;
                }

                t_sphere = t_list[j];
                t_d = calculateSphere(p, direction, t_sphere.center, t_sphere.radius);
                r = createRay(p, direction);
            }
        }
    }


    /// figure trace //
    //if (final_color.x == -1.0) {
    //    for (int i = 0; i < triangles_len; i++) {
    //        if (i % 2 == 0) {
    //            continue;
    //        }
    //        Triangle tri = triangles[i];
    //        if (rayTriangleIntersect(r.orig, r.dir, tri.v1.xyz, tri.v2.xyz, tri.v3.xyz)) {
    //            final_color = texture(texture_diffuse1, tri.texCoords.xy).xyz; //vec4(1, 0, 0, 0);
    //            break;
    //        }
    //    }
    //}
    ///

    if (final_color.x == 0) {

        vec3 unit_direction = unit_vector(r.dir);

        float blend_a = 0.9 * (unit_direction.y + 1.0);
        vec3 white = vec3(1.0, 1.0, 1.0);
        vec3 blue = vec3(0.2, 0.2, 0.7);
        final_color= (1.0-blend_a)*white + blend_a*blue;
    }

    return final_color;

}


void main() {
    ivec2 pixelCoords = ivec2(gl_FragCoord.xy);
    float aspect_ratio = float(width) / float(height);

    float x = float(pixelCoords.x) / float(width);
    float y = (float(pixelCoords.y) / float(height) - (1.0 - aspect_ratio) / 2.0) / aspect_ratio;


    vec3 sphere_coords = vec3(0.5, 0.5, -0.8);
    float radius = 0.1;

    float radius2 =100;
    vec3 sphere_coords2 = vec3(0.5, 0.5 - radius2 - radius - 0.03, -1);

    Sphere [2]t_list = {Sphere(sphere_coords, radius), Sphere(sphere_coords2, radius2)}; //, 



    vec3 temp_dir = vec3(x, y, -1) - camera_center;


    int samples_per_pixel = 200;
    float sample_side = 0.001;

    vec4 sampled_color;

    for (int i = 0; i < samples_per_pixel; i++) {
        float rand_dir = sample_side * (2 * rand(gl_FragCoord.xy + i) - 1);

        
        vec3 sampled_dir = temp_dir + rand_dir;

        Ray r = createRay(camera_center, sampled_dir);

        sampled_color += vec4(inter_ray_color(r, t_list, i), 0);
    }


    FragColor = sampled_color/ samples_per_pixel;  
}