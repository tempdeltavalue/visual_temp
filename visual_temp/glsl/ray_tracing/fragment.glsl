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


layout(std430, binding = 3) buffer TrianglesBuffer {
    Triangle triangles[];
};


/// Ray struct

struct Ray {
    vec3 orig;
    vec3 dir;
};


struct Sphere {
    vec3 center;
    vec3 color;
    vec3 albedo;
    float radius;
    bool is_reflect;
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


vec3 inter_ray_color(Ray r, Sphere[4] t_list, int upper_seed) {
    vec3 attenuation = vec3(0, 0, 0);

    int first_sphere_i = -1;

    int depth = 0;
    int max_depth = 10;
    float t_d = 0;


    for (int i = 0; i < 4; i++) {
        if (depth > max_depth) {
            break;
        }

        Sphere t_sphere = t_list[i];


        t_d = calculateSphere(r.orig, r.dir, t_sphere.center, t_sphere.radius);


        while (t_d > 0.0) {
            if (depth > max_depth) {
                break;
            }

            if (first_sphere_i == -1) {
                first_sphere_i = i;
            }

            depth += 1;

            vec3 p = rayAt(r, t_d);


            vec3 N = (p - t_sphere.center) / t_sphere.radius;

            vec3 direction;
            vec2 seed = gl_FragCoord.xy + p.y + N.xy + float(upper_seed + depth + i);
            //seed = (normalize(seed) * 2.0 - 1.0) * 0.5; // This transforms fragPos to range from -0.5 to 0.5.

            if (t_sphere.is_reflect) {
                direction = reflect(unit_vector(r.dir), N);
            }
            else {

                direction = N + random_on_unit_sphere(seed);

            }


            if (dot(random_on_unit_sphere(seed), N) > 0.0) // In the same hemisphere as the normal
                attenuation = vec3(1, 0, 0);
            else
                attenuation = vec3(1, 1, 0);

            if (i == 1) {
                attenuation = N + random_on_unit_sphere(seed);
            }
            //attenuation = t_sphere.color;
   

            t_d = 0;

            for (int j = 0; j < 4; j++) {

                t_sphere = t_list[j];
                t_d = calculateSphere(p, direction, t_sphere.center, t_sphere.radius);
                if (t_d > 0) {
                    r = createRay(p, direction);
                    break;
                }
            }
        }

    }

    vec3 final_color = vec3(0, 0, 0);
    if (attenuation.x != 0) {
        //final_color = ( t_list[first_sphere_i].color * (1-t_list[first_sphere_i].albedo)  +  attenuation * (t_list[first_sphere_i].albedo)) / depth;
        final_color = attenuation / depth;

    }

    //if (final_color.x != 0) {
    //    final_color /= max_depth;
    //}

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
        final_color = (1.0 - blend_a) * white + blend_a * blue;
    }

    return final_color;

}


void main() {
    ivec2 pixelCoords = ivec2(gl_FragCoord.xy);
    float aspect_ratio = float(width) / float(height);

    float x = float(pixelCoords.x) / float(width);
    float y = (float(pixelCoords.y) / float(height) - (1.0 - aspect_ratio) / 2.0) / aspect_ratio;

    float small_radius = 0.1;
    Sphere sphere1 = Sphere(vec3(0.5, 0.52, -0.8), vec3(0.5, 0.5, 0.2), vec3(0.9, 0.9, 0.9), small_radius, true);
    Sphere sphere2 = Sphere(vec3(0.7, 0.58, -0.8), vec3(0.9, 0.2, 0.3), vec3(0.2, 0.2, 0.2), 0.05, true);

    Sphere sphere3 = Sphere(vec3(0.28, 0.47, -0.8), vec3(0.5, 0.2, 0.7), vec3(0.2, 0.2, 0.2), small_radius, false);

    float surface_radius = 10000;
    Sphere sphere4 = Sphere(vec3(0.5, 0.5 - surface_radius - small_radius - 0.03, -1), vec3(0.2, 0.7, 0.3), vec3(0.2, 0.2, 0.2), surface_radius, false);


    Sphere[4]t_list = { sphere1, sphere2, sphere3, sphere4 }; //, 



    vec3 temp_dir = vec3(x, y, -1) - camera_center;


    int samples_per_pixel = 300;
    float sample_side = 0.001;

    vec4 sampled_color;

    for (int i = 0; i < samples_per_pixel; i++) {
        float rand_dir = sample_side * (2 * rand(gl_FragCoord.xy + i) - 1);


        vec3 sampled_dir = temp_dir + rand_dir;

        Ray r = createRay(camera_center, sampled_dir);

        sampled_color += vec4(inter_ray_color(r, t_list, i), 0);
    }


    float gamma = 1.5;
    FragColor = sampled_color / samples_per_pixel * gamma;


}