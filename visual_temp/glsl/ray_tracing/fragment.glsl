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


vec3 inter_ray_color(Ray r, Sphere[4] t_list, float upper_seed) {
    int spheres_len = 4;
    vec3 attenuation = vec3(0, 0, 0);

    int first_sphere_i = -1;

    int depth = 0;
    int max_depth = 40;
    float t_d = 0;
    bool is_hit = false;

    for (int i = 0; i < spheres_len; i++) {
        if (depth > max_depth || is_hit) {
            break;
        }

        Sphere t_sphere = t_list[i];


        t_d = calculateSphere(r.orig, r.dir, t_sphere.center, t_sphere.radius);


        while (t_d > 0.0) {
            depth += 1;

            if (depth > max_depth) {
                break;
            }

            if (first_sphere_i == -1) {
                first_sphere_i = i;
            }

            is_hit = true;

            vec3 p = rayAt(r, t_d);


            vec3 N = (p - t_sphere.center) / t_sphere.radius;


            vec2 seed = vec2(upper_seed, upper_seed + 1.);
            vec3 direction;

            if (t_sphere.is_reflect) {
                direction = reflect(unit_vector(r.dir), N);
            }
            else {
                vec3 rand_c = random_on_unit_sphere(seed);

                if (rand_c.z < 0) {
                    rand_c.z *= -1;
                }

                direction = N + rand_c;
                attenuation = t_sphere.color;

            }



            t_d = 0;

            for (int j = 0; j < spheres_len; j++) {
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
        final_color = (t_list[first_sphere_i].color * (1-t_list[first_sphere_i].albedo)  +  attenuation * (t_list[first_sphere_i].albedo)) / depth;
    }
    else {
        // background
        vec3 unit_direction = unit_vector(r.dir);

        float blend_a = 0.9 * (unit_direction.y + 1.0);
        vec3 white = vec3(1.0, 1.0, 1.0);
        vec3 blue = vec3(1, 0.2, 0.7);
        final_color = (1.0 - blend_a) * white + blend_a * blue;

        // ?
        if (first_sphere_i != -1) {
            Sphere sphere = t_list[first_sphere_i];
            if (sphere.is_reflect) {
                final_color = normalize(final_color + (sphere.albedo * sphere.color));
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

    return final_color;

}

mat3 rotationMatrix(float angle, vec3 axis) {
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;

    vec3 a = normalize(axis);

    mat3 result = mat3(
        oc * a.x * a.x + c, oc * a.x * a.y - a.z * s, oc * a.x * a.z + a.y * s,
        oc * a.x * a.y + a.z * s, oc * a.y * a.y + c, oc * a.y * a.z - a.x * s,
        oc * a.x * a.z - a.y * s, oc * a.y * a.z + a.x * s, oc * a.z * a.z + c
    );

    return result;
}


void main() {
    ivec2 pixelCoords = ivec2(gl_FragCoord.xy);
    float aspect_ratio = float(width) / float(height);

    float x = float(pixelCoords.x) / float(width);
    float y = (float(pixelCoords.y) / float(height) - (1.0 - aspect_ratio) / 2.0) / aspect_ratio;

    float small_radius = 0.1;
    Sphere sphere1 = Sphere(vec3(0.52, 0.52, -0.8), vec3(0.5, 0.5, 0.2), vec3(0.9, 0.9, 0.9), small_radius, true);
    Sphere sphere2 = Sphere(vec3(0.7, 0.58, -0.8), vec3(0.9, 0.2, 0.3), vec3(0.2, 0.2, 0.2), 0.05, true);

    Sphere sphere3 = Sphere(vec3(0.3, 0.5, -0.8), vec3(0.5, 0.2, 0.7), vec3(0.2, 0.2, 0.2), small_radius, false);

    float surface_radius = 100;
    Sphere sphere4 = Sphere(vec3(0.5, 0.5 - surface_radius - small_radius, -1), vec3(0.2, 0.7, 0.3), vec3(0.2, 0.2, 0.2), surface_radius, false);


    Sphere[4]t_list = { sphere3, sphere1, sphere2,  sphere4 }; //, 



    // camera rot 
    float rotation_angle = -5;
    float angle = radians(rotation_angle); // Convert the rotation angle to radians
    vec3 axis = vec3(0.0, 0.0, 1.0); // Rotation axis (e.g., around the Z-axis)

    mat3 rotation = rotationMatrix(angle, axis);


    vec3 temp_dir = vec3(x, y, -1) - camera_center;
    temp_dir = rotation * temp_dir;


    int samples_per_pixel = 100;
    float sample_side = 0.001;

    vec4 sampled_color;

    for (int i = 0; i < samples_per_pixel; i++) {
        float rand_dir = sample_side * (2 * rand(gl_FragCoord.xy + i) - 1);


        vec3 sampled_dir = temp_dir + rand_dir;

        Ray r = createRay(camera_center, sampled_dir);

        sampled_color += vec4(inter_ray_color(r, t_list, float(i) + rand_dir), 0);
    }


    float gamma = 1.5;
    FragColor = sampled_color / samples_per_pixel * gamma;


}