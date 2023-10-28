#version 430 core

#include vec3_utils.glsl

out vec4 FragColor;


uniform int width;
uniform int height;
uniform int triangles_len;
uniform int spheres_count;

uniform vec3 camera_center;

uniform vec2 mousePos;

uniform float current_time;

uniform sampler2D texture_diffuse1;




//struct Sphere {
//    vec3 center;
//    vec3 color;
//    vec3 albedo;
//    float radius;
//    bool is_reflect;
//    float fuzz; // only for reflect mat
//    bool is_dielectric;
//};

struct Triangle {
    vec4 v1;
    vec4 v2;
    vec4 v3;
    vec4 texCoords;

};

//layout(std430, binding = 3) buffer TrianglesBuffer {
//    Triangle triangles[];
//};

struct Sphere {
    vec4 center;
    vec4 color;
    vec4 albedo; 
    vec4 radius; //float
    vec4 is_reflect; // bool
    vec4 fuzz; // float only for reflect mat
    vec4 is_dielectric; // bool

    //moving
    vec4 is_moving; // bool
    vec4 center2;
};


vec4 getCenterAtTime(Sphere sphere, float time) {
    // Linearly interpolate from center1 to center2 according to time, where t=0 yields
    // center1, and t=1 yields center2.
    return sphere.center + time * (sphere.center2 - sphere.center);
}



layout(std430, binding = 2) buffer SphereBuffer {
    Sphere spheres[];
};



/// Ray struct

struct Ray {
    vec3 orig;
    vec3 dir;
    float time;
};



Ray createRay(vec3 origin, vec3 direction, float time) {
    Ray ray;
    ray.orig = origin;
    ray.dir = direction;
    ray.time = time;
    return ray;
}

vec3 rayAt(Ray ray, float t) {
    return ray.orig + t * ray.dir;
}


#include figure_trace.glsl

// GLSL doesn't support recursion 

vec3 inter_ray_color(Ray r, float upper_seed) {

    vec3 attenuation = vec3(0, 0, 0);

    int first_sphere_i = -1;

    int depth = 0;
    int max_depth = 10;
    float t_d = 0;
    bool is_hit = false;

    for (int i = 0; i < spheres_count; i++) {
        if (depth > max_depth || is_hit) {
            break;
        }

        Sphere t_sphere = spheres[i];

        t_d = calculateSphere(r, t_sphere);


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


            vec3 N = (p - t_sphere.center.xyz) / t_sphere.radius.x;


            vec2 seed = vec2(upper_seed, upper_seed + 1.);
            vec3 direction;

            vec3 rand_c = random_on_unit_sphere(seed);


            bool is_sphere_dielectric = t_sphere.is_dielectric.x > 0.5;
            bool is_sphere_reflect = t_sphere.is_reflect.x > 0.5;

            if (is_sphere_dielectric) {
                float ir = 0.8; //1.5;
                float refraction_ratio = N.z > 0 ? (1.0 / ir) : ir;

                direction = refract(r.dir, N, refraction_ratio); //  r.dir; //

                attenuation += vec3(0, 0, 0);

            } else if (is_sphere_reflect) {
                direction = reflect(r.dir, N);
                direction += t_sphere.fuzz.x * rand_c;  //fuzz.x

                attenuation += t_sphere.color.xyz * (1 - t_sphere.albedo.xyz);
            }
            else {

                if (rand_c.z < 0) {
                    rand_c.z *= -1;
                }

                direction = N + rand_c;
                attenuation += t_sphere.color.xyz;
            }

            t_d = 0;

            for (int j = 0; j < spheres_count; j++) {
                t_sphere = spheres[j];
                r = createRay(p, direction, r.time);

                t_d = calculateSphere(r, t_sphere);
                if (t_d > 0) {
                    break;
                }
            }
        }

    }

    vec3 unit_direction = unit_vector(r.dir);

    float blend_a = 0.9 * (unit_direction.y + 1.0);
    vec3 white = vec3(1.0, 1.0, 1.0);
    vec3 blue = vec3(1, 0.4, 0.7);
    vec3 final_color = (1.0 - blend_a) * white + blend_a * blue;

    if (attenuation.x != 0) {
        final_color = normalize(final_color + attenuation) / depth;
    }

    /// figure trace from mesh //
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


mat3 calculateRotationMatrix(vec2 mousePos, float sensitivity) {
    // Calculate the rotation angles based on mouse position
    float angleX = radians((mousePos.y - 0.5) * sensitivity);
    float angleY = radians((mousePos.x - 0.5) * sensitivity);

    // Define the rotation axis (e.g., around the Z-axis)
    vec3 axis = vec3(0.0, 0.0, 1.0);

    // Calculate the rotation matrices for X and Y rotations
    mat3 rotateX = mat3(
        1.0, 0.0, 0.0,
        0.0, cos(angleX), -sin(angleX),
        0.0, sin(angleX), cos(angleX)
    );

    mat3 rotateY = mat3(
        cos(angleY), 0.0, sin(angleY),
        0.0, 1.0, 0.0,
        -sin(angleY), 0.0, cos(angleY)
    );

    // Combine the rotation matrices
    mat3 rotationMatrix = rotateY * rotateX;

    return rotationMatrix;
}



void main() {
    ivec2 pixelCoords = ivec2(gl_FragCoord.xy);
    float aspect_ratio = float(width) / float(height);

    float x = float(pixelCoords.x) / float(width);
    float y = (float(pixelCoords.y) / float(height) - (1.0 - aspect_ratio) / 2.0) / aspect_ratio;

    float small_radius = 0.08;
    //Sphere sphere3 = Sphere(vec3(0.25, 0.5, -0.8), vec3(0.5, 0.2, 0.7), vec3(0.2, 0.2, 0.2), small_radius, false, 0, false);

    //Sphere sphere1 = Sphere(vec3(0.42, 0.52, -0.8), vec3(0.5, 0.5, 0.2), vec3(0.9, 0.9, 0.9), small_radius, true, 0.01, false);
    //Sphere sphere2 = Sphere(vec3(0.55, 0.58, -0.8), vec3(0.9, 0.2, 0.3), vec3(0.2, 0.2, 0.2), 0.05, true, 0.3, false);


    //float surface_radius = 100;
    //Sphere sphere4 = Sphere(vec3(0.5, 0.5 - surface_radius - small_radius, -1), vec3(0.2, 0.8, 0.2), vec3(1, 1, 1), surface_radius, false, 0, false);

    //Sphere sphere5 = Sphere(vec3(0.6, 0.5, -0.5), vec3(1, 1, 1), vec3(0.2, 0.2, 0.2), small_radius, false, 0, true);


    //Sphere[5]t_list = { sphere5, sphere3, sphere1, sphere2,  sphere4 }; //, 



    // camera rot 
    mat3 rotation = calculateRotationMatrix(mousePos, 15.0);


    vec3 temp_dir = vec3(x, y, -1) - camera_center;
    temp_dir = rotation * temp_dir;


    int samples_per_pixel = 100;
    float sample_side = 0.001;

    vec4 sampled_color;

    for (int i = 0; i < samples_per_pixel; i++) {
        float rand_f = rand(gl_FragCoord.xy + i);
        float rand_dir = sample_side * (2 * rand_f - 1);


        vec3 sampled_dir = temp_dir + rand_dir;

        Ray r = createRay(camera_center, sampled_dir, rand_f);

        sampled_color += vec4(inter_ray_color(r, float(i) + rand_dir), 0);
    }


    float gamma = 1.2;
    FragColor = sampled_color / samples_per_pixel * gamma;


}