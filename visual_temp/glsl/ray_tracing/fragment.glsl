#version 430 core

#include sphere_shader.incl
#include vec3_utils.incl

out vec4 FragColor;


uniform int width;
uniform int height;

uniform vec3 camera_center;

uniform vec2 mousePos;

uniform float current_time;


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


vec3 inter_ray_color(Ray r, Sphere[2] t_list) {
    vec3 final_color = vec3(-1.0, -1.0, -1.0); // null

    for (int i = 0; i < 2; i++) {
        Sphere t_sphere = t_list[i];
        float t_d = calculateSphere(r.orig, r.dir, t_sphere.center, t_sphere.radius);

        int depth = 0;
        float coef = 1;

        while (t_d > 0.0) {
            
            if (depth > 5) {
                break;
            }

            vec3 p = rayAt(r, t_d); // intersection point 


            vec3 N = (p - t_sphere.center)/t_sphere.radius; //vec3(mousePos.x /float(width) , mousePos.y /float(height), -1);

            final_color = 0.5*(N + vec3(1, 1, 1));

            float seed = current_time + float(depth) + i;

            vec3 direction = random_on_hemisphere(N, seed); 

            //final_color = direction;
            //direction *= 1.1;

            // what if ray bounced from sphere 1 and intersect with sphere two ?
            //t_d = 0.0;

    
            for (int j = 0; j < 2; j++) {
                if (i == j) { 
                    continue;
                }
                Sphere t_sphere2 = t_list[j];
                t_d = calculateSphere(p, direction, t_sphere2.center, t_sphere2.radius);

                if (t_d > 0.0) {
                    coef *= 0.5;

                    break;
                }
            }


            depth += 1;
        }

        if (final_color.x != -1.0 && final_color.y != -1.0 && final_color.z != -1.0) {
            final_color *= coef;
        }
    }

    if (final_color.x == -1.0 && final_color.y == -1.0 && final_color.z == -1.0) {

        vec3 unit_direction = unit_vector(r.dir);

        float blend_a = 0.9 * (unit_direction.y + 1.0);
        vec3 white = vec3(1.0, 1.0, 1.0);
        vec3 blue = vec3(0.5, 0.2, 0.7);
        final_color= (1.0-blend_a)*white + blend_a*blue;
    }

    return final_color;

}


void main() {
    ivec2 pixelCoords = ivec2(gl_FragCoord.xy);
    float aspect_ratio = float(width) / float(height);

    float x = float(pixelCoords.x) / float(width);
    float y = (float(pixelCoords.y) / float(height) - (1.0 - aspect_ratio) / 2.0) / aspect_ratio;



    vec3 temp_dir = vec3(x, y, -1) - camera_center; 
    Ray r = createRay(camera_center, temp_dir);


    vec3 sphere_coords = vec3(0.5, 0.5, -1);
    float radius = 0.1;

    float radius2 =10;
    vec3 sphere_coords2 = vec3(0.5, 0.5 - radius2 - radius - 0.07, -1);


    Sphere [2]t_list = {Sphere(sphere_coords, radius), Sphere(sphere_coords2, radius2)}; //, 

    FragColor = vec4(inter_ray_color(r, t_list), 0);
}