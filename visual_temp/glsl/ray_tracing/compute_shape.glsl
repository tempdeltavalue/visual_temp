#version 430 core

layout( local_size_x = 64, local_size_y =1, local_size_z = 1  ) in;

uniform int width;
uniform int height;


layout(std430, binding=0) buffer pixelsBuffer
{
    vec3 pixels[];
};



struct Ray {
    vec3 orig;
    vec3 dir;
};

Ray createRay(vec3 origin, vec3 direction) {
    Ray ray;
    ray.orig = origin;
    ray.dir = direction;
    return ray;
}

vec3 rayOrigin(Ray ray) {
    return ray.orig;
}

vec3 rayDirection(Ray ray) {
    return ray.dir;
}

vec3 rayAt(Ray ray, float t) {
    return ray.orig + t * ray.dir;
}

vec3 unit_vector(vec3 v) {
    float length = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    return vec3(v.x / length, v.y / length, v.z / length);
}

bool hit_sphere(vec3 center, float radius, Ray r) {
    vec3 oc = r.orig - center;
    float a = dot(r.dir, r.dir);
    float b = 2.0 * dot(oc, r.dir);
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - 4.0 * a * c;
    return (discriminant >= 0.0);
}


// Function to compute the ray color
vec3 ray_color(Ray r) {
    vec3 sphere_coords = vec3(0.5, 0.1, 0);
    if (hit_sphere(sphere_coords, 0.2, r))
        return vec3(1, 0, 0);

    vec3 unit_direction = unit_vector(r.dir);
    float a = 0.5 * (unit_direction.y + 1.0);
    vec3 white = vec3(1.0, 1.0, 1.0);
    vec3 blue = vec3(0.5, 0.7, 1.0);
    return (1.0 - a) * white + a * blue;
}

void main() {
    uint flattenedIndex = gl_GlobalInvocationID.x;

    uint i = flattenedIndex / height;
    uint j = flattenedIndex % height;
    vec3 camera_center = vec3(0, 0, 0);

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            vec3 ray_direction = vec3(i, j, 0);

            Ray r = createRay(camera_center, ray_direction);

            pixels[flattenedIndex] = ray_color(r);
        }
    }
}