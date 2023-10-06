#version 430 core

layout( local_size_x = 64, local_size_y =1, local_size_z = 1  ) in;

uniform int width;
uniform int height;

uniform vec3 pixel00_loc;
uniform vec3 pixel_delta_u;
uniform vec3 pixel_delta_v;
uniform vec3 camera_center;


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
    vec3 sphere_coords = vec3(0,0, -1);
    if (hit_sphere(sphere_coords, 0.001, r))
        return vec3(1, 0, 0);

    vec3 unit_direction = unit_vector(r.dir);


    float a = 0.5 * (unit_direction.y + 1.0);
    vec3 white = vec3(1.0, 1.0, 1.0);
    vec3 blue = vec3(0.5, 0.7, 1.0);
    vec3 temp = (1.0-a)*white + a*blue;

    return vec3(r.dir.x / 255, r.dir.y / 255 ,0);
}

void main() {
    uint flattenedIndex = gl_GlobalInvocationID.x;

    uint j = flattenedIndex / width;
    uint i = flattenedIndex % width;

    //vec3 pixel_center = vec3(i, j, 0); // pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
    //vec3 ray_direction = pixel_center - camera_center;


    Ray r = createRay(vec3(0,0,0), vec3(i, j, 0));
    pixels[flattenedIndex] = ray_color(r);
}