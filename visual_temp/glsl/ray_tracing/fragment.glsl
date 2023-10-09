#version 430 core


out vec4 FragColor;


uniform int width;
uniform int height;

uniform vec3 camera_center;

uniform vec2 mousePos;


/// Ray struct

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

float length(vec3 v) {
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

vec3 unit_vector(vec3 v) {
    float length = length(v);
    return vec3(v.x / length, v.y / length, v.z / length);
}

float distance(vec3 p1, vec3 p2) {
    float dx = p2.x - p1.x;
    float dy = p2.y - p1.y;
    float dz = p2.z - p1.z;

    return sqrt(dx * dx + dy * dy + dz * dz);
}


vec3 ray_color(Ray r) {
    vec3 sphere_coords = vec3(0.5, 0.5, -1);
    float radius = 0.1;

    vec3 oc = r.orig - sphere_coords;
    float a = dot(r.dir, r.dir);
    float b = 2.0 * dot(oc, r.dir);
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - 4.0 * a * c;

    if (discriminant >= 0.0) {
        float t = (-b - sqrt(discriminant) ) / (2.0*a);
        vec3 N = unit_vector(rayAt(r, t) - vec3(mousePos.x /float(width) , mousePos.y /float(height) ,-1));
        return 0.5*vec3(N.x+1, N.y+1, N.z+1);
    }
    


    vec3 unit_direction = unit_vector(r.dir);

    float blend_a = 0.9 * (unit_direction.y + 1.0);
    vec3 white = vec3(1.0, 1.0, 1.0);
    vec3 blue = vec3(0.5, 0.2, 0.7);
    vec3 background = (1.0-blend_a)*white + a*blue;

    return background;
}

////



void main() {
    ivec2 pixelCoords = ivec2(gl_FragCoord.xy);
    float aspect_ratio = float(width) / float(height);

    float x = float(pixelCoords.x) / float(width);
    float y = (float(pixelCoords.y) / float(height) - (1.0 - aspect_ratio) / 2.0) / aspect_ratio;





    vec3 temp_dir = vec3(x, y, -1) - camera_center; 

    Ray r = createRay(camera_center, temp_dir);
    FragColor = vec4(ray_color(r), 0);
}