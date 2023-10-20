﻿

float length(vec3 v) {
    return sqrt(dot(v, v));
}

vec3 unit_vector(vec3 v) {
    float length = length(v);
    return vec3(v.x / length, v.y / length, v.z / length);
}


//// random

float PHI = 1.61803398874989484820459;  // Φ = Golden Ratio   

// Doesn't work correctly during pixel sampling
float gold_noise(in vec2 xy, in float seed) {
    return fract(tan(distance(xy * PHI, xy) * seed) * xy.x);
}


float rand(vec2 co) {
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

vec3 rand_vec3_golden_noise(vec2 seed) {
    float randX = gold_noise(gl_FragCoord.xy, seed.x);
    float randY = gold_noise(gl_FragCoord.yx, seed.x);
    float randZ = gold_noise(gl_FragCoord.yx * 0.5, seed.x); 

    return vec3(randX, randY, randZ);
}


vec3 rand_vec3(vec2 seed) {
    float randX = rand(gl_FragCoord.xy + seed);
    float randY = rand(gl_FragCoord.yx + seed);
    float randZ = rand(gl_FragCoord.xy * 0.5 + seed);

    return vec3(randX, randY, randZ);
}


vec3 random_on_unit_sphere(vec2 seed) {
    float theta = rand(seed) * 2.0 * 3.14159265359; // Random angle between 0 and 2*pi
    float phi = rand(seed) * 2.0 * 3.14159265359; // Random angle between 0 and 2*pi

    float x = cos(theta) * cos(phi);
    float y = sin(phi);
    float z = sin(theta) * cos(phi);

    return vec3(x, y, z);
}

vec3 random_unit_vector(vec2 seed) {
    return unit_vector(random_on_unit_sphere(seed));
}

vec3 random_on_hemisphere(vec3 normal, vec2 seed) {
    vec3 on_unit_sphere = random_unit_vector(seed);
    if (dot(on_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
        return on_unit_sphere;
    else
        return -on_unit_sphere;
}


vec3 random_on_hemisphere2(vec3 N, vec2 seed) {
    vec3 randomDirection;
    float d = 0.0;

    // Generate a random direction on the upper hemisphere
    do {
        randomDirection = vec3(
            rand(vec2(gl_FragCoord.xy + seed)),
            rand(vec2(gl_FragCoord.yx + seed)),
            rand(vec2(gl_FragCoord.yx * 0.5 + seed))
        ) * 2.0 - 1.0;

        d = dot(randomDirection, N);
    } while (d < 0.0);

    // Convert randomDirection to a unit vector manually
    randomDirection =  unit_vector(randomDirection);

    return randomDirection;
}


vec3 reflect(vec3 v, vec3 n) {
    return v - 2 * dot(v, n) * n;
}


///////////////





// hemisphere rotation
vec3 rotateVector(vec3 v, vec3 axis, float angle) {
    float cosAngle = cos(angle);
    float sinAngle = sin(angle);
    vec3 rotatedVec = v * cosAngle + cross(axis, v) * sinAngle + axis * dot(axis, v) * (1.0 - cosAngle);
    return rotatedVec;
}

vec3 randomVectorInRotatedHemisphere(vec3 normal, vec2 seed) {
    // Generate a random direction within the original hemisphere

    float theta = rand(vec2(gl_FragCoord.xy) + seed) * 2.0 * 3.14159265359; // Random angle between 0 and 2*pi
    float phi = rand(vec2(gl_FragCoord.yx) + seed) * 0.5 * 3.14159265359; // Random angle between 0 and pi/2

    float x = cos(theta) * sin(phi);
    float y = cos(phi);
    float z = sin(theta) * sin(phi);

    vec3 randomDirection = unit_vector(vec3(x, y, z));

    // Calculate the rotation axis and angle
    vec3 upDirection = vec3(0.0, 0.0, 1.0); // The positive z-axis
    vec3 rotationAxis = cross(upDirection, normal);
    float rotationAngle = acos(dot(upDirection, normal));

    // Rotate the generated direction to align with the given normal
    randomDirection = rotateVector(randomDirection, rotationAxis, rotationAngle);

    return randomDirection;
}
