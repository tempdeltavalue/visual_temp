
// https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm

bool rayTriangleIntersect(
    vec3 orig, vec3 dir,
    vec3 v0, vec3 v1, vec3 v2)
{
    // Compute the plane's normal
    vec3 edge0 = v1 - v0;
    vec3 v0v2 = v2 - v0;


    vec3 N = cross(edge0, v0v2);


    // Check if the ray and plane are parallel
    float NdotRayDirection = dot(N, dir);
    if (abs(NdotRayDirection) < 1e-6)
        return false; // They are parallel, so they don't intersect!

    // Compute d parameter using equation 2
    float d = -dot(N, v0);

    // Compute t (equation 3)
    float t = -(dot(N, orig) + d) / NdotRayDirection;

    // Check if the triangle is behind the ray
    if (t < 0.0)
        return false; // The triangle is behind

    // Compute the intersection point using equation 1
    vec3 P = orig + t * dir;

    // Step 2: Inside-Outside Test
    vec3 C; // Vector perpendicular to the triangle's plane

    // Edge 0
    vec3 vp0 = P - v0;
    C = cross(edge0, vp0);

    if (dot(N, C) < 0.0)
        return false; // P is on the right side

    // Edge 1
    vec3 edge1 = v2 - v1;
    vec3 vp1 = P - v1;
    C = cross(edge1, vp1);
    if (dot(N, C) < 0.0)
        return false; // P is on the right side

    // Edge 2
    vec3 edge2 = v0 - v2;
    vec3 vp2 = P - v2;
    C = cross(edge2, vp2);
    if (dot(N, C) < 0.0)
        return false; // P is on the right side;

    return true; // This ray hits the triangle
}

float calculateSphere(Ray r, Sphere sphere) {
    vec4 center = sphere.is_moving.x > 0.5 ? getCenterAtTime(sphere, r.time) : sphere.center;

    vec3 oc = r.orig - center.xyz;
    float a = dot(r.dir, r.dir);
    float half_b = dot(oc, r.dir);
    float c = dot(oc, oc) - sphere.radius.x * sphere.radius.x;
    float discriminant = half_b * half_b - a * c;
       
    if (discriminant <= 0.0) {
        return -1;
    }

    float sqrt_d = sqrt(discriminant);
    float root = (-half_b - sqrt_d) / a;

    //float ray_tmin = -0.001;
    //float ray_tmax = 1;

    //if (root <= ray_tmin || ray_tmax <= root) {
    //    root = (-half_b + sqrt_d) / a;
    //    if (root <= ray_tmin || ray_tmax <= root)
    //        return -1;
    //}


    return root;
    
}