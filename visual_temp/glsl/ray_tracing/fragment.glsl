#version 430 core

layout(std430, binding = 0) buffer pixelsBuffer {
    vec3 pixels[];
};

out vec4 FragColor;

void main() {
    ivec2 pixelCoords = ivec2(gl_FragCoord.xy);

    // Access the color for the current pixel from the SSBO
    FragColor = vec4(pixels[pixelCoords.x + pixelCoords.y * 800], 0);
}