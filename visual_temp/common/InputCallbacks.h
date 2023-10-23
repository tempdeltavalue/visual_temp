#pragma once
#include <GLFW/glfw3.h>
#include <glm.hpp>
#include <iostream>

glm::vec3 camera_center = glm::vec3(0.5, 0.5, 0);
float mouseX = 0.5;
float mouseY = 0.5;



void cursorCallback(GLFWwindow* win, double xPos, double yPos) {
    std::cout << "Cursor Position at (" << xPos << " : " << yPos << std::endl;
    mouseX = xPos;
    mouseY = yPos;
}


void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
    {
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);
        mouseX = xpos;
        mouseY = ypos;
        std::cout << "Cursor Position at (" << xpos << " : " << ypos << std::endl;
    }
}


void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{

    float delta = 0.01;
    int state = glfwGetKey(window, GLFW_KEY_W);
    if (state == GLFW_PRESS)
    {
        camera_center.z -= delta;
    }

    state = glfwGetKey(window, GLFW_KEY_S);
    if (state == GLFW_PRESS)
    {
        camera_center.z += delta;
    }

    state = glfwGetKey(window, GLFW_KEY_A);
    if (state == GLFW_PRESS)
    {
        camera_center.x -= delta;
    }


    state = glfwGetKey(window, GLFW_KEY_D);
    if (state == GLFW_PRESS)
    {
        camera_center.x += delta;
    }
}