#include <string>
#include <glm.hpp>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glad/glad.h>

#ifndef N_BODY_SHADER_H
#define N_BODY_SHADER_H


class Shader {
public:
    // Shader ID
    unsigned int ID;

    virtual ~Shader();


    Shader();
    Shader(GLuint shaderType, std::string& computeShaderPath, unsigned int pID =-1);

    /**
     * Activates the shader
     */
    void use() const;

    // Uniform functions
    // ------------------------------------------------------------------------
    void setBool(const std::string &name, bool value) const;

    // ------------------------------------------------------------------------
    void setInt(const std::string &name, int value) const;

    // ------------------------------------------------------------------------
    void setFloat(const std::string &name, float value) const;

    // ------------------------------------------------------------------------
    void setVec2(const std::string &name, const glm::vec2 &value) const;

    void setVec2(const std::string &name, float x, float y) const;


    // ------------------------------------------------------------------------
    void setVec3(const std::string &name, const glm::vec3 &value) const;

    void setVec3(const std::string &name, float x, float y, float z) const;

    // ------------------------------------------------------------------------
    void setVec4(const std::string &name, const glm::vec4 &value) const;

    void setVec4(const std::string &name, float x, float y, float z, float w) const;

    // ------------------------------------------------------------------------
    void setMat2(const std::string &name, const glm::mat2 &mat) const;

    // ------------------------------------------------------------------------
    void setMat3(const std::string &name, const glm::mat3 &mat) const;

    // ------------------------------------------------------------------------
    void setMat4(const std::string &name, const glm::mat4 &mat) const;


protected:

    void checkCompileErrors(unsigned int shader, std::string type);


    void fixPathSeparators(std::string &path);



    unsigned int compileShader(GLuint shaderType, const char *code);


    std::string readFileToString(std::string& filePath);

};
#endif //SHADER_H
