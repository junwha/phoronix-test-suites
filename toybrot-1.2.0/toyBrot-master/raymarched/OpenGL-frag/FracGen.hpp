#ifndef FRACGEN_HPP_DEFINED
#define FRACGEN_HPP_DEFINED

#include <vector>
#include <memory>
#include <map>

#include "defines.hpp"
#include "Vec.hxx"
#include "dataTypes.hxx"

#include <GL/gl.h>

using RGBA = Vec4<tbFPType>;
using colourVec = std::vector<RGBA>;
using FracPtr = std::shared_ptr< colourVec >;
using CameraPtr = std::shared_ptr< Camera<tbFPType> >;
using ParamPtr = std::shared_ptr< Parameters<tbFPType> >;

class FracGen
{
public:

    FracGen(bool benching = false, CameraPtr c = nullptr, ParamPtr p = nullptr);
    ~FracGen();
    FracPtr getBuffer() noexcept { return outBuffer;}
    size_t outLength() const noexcept { return outBuffer->size();}
    size_t outSize() const noexcept { return sizeof(RGBA) * outBuffer->size();}

    void Generate();

private:

    void setUniforms();

    bool bench;

    GLuint glProgram;

    GLint outBuffLocation;
    GLint cameraLocation;
    GLint paramsLocation;

    CameraPtr cam;
    ParamPtr parameters;
    FracPtr outBuffer;

    std::map<std::string, GLint> camLocs, paramLocs;
};

#endif //FRACGEN_HPP_DEFINED
