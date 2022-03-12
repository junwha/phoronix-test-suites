#ifndef FRACGEN_HPP_DEFINED
#define FRACGEN_HPP_DEFINED

#include <vector>
#include <memory>

#include "defines.hpp"
#include "Vec.hxx"
#include "dataTypes.hxx"

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION  120
#define CL_HPP_MINIMUM_OPENCL_VERSION  120
#include <CL/cl2.hpp>


using RGBA = Vec4<tbFPType>;
using colourVec = std::vector<RGBA>;
using FracPtr = std::shared_ptr< colourVec >;
using CameraPtr = std::shared_ptr< Camera<tbFPType> >;
using ParamPtr = std::shared_ptr< Parameters<tbFPType> >;

class FracGen
{
public:

    FracGen(bool benching = false, CameraPtr c = nullptr, ParamPtr p = nullptr, int reqPlat = -1, int reqDev = -1, bool cpu = false, std::string vendor = "");
    ~FracGen();
    FracPtr getBuffer() noexcept { return outBuffer;}
    size_t outLength() const noexcept { return outBuffer->size();}
    size_t outSize() const noexcept { return sizeof(RGBA) * outBuffer->size();}

    void Generate();

    static void listOpenCLDevices();

private:

    void compileOpenCLKernel();
    static std::string openCLDeviceTypeString(size_t t);

    bool bench;

    cl::Context context;
    std::vector<cl::Device> devices;
    cl::CommandQueue queue;
    cl::Kernel clGen;
    cl::Program clProgram;
    CameraPtr cam;
    ParamPtr parameters;
    FracPtr outBuffer;
};

#endif //FRACGEN_HPP_DEFINED
