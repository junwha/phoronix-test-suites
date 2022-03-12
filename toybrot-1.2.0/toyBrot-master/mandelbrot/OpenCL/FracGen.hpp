#ifndef FRACGEN_HPP_DEFINED
#define FRACGEN_HPP_DEFINED

#include <vector>
#include <cstdint>
#include <SDL_pixels.h>

#define __CL_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION  120
#define CL_HPP_MINIMUM_OPENCL_VERSION  120
#include <CL/cl2.hpp>


struct Region{double Imin,Imax,Rmin,Rmax;}; //This struct delimits a region in the Argand-Gauss Plane (R X I)

bool operator==(const Region& r1, const Region& r2);

class FracGen
{
public:

    FracGen(bool bench = false, bool cpu = false, std::string vendor = "");
    ~FracGen();

    void Generate(uint32_t* img, SDL_PixelFormat* format, int width, int height, Region r);


private:

    void compileOpenCLKernel();
    cl::Context context;
    std::vector<cl::Device> devices;
    cl::CommandQueue queue;
    cl::Kernel clGen;
    cl::Program clProgram;
};

#endif //FRACGEN_HPP_DEFINED
