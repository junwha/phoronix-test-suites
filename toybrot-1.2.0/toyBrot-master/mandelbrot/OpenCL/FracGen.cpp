#include "FracGen.hpp"

#include <iostream>
#include <fstream>
#include <cfloat>
#include <string>

//These two would probably be better expressed by:
// {uint, uchar8} and {double4} respectively
// but I've left them like this so it's easier to read

typedef struct __attribute__ ((packed)) _sdl_pf_cl
{
    _sdl_pf_cl(SDL_PixelFormat format)
        : Amask{format.Amask}
        , Rloss{format.Rloss}
        , Gloss{format.Gloss}
        , Bloss{format.Bloss}
        , Aloss{format.Aloss}
        , Rshift{format.Rshift}
        , Gshift{format.Gshift}
        , Bshift{format.Bshift}
        , Ashift{format.Ashift}
    {}

    cl_uint Amask;
    cl_uchar Rloss;
    cl_uchar Gloss;
    cl_uchar Bloss;
    cl_uchar Aloss;
    cl_uchar Rshift;
    cl_uchar Gshift;
    cl_uchar Bshift;
    cl_uchar Ashift;
} sdl_pf_cl;

typedef struct __attribute__ ((packed)) _reg_cl
{
    _reg_cl(Region r)
        : Rmin{r.Rmin}
        , Rmax{r.Rmax}
        , Imin{r.Imin}
        , Imax{r.Imax}
    {}
    cl_double Rmin;
    cl_double Rmax;
    cl_double Imin;
    cl_double Imax;
} reg_cl;


void FracGen::compileOpenCLKernel()
{
    std::ifstream srcFile("FracGen.cl");
    std::vector<std::string> lines;
    std::string ln;
    lines.push_back(ln);
    if(srcFile.is_open())
    {
        while(std::getline(srcFile,ln))
        {
            ln.append("\n");
            lines.push_back(ln);
        }
    }
    else
    {
        std::cerr << "Could not open OpenCL source file" << std::endl;
        exit(1);
    }
    srcFile.close();
    clProgram = cl::Program(context,lines);

    try
    {
        clProgram.build(devices);
    }
    catch (cl::Error e)
    {
        std::cerr
        << "OpenCL compilation error" << std::endl
        << clProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0])
        << std::endl;
        exit(1);
    }
    clGen = cl::Kernel(clProgram, "calculateIterations");
}


void FracGen::Generate(uint32_t* v, SDL_PixelFormat* format, int width, int height, Region r)
{
    if(format == nullptr)
    {
        return;
    }

    size_t bufferSize = sizeof(uint32_t) * static_cast<size_t>(width) * static_cast<size_t>(height);

    sdl_pf_cl fmt(*format);
    reg_cl    reg(r);
    try
    {
        const cl::NDRange dims(static_cast<size_t>(width), static_cast<size_t>(height));
        const cl::NDRange offset(0,0);

        cl::Buffer v_dev(context,CL_MEM_WRITE_ONLY, bufferSize);

        clGen.setArg(0, v_dev);
        clGen.setArg(1, width);
        clGen.setArg(2, height);
        clGen.setArg(3, reg);
        clGen.setArg(4, fmt);
        queue.enqueueNDRangeKernel(clGen, cl::NullRange, dims, cl::NullRange);
        queue.enqueueReadBuffer(v_dev,CL_TRUE,0,bufferSize,v);
    }
    catch (cl::Error err)
    {
        std::cerr
            << "OpenCL error: "
            << err.what() << "(" << err.err() << ")"
            << std::endl;
        return;
    }
}

FracGen::FracGen(bool bench, bool cpu, std::string vendor)
{
    try
    {
        // Get list of OpenCL platforms.
        std::vector<::cl::Platform> platform;
        cl::Platform::get(&platform);

        if (platform.empty())
        {
            std::cerr << "OpenCL platforms not found." << std::endl;
            exit(1);
        }

        for(auto p = platform.begin(); devices.empty() && p != platform.end(); p++)
        {
            std::vector<cl::Device> pldev;
            try
            {
                if(cpu)
                {
                    p->getDevices(CL_DEVICE_TYPE_CPU, &pldev);
                }
                else
                {
                    p->getDevices(CL_DEVICE_TYPE_GPU, &pldev);
                }

                for(auto d = pldev.begin(); devices.empty() && d != pldev.end(); d++)
                {
                    if (!d->getInfo<CL_DEVICE_AVAILABLE>()) continue;


                    std::string ext = d->getInfo<CL_DEVICE_EXTENSIONS>();
                    std::string devvendor = d->getInfo<CL_DEVICE_VENDOR>();
                    if(vendor != "" && devvendor.find(vendor) == std::string::npos)
                    {
                        continue;
                    }

                    if ( (ext.find("cl_khr_fp64") == std::string::npos &&
                         ext.find("cl_amd_fp64") == std::string::npos))
                    {
                        continue;
                    }
                    devices.push_back(*d);
                    context = cl::Context(devices);
                 }
            }
            catch(...)
            {
                devices.clear();
            }
        }
    }
    catch (const cl::Error &err)
    {
           std::cerr
               << "OpenCL error: "
               << err.what() << "(" << err.err() << ")"
               << std::endl;
           return;
    }
    if (devices.empty())
    {
        if(vendor != "")
        {
            std::cout << "Specifically requested vendor = " << vendor << std::endl;
        }
        std::cerr << "Devices with double precision not found." << std::endl;
        exit(1);
    }
    static bool once = false;
    if(!once || !bench )
    {
        once = true;
        if(vendor != "")
        {
            std::cout << "Specifically requested vendor = " << vendor << std::endl;
        }
        std::cout << "Running OpenCL on: " << devices[0].getInfo<CL_DEVICE_NAME>()   << std::endl;
        std::cout << "With vendor: "       << devices[0].getInfo<CL_DEVICE_VENDOR>() << std::endl;
        std::cout << "Driver version: "    << devices[0].getInfo<CL_DEVICE_VERSION>() << std::endl;
        std::cout << "OpenCL C version: "  << devices[0].getInfo<CL_DEVICE_OPENCL_C_VERSION>() << std::endl;
    }

    try
    {
        queue = cl::CommandQueue{context, devices[0]};
    }
    catch (const cl::Error &err)
    {
           std::cerr
               << "OpenCL error: "
               << err.what() << "(" << err.err() << ")"
               << std::endl;
           exit(1);
    }
    compileOpenCLKernel();

}

FracGen::~FracGen()
{
   // hipDeviceReset();
}


bool operator==(const Region &r1, const Region &r2)
{
    return (   (r1.Imax - r2.Imax <= DBL_EPSILON) && (r1.Imin - r2.Imin <= DBL_EPSILON)
            && (r1.Rmax - r2.Rmax <= DBL_EPSILON) && (r1.Rmin - r2.Rmin <= DBL_EPSILON) );
}
