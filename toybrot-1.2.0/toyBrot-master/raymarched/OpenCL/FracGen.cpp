#include "FracGen.hpp"

#include <iostream>
#include <fstream>
#include <cfloat>
#include <string>

#ifdef TOYBROT_USE_DOUBLES
    using cl_tbFPType  = cl_double;
    using cl_tbFPType3 = cl_double3;
#else
    using cl_tbFPType  = cl_float;
    using cl_tbFPType3 = cl_float3;
#endif

/******************************************************************************
 *
 * OpenCL device side data structs
 *
 ******************************************************************************/

using cl_cam = struct __attribute__ ((packed)) _cl_cam
{
    _cl_cam(const CameraPtr c)
        : pos{{c->Pos().X(), c->Pos().Y(), c->Pos().Z()}}
        , up{{c->Up().X(), c->Up().Y(), c->Up().Z()}}
        , right{{c->Right().X(), c->Right().Y(), c->Right().Z()}}
        , target{{c->Target().X(), c->Target().Y(), c->Target().Z()}}
        , near{c->Near()}
        , fovY{c->FovY()}
        , width {static_cast<cl_uint>(c->ScreenWidth() )}
        , height{static_cast<cl_uint>(c->ScreenHeight())}
        , screenTopLeft{{c->ScreenTopLeft().X(), c->ScreenTopLeft().Y(), c->ScreenTopLeft().Z()}}
        , screenUp{{c->ScreenUp().X(), c->ScreenUp().Y(), c->ScreenUp().Z()}}
        , screenRight{{c->ScreenRight().X(), c->ScreenRight().Y(), c->ScreenRight().Z()}}
    {}

    cl_tbFPType3 pos;
    cl_tbFPType3 up;
    cl_tbFPType3 right;
    cl_tbFPType3 target;

    cl_tbFPType near;
    cl_tbFPType fovY;

    cl_uint  width;
    cl_uint  height;

    cl_tbFPType3 screenTopLeft;
    cl_tbFPType3 screenUp;
    cl_tbFPType3 screenRight;
};


using cl_params = struct __attribute__ ((packed)) _cl_params
{
    _cl_params(const ParamPtr c)
        : hueFactor{c->HueFactor()}
        , hueOffset{static_cast<cl_int>(c->HueOffset())}
        , valueFactor{c->ValueFactor()}
        , valueRange{c->ValueRange()}
        , valueClamp{c->ValueClamp()}
        , satValue{c->SatValue()}
        , bgRed{c->BgRed()}
        , bgGreen{c->BgGreen()}
        , bgBlue{c->BgBlue()}
        , bgAlpha{c->BgAlpha()}
        , maxRaySteps{static_cast<cl_uint>(c->MaxRaySteps())}
        , collisionMinDist{c->CollisionMinDist()}
        , fixedRadiusSq{c->FixedRadiusSq()}
        , minRadiusSq{c->MinRadiusSq()}
        , foldingLimit{c->FoldingLimit()}
        , boxScale{c->BoxScale()}
        , boxIterations{static_cast<cl_uint>(c->BoxIterations())}
    {}

    cl_tbFPType hueFactor;
    cl_int   hueOffset;
    cl_tbFPType valueFactor;
    cl_tbFPType valueRange;
    cl_tbFPType valueClamp;
    cl_tbFPType satValue;
    cl_tbFPType bgRed;
    cl_tbFPType bgGreen;
    cl_tbFPType bgBlue;
    cl_tbFPType bgAlpha;

    cl_uint   maxRaySteps;
    cl_tbFPType collisionMinDist;

    cl_tbFPType fixedRadiusSq;
    cl_tbFPType minRadiusSq;
    cl_tbFPType foldingLimit;
    cl_tbFPType boxScale;
    cl_uint   boxIterations;

};

/******************************************************************************
 *
 * OpenCL setup and kernel call
 *
 ******************************************************************************/

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

    std::string clCompileOptions("");

    #ifdef TOYBROT_USE_DOUBLES
        clCompileOptions += " -DTOYBROT_USE_DOUBLES ";
    #endif

    #ifndef NDEBUG
        clCompileOptions += " -DTOYBROT_DEBUG ";
    #else
        clCompileOptions += " -cl-fast-relaxed-math -cl-mad-enable -cl-no-signed-zeros ";
    #endif

    try
    {        

        clProgram.build(devices, clCompileOptions.c_str());

    }
    catch (cl::Error e)
    {
        std::cerr
        << "OpenCL compilation error" << std::endl
        << clProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0])
        << std::endl;
        exit(1);
    }

    try
    {
        clGen = cl::Kernel(clProgram, "traceRegion");
    }
    catch (cl::Error err)
    {
        std::cerr
            << "OpenCL kernel creation error: "
            << err.what() << "(" << err.err() << ")"
            << std::endl;
        return;
    }
}


void FracGen::Generate()
{

    try
    {
        /*
         * For OpenCL 1.2 we need to make sure the global dimensions are divisible by the worgroup
         * size, since we explicitly required a certain group size from the kernel.
         * So we do some padding here. Note that if you're running on an OpenCL 2 platform, such as
         * AMD's OpenCL implementation, it will not care unless you pass in additional compilation
         * flags
         */

        const cl::NDRange workgroup(16,16,1);
        const size_t globalWidth = cam->ScreenWidth()%workgroup[0] == 0? cam->ScreenWidth() : ((cam->ScreenWidth()/workgroup[0])+1)*workgroup[0];
        const size_t globalHeight = cam->ScreenHeight()%workgroup[1] == 0? cam->ScreenHeight() : ((cam->ScreenHeight()/workgroup[1])+1)*workgroup[1];
        const cl::NDRange dims(globalWidth, globalHeight, 1);
        const cl::NDRange offset(0,0);

        cl_cam c(cam);
        cl_params p(parameters);

        cl::Buffer v_dev(context, CL_MEM_WRITE_ONLY, outSize());
        cl::Buffer v_cam(context, CL_MEM_READ_ONLY, sizeof(cl_cam));
        cl::Buffer v_params(context, CL_MEM_READ_ONLY, sizeof(cl_params));

        clGen.setArg(0, v_dev);
        clGen.setArg(1, v_cam);
        clGen.setArg(2, v_params);
        queue.enqueueWriteBuffer(v_cam,    cl_bool(CL_TRUE), 0, sizeof(cl_cam),    &c);
        queue.enqueueWriteBuffer(v_params, cl_bool(CL_TRUE), 0, sizeof(cl_params), &p);
        queue.enqueueNDRangeKernel(clGen, cl::NullRange, dims, workgroup);
        queue.enqueueReadBuffer(v_dev, cl_bool(CL_TRUE), 0, outSize(), outBuffer->data());

        queue.finish();

    }
    catch (cl::Error err)
    {
        std::cerr
            << "OpenCL runtime error: "
            << err.what() << "(" << err.err() << ")"
            << std::endl;
        return;
    }
}

std::string FracGen::openCLDeviceTypeString(size_t t)
{
    std::string typestr;
    if((t & 0x2) != 0)
    {
        typestr = "CPU";
    }
    else if((t & 0x4) != 0)
    {
        typestr = "GPU";
    }
    else if((t & 0x8) != 0)
    {
        typestr = "ACCELERATOR";
    }
    else if((t & 0x16) != 0)
    {
        typestr = "CUSTOM";
    }

    if((t & 0x1) != 0)
    {
        typestr += "*";
    }
    return typestr;
}

FracGen::FracGen(bool benching, CameraPtr c, ParamPtr p, int reqPlat, int reqDev, bool cpu, std::string vendor)
    : bench{benching}
    , cam{c}
    , parameters{p}
{
    outBuffer = std::make_shared< colourVec >(cam->ScreenWidth()*cam->ScreenHeight());

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

        if(reqPlat >= 0)
        {
            // We have requested a specific device
            if (static_cast<size_t>(reqPlat) >= platform.size())
            {
                std::cerr << "Requested invalid platform (use -l to list available)" << std::endl;
                exit(2);
            }
            std::vector<cl::Device> pldev;

            platform[reqPlat].getDevices(CL_DEVICE_TYPE_ALL, &pldev);

            if (reqDev < 0 || static_cast<size_t>(reqDev) >= pldev.size())
            {
                std::cerr << "Requested invalid device (use -l to list available)" << std::endl;
                exit(2);
            }
            devices.push_back(pldev[reqDev]);
            context = cl::Context(devices);
        }
        else
        {

            for(size_t i = 0; devices.empty() && i < platform.size(); i++)
            {
                std::vector<cl::Device> pldev;
                try
                {
                    if(cpu)
                    {
                        platform[i].getDevices(CL_DEVICE_TYPE_CPU, &pldev);
                    }
                    else
                    {
                        platform[i].getDevices(CL_DEVICE_TYPE_GPU, &pldev);
                    }

                    for(size_t d = 0; devices.empty() && d < pldev.size(); d++)
                    {
                        if (!pldev[d].getInfo<CL_DEVICE_AVAILABLE>()) continue;

                        std::string ext = pldev[d].getInfo<CL_DEVICE_EXTENSIONS>();
                        std::string devvendor = pldev[d].getInfo<CL_DEVICE_VENDOR>();

                        if(vendor != "" && devvendor.find(vendor) == std::string::npos)
                        {
                            continue;
                        }
                        #ifdef TOYBROT_USE_DOUBLES
                            if ( (ext.find("cl_khr_fp64") == std::string::npos &&
                                 ext.find("cl_amd_fp64") == std::string::npos))
                            {
                                continue;
                            }
                        #endif
                        devices.push_back(pldev[d]);
                        context = cl::Context(devices);
                     }
                    std::cout << std::endl;
                }
                catch(...)
                {
                    devices.clear();
                }
            }
        }
    }
    catch (const cl::Error &err)
    {
           std::cerr
               << "OpenCL initialising error: "
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
        //A likely cause if you're using doubles
        //std::cerr << "Devices with double precision not found." << std::endl;
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

void FracGen::listOpenCLDevices()
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

        std::cout << "Probing OpenCL platforms available" << std::endl;

        for(size_t i = 0; i < platform.size(); i++)
        {
            std::string plname = platform[i].getInfo<CL_PLATFORM_NAME>();
            std::string plvendor = platform[i].getInfo<CL_PLATFORM_VENDOR>();
            std::cout << "#" << i << " -> " << plname << " [" << plvendor << "]" << std::endl;

            std::vector<cl::Device> pldev;

            platform[i].getDevices(CL_DEVICE_TYPE_ALL, &pldev);

            for(size_t d = 0; d < pldev.size(); d++)
            {
                if (!pldev[d].getInfo<CL_DEVICE_AVAILABLE>()) continue;

                std::string ext = pldev[d].getInfo<CL_DEVICE_EXTENSIONS>();
                std::string devname = pldev[d].getInfo<CL_DEVICE_NAME>();
                std::string devtype = FracGen::openCLDeviceTypeString(pldev[d].getInfo<CL_DEVICE_TYPE>());

                std::cout << "    #" << i << " " << d << " -> " << devname << " [" << devtype << "]" << std::endl;

             }
            std::cout << std::endl;
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
}

FracGen::~FracGen()
{}
