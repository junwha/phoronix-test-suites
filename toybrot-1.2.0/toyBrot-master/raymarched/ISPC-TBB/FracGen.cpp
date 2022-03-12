#include "FracGen.hpp"

#if TOYBROT_USE_DOUBLES
    #include "FracGen_ispc.db.h"
#else
    #include "FracGen_ispc.h"
#endif

#include <iostream>
#include <cfloat>
#include <atomic>
#include <vector>

#include <thread>
#include <future>

#include <tbb/tbb.h>

#ifndef _TOYBROT_ISPC_ARCH_
    #define _TOYBROT_ISPC_ARCH_ = "Unknown(host?)"
#endif

/******************************************************************************
 *
 * Thread spawning section
 *
 ******************************************************************************/

bool FracGen::Generate()
{

    if(outBuffer->size() != cam->ScreenWidth()*cam->ScreenHeight())
    {
        outBuffer->resize(cam->ScreenWidth()*cam->ScreenHeight());
    }

    bool finishedGeneration = false;
    int heightStep = bench ? cam->ScreenHeight() : 20;

    size_t grainsize = ispc::gangSize() * 4/** ispc::gangSize() * 0.25*/;
    tbb::blocked_range2d<int,int> range(lastHeight, std::min(lastHeight+heightStep, static_cast<size_t>(cam->ScreenHeight())), grainsize, 0, cam->ScreenWidth(), grainsize);


    auto traceTask = [this](const tbb::blocked_range2d<int,int>& r){return ispc::traceRegion(  reinterpret_cast<tbFPType(*)[4]>(outBuffer->data()),
                                                                            *reinterpret_cast<const ispc::Camera*>(cam.get()),
                                                                            *reinterpret_cast<const ispc::Parameters*>(parameters.get()),
                                                                            r.rows().begin(), r.rows().end(),
                                                                            r.cols().begin(), r.cols().end());};

    tbb::parallel_for(range, traceTask);

    lastHeight+= heightStep;

    if (lastHeight >=static_cast<size_t>(cam->ScreenHeight()) )
    {
        lastHeight = 0;
        finishedGeneration = true;
    }

    return finishedGeneration;
}

FracGen::FracGen(bool benching, CameraPtr c, ParamPtr p)
    : bench{benching}
    , cam{c}
    , parameters{p}
    , lastHeight{0}
{

    outBuffer = std::make_shared< colourVec >(cam->ScreenWidth()*cam->ScreenHeight());

    static bool once = false;
    if(!bench || !once)
    {
        std::cout << "ISPC running with a " << ispc::gangSize() << "-wide gang" << std::endl;
        std::cout << "ISPC_ARCH =  " << _TOYBROT_ISPC_ARCH_ << std::endl;
        once = true;
    }
}

std::string FracGen::getFlavour() noexcept
{
    std::string s = "TBB+ISPC-";
    s += _TOYBROT_ISPC_ARCH_;
    #if TOYBROT_USE_DOUBLES
        s += "-db";
    #endif
    return s;
}

FracGen::~FracGen()
{}



