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

#ifndef _TOYBROT_ISPC_ARCH_
    #define _TOYBROT_ISPC_ARCH_ = "Unknown(host?)"
#endif

static size_t oversubscriptionFactor = 20;
static size_t numTasks = std::thread::hardware_concurrency()*oversubscriptionFactor;
static ispc::LoopType loop = ispc::LoopType::TILED;

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


    std::vector<std::future<bool> > results(numTasks);
//    ispc::Camera c(*reinterpret_cast<const ispc::Camera*>(cam.get()));
//    ispc::Parameters p(*reinterpret_cast<const ispc::Parameters*>(parameters.get()));


    auto traceTask = [this, heightStep, numTasks = numTasks,h0 = lastHeight, mode = loop](size_t idx)
                                                                            {
                                                                                return ispc::traceRegion(
                                                                                            reinterpret_cast<tbFPType(*)[4]>(outBuffer->data()),
                                                                                            *reinterpret_cast<const ispc::Camera*>(cam.get()),
                                                                                            *reinterpret_cast<const ispc::Parameters*>(parameters.get()),
                                                                                            h0, heightStep, idx, numTasks, mode);
                                                                            };

    for(size_t i = 0; i < numTasks; i++)
    {
        results[i] = std::async(std::launch::async, traceTask, i);
    }

    lastHeight+= heightStep;

    for(auto& task: results)
    {
        //Block until all tasks are finished
        if(task.get())
        {
            //If one of them is done, they all must be
            lastHeight = 0;
            finishedGeneration = true;
//            std::cout << "Minimum bounds = " << boxMins.X() << " " << boxMins.Y() << " " << boxMins.Z() << std::endl;
//            std::cout << "Maximum bounds = " << boxMaxes.X() << " " << boxMaxes.Y() << " " << boxMaxes.Z() << std::endl;
        }
    }
    return finishedGeneration;
}

FracGen::FracGen(bool benching, CameraPtr c, ParamPtr p, LoopMode mode, size_t threadMulti)
    : bench{benching}
    , cam{c}
    , parameters{p}
    , lastHeight{0}
{

    outBuffer = std::make_shared< colourVec >(cam->ScreenWidth()*cam->ScreenHeight());

    if(threadMulti != 0)
    {
        oversubscriptionFactor = threadMulti;
        numTasks = std::thread::hardware_concurrency() * oversubscriptionFactor;
    }



    std::string loopString;

    switch (mode)
    {
        case LoopMode::FOREACH:
            loop = ispc::LoopType::FOREACH;
            loopString = "foreach(x = 0 ... width, y = 0 ... height)";
            break;
        case LoopMode::TILED:
            loop = ispc::LoopType::TILED;
            loopString = "foreach_tiled(x = 0 ... width, y = 0 ... height)";
            break;
        case LoopMode::STRIDED:
            loop = ispc::LoopType::STRIDED;
            loopString = "\"strided\" for(idx = a; idx < b; idx += ProgramCount)";
            break;

    }
    static bool once = false;
    if(!bench || !once)
    {
        std::cout << "System reports " << std::thread::hardware_concurrency() << " native threads" << std::endl;
        std::cout << "ISPC running on " << numTasks << " parallel std::tasks" << std::endl;
        std::cout << "Each task is on " << ispc::gangSize() << "-wide gangs" << std::endl;
        std::cout << "ISPC loop is a -> " << loopString << std::endl;
        std::cout << "ISPC_ARCH =  " << _TOYBROT_ISPC_ARCH_ << std::endl;
        once = true;
    }

}

std::string FracGen::getFlavour() noexcept
{
    std::string s = "ISPC-";
    s += _TOYBROT_ISPC_ARCH_;
    #if TOYBROT_USE_DOUBLES
        s += "-db";
    #endif
    return s;
}

FracGen::~FracGen()
{}



