#include "FracGen.hpp"

#include <iostream>
#include <cfloat>
#include <functional>
#include <atomic>
#include <vector>

#include <omp.h>

//static Vec3f boxMaxes;
//static Vec3f boxMins;

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
    int heightStep = bench ? cam->ScreenHeight() : 10;

    traceRegion(*outBuffer, *cam, lastHeight, heightStep);

    lastHeight+= heightStep;
    if(lastHeight >= static_cast<size_t>(cam->ScreenHeight()))
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
//      std::cout << "OpenMP reports " << omp_get_num_procs() << " native threads" << std::endl;
//      once = true;
    }
}

FracGen::~FracGen()
{}



