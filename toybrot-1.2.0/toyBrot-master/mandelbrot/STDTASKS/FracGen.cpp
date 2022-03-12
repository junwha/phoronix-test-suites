#include "FracGen.hpp"

#include <iostream>
#include <cfloat>
#include <atomic>
#include <thread>
#include <future>
#include <vector>

static size_t tasksPerThread = 4;
static size_t numTasks=std::thread::hardware_concurrency()*tasksPerThread;

RGBA getColour(unsigned int it)
{
  RGBA colour;
  colour.r = it == 25600? 0 : static_cast<uint8_t>(std::min(it, 255u));
  colour.g = it == 25600? 0 : static_cast<uint8_t>(std::min(it, 255u));
  colour.b = it == 25600? 0 : static_cast<uint8_t>(std::min(it, 255u));
  return colour;
}

RGBA::operator uint32_t() const
{
    uint32_t colour = 0;
    colour = colour | r;
    colour = colour << 8;
    colour = colour | g;
    colour = colour << 8;
    colour = colour | b;
    colour = colour << 8;
    colour = colour | a;
    return colour;

}

uint32_t MapSDLRGBA(RGBA colour,  SDL_PixelFormat format)
{
    return    ( colour.r >> format.Rloss) << format.Rshift
            | ( colour.g >> format.Gloss) << format.Gshift
            | ( colour.b >> format.Bloss) << format.Bshift
            | ((colour.a >> format.Aloss) << format.Ashift & format.Amask  );
}

bool calculateIterations(uint32_t* data,
                         int width, int height, int heightStep,
                         Region r,
                         SDL_PixelFormat format, int h0, size_t idx)
{

    unsigned int iteration_factor = 100;
    unsigned int max_iteration = 256 * iteration_factor;

    double incX = (r.Rmax - r.Rmin)/width;
    double incY = (r.Imax - r.Imin)/height;
    incX = incX < 0 ? -incX : incX;
    incY = incY < 0 ? -incY : incY;

    for(int h = h0; h < h0+heightStep; h++)
    {
        if (h >= height)
        {
            return true;
        }

        //Initially intuitive/illustrative division
//        for(int w = (idx)*(width/numTasks);
//                w <= ((idx)+1)*(width/numTasks);
//                w++)

        //Newer prefetcher-friendly version
        for(int w = 0 + static_cast<int>(idx); w < width; w+= numTasks )
        {
            //These tasks don't automagically know their ID nor the total of tasks running
            double x = r.Rmin+(w*incX);
            double y = r.Imax-(h*incY);
            double x0 = x;
            double y0 = y;

            unsigned int iteration = 0;

            while ( (x*x + y*y <= 4)  &&  (iteration < max_iteration) )
            {
                double xtemp = x*x - y*y + x0;
                y = 2*x*y + y0;

                x = xtemp;

                iteration++;
            }

            data[(h*width)+w] = MapSDLRGBA(getColour(iteration), format);
        }
    }
    return false;
}

bool FracGen::Generate(uint32_t* v, SDL_PixelFormat* format, int width, int height, Region r)
{
    if(format == nullptr)
    {
        return false;
    }
    static std::atomic<int> h {0};
    bool finishedGeneration = false;
    int heightStep = bench ? height : 10;
    /** Make a clean lambda to wrap the function call for std::async
    *   Can also be achieved by:
    *  - making the function a functor instead
    *  - using either std::function or std::bind
    *
    *   Lambda has you not worrying about type and is also the
    *   most sanitary conversion. Nothing in the function itself changes
    */
    auto genFrac = [](uint32_t* data, int width, int height, int heightStep, Region r, SDL_PixelFormat format, int h0, size_t idx)
                     {return calculateIterations(data,width,height, heightStep, r,format,h0,idx);};

    std::vector< std::future<bool>> tasks(numTasks);
    for(unsigned int i = 0; i < numTasks; i++)
    {
        tasks[i] = std::async(std::launch::async, genFrac, v, width, height, heightStep, r, *format, h.load(), i);
    }
    h+= heightStep;

    for(unsigned int i = 0; i < tasks.size(); i++)
    {
        //Block until all tasks are finished
        if(tasks[i].get())
        {
            //If one of them is done, they all must be
            h.store(0);
            finishedGeneration = true;
        }
    }
    return finishedGeneration;
}

FracGen::FracGen(bool benching)
    :bench{benching}
{
    if(!bench)
    {
      std::cout << " System reports " << std::thread::hardware_concurrency() << " native threads" << std::endl;
      std::cout << " We're going to spawn " << std::thread::hardware_concurrency()*tasksPerThread << " tasks" << std::endl;
    }
}

FracGen::~FracGen()
{}


bool operator==(const Region &r1, const Region &r2)
{
    return (   (r1.Imax - r2.Imax <= DBL_EPSILON) && (r1.Imin - r2.Imin <= DBL_EPSILON)
            && (r1.Rmax - r2.Rmax <= DBL_EPSILON) && (r1.Rmin - r2.Rmin <= DBL_EPSILON) );
}
