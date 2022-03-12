#include "FracGen.hpp"

#include <iostream>
#include <cfloat>
#include <atomic>
#include <thread>
#include <vector>

static size_t ThreadMulti = 4;
static size_t numThreads=std::thread::hardware_concurrency()*ThreadMulti;

RGBA getColour(unsigned int it)
{
  RGBA colour;
  colour.r = it == 25600? 0 : static_cast<uint8_t>(std::min(it, 255u));
  colour.g = it == 25600? 0 : static_cast<uint8_t>(std::min(it, 255u));
  colour.b = it == 25600? 0 : static_cast<uint8_t>(std::min(it, 255u));
  colour.r = static_cast<uint8_t>(std::abs(255 - colour.r));
  colour.g = static_cast<uint8_t>(std::abs(255 - colour.g));
  colour.b = static_cast<uint8_t>(std::abs(255 - colour.b));

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

void calculateIterations(uint32_t* data,
                         int width, int height, int heightStep,
                         Region r,
                         SDL_PixelFormat format, int h0,
                         size_t idx, std::vector<bool>& results)
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
            results[idx] = true;
            return;
        }

        //Initially intuitive/illustrative division
//        for(int w = (idx%numThreads)*(width/numThreads);
//                w < ((idx%numThreads)+1)*(width/numThreads);
//                w++)

        //Newer prefetcher-friendly version
        for(int w = 0 + static_cast<int>(idx); w < width; w+= numThreads )
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
    results[idx] = false;
    return;
}

bool FracGen::Generate(uint32_t* v, SDL_PixelFormat* format, int width, int height, Region r)
{
    if(format == nullptr)
    {
        return false;
    }
    static std::atomic<int> h {0};
    static std::vector< std::thread> threadPool(numThreads);

    std::vector<bool> results(threadPool.size());

    bool finishedGeneration = false;
    int heightStep = bench ? height : 10;

    for(unsigned int i = 0; i < numThreads; i++)
    {
        //Fire up all threads
        threadPool[i] = std::thread([v, width, height, heightStep, r, fmt = *format, h0 = h.load(), idx = i, &results](){calculateIterations(v, width, height, heightStep, r, fmt, h0, idx, results);});
    }
    h+= heightStep;
    for(auto& td : threadPool)
    {
        //wait until all threads complete
        td.join();
    }
    //Block until all tasks are finished
    for(bool b : results)
    {
        if(b)
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
      std::cout << " We're going to spawn " << std::thread::hardware_concurrency()*ThreadMulti << " threads in our pool" << std::endl;
    }
}

FracGen::~FracGen()
{}


bool operator==(const Region &r1, const Region &r2)
{
    return (   (r1.Imax - r2.Imax <= DBL_EPSILON) && (r1.Imin - r2.Imin <= DBL_EPSILON)
            && (r1.Rmax - r2.Rmax <= DBL_EPSILON) && (r1.Rmin - r2.Rmin <= DBL_EPSILON) );
}
