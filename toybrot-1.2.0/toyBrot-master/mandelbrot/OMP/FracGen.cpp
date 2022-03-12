#include "FracGen.hpp"

#include <iostream>
#include <cfloat>
#include <atomic>

#include<omp.h>

RGBA getColour(unsigned int it)
{
  RGBA colour;
  colour.r = it == 25600? 0 : static_cast<uint8_t>(std::min(it, 255u));
  colour.g = it == 25600? 0 : static_cast<uint8_t>(std::min(it, 255u));
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
                         int width, int height,
                         Region r,
                         SDL_PixelFormat format, int h0)
{

    unsigned int iteration_factor = 100;
    unsigned int max_iteration = 256 * iteration_factor;

    double incX = (r.Rmax - r.Rmin)/width;
    double incY = (r.Imax - r.Imin)/height;
    incX = incX < 0 ? -incX : incX;
    incY = incY < 0 ? -incY : incY;

    for(int h = h0; h < h0+10; h++)
    {
        if (h >= height)
        {
            return true;
        }
        //For OpenMP, somewhat like for GPUs we're assigned a thread ID automagically

        //Initially intuitive/illustrative division
//        for(int w = (omp_get_thread_num()%omp_get_num_threads())*(width/omp_get_num_threads());
//                w < ((omp_get_thread_num()%omp_get_num_threads())+1)*(width/omp_get_num_threads());
//                w++)

        //Newer prefetcher-friendly version
        for(int w = 0 + omp_get_thread_num(); w < width; w+= omp_get_num_threads() )
        {
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

    std::vector<bool> results(static_cast<size_t>(omp_get_num_procs()));
    #pragma omp parallel for
    for(unsigned int i = 0; i < omp_get_num_procs(); i++)
    {
        results[i]=calculateIterations(v, width, height, r, *format, h);
    }
    h+= 10;

    for(unsigned int i = 0; i < results.size(); i++)
    {
        if(results[i])
        {
            h.store(0);
            finishedGeneration = true;
        }
    }
    return finishedGeneration;
}

FracGen::FracGen(bool benching)
    :bench{benching}
{
//      hipDeviceProp_t devProp;
//      hipGetDeviceProperties(&devProp, 0);
//      std::cout << " System minor "    << devProp.minor << std::endl;
//      std::cout << " System major "    << devProp.major << std::endl;
//      std::cout << " Device name "     << devProp.name  << std::endl;
}

FracGen::~FracGen()
{}


bool operator==(const Region &r1, const Region &r2)
{
    return (   (r1.Imax - r2.Imax <= DBL_EPSILON) && (r1.Imin - r2.Imin <= DBL_EPSILON)
            && (r1.Rmax - r2.Rmax <= DBL_EPSILON) && (r1.Rmin - r2.Rmin <= DBL_EPSILON) );
}
