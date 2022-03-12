#include "FracGen.hpp"

#include <hip/hip_runtime.h>
#include <iostream>
#include <cfloat>

__device__ RGBA getColour(unsigned int it)
{
  RGBA colour;
  colour.r = it == 25600? 0 : min(it, 255u);
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

__device__ uint32_t MapSDLRGBA(RGBA colour,  SDL_PixelFormat format)
{
    return  ( colour.r >> format.Rloss) << format.Rshift
            | (colour.g >> format.Gloss) << format.Gshift
            | (colour.b >> format.Bloss) << format.Bshift
            | ((colour.a >> format.Aloss) << format.Ashift & format.Amask  );
}

__global__ void calculateIterations(uint32_t* data, int width, int height, Region r, SDL_PixelFormat format)
{
    int row = hipThreadIdx_x;
    int col = hipBlockIdx_x;
    int index = ((row*width)+col);
    if (index > width*height)
    {
        return;
    }
    unsigned int iteration_factor = 100;
    unsigned int max_iteration = 256 * iteration_factor;

    double incX = (r.Rmax - r.Rmin)/width;
    double incY = (r.Imax - r.Imin)/height;
    incX = incX < 0 ? -incX : incX;
    incY = incY < 0 ? -incY : incY;

    double x = r.Rmin+(col*incX);
    double y = r.Imax-(row*incY);
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

    data[index] = MapSDLRGBA(getColour(iteration), format);
    //data[index] = iteration;


}

void FracGen::Generate(uint32_t* v, SDL_PixelFormat* format, int width, int height, Region r)
{
    if(format == nullptr)
    {
        return;
    }
    uint32_t* devVect;
    hipMalloc( &devVect, width*height*sizeof(uint32_t));

    hipLaunchKernelGGL(calculateIterations, dim3(width), dim3(height), 0, 0, devVect, width, height, r, *format);
    hipDeviceSynchronize();
    hipMemcpy( v, devVect, width*height*sizeof(uint32_t), hipMemcpyDeviceToHost);
    hipFree(devVect);
}

FracGen::FracGen(bool benching)
{
      hipDeviceProp_t devProp;
      hipGetDeviceProperties(&devProp, 0);
      static bool once = false;
      if(!once || !benching )
      {
          once = true;
          std::cout << " System minor "    << devProp.minor << std::endl;
          std::cout << " System major "    << devProp.major << std::endl;
          std::cout << " Device name "     << devProp.name  << std::endl;
      }
}

FracGen::~FracGen()
{
    hipDeviceReset();
}


bool operator==(const Region &r1, const Region &r2)
{
    return (   (r1.Imax - r2.Imax <= DBL_EPSILON) && (r1.Imin - r2.Imin <= DBL_EPSILON)
            && (r1.Rmax - r2.Rmax <= DBL_EPSILON) && (r1.Rmin - r2.Rmin <= DBL_EPSILON) );
}
