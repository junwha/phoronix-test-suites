#if defined(cl_khr_fp64)
#  pragma OPENCL EXTENSION cl_khr_fp64: enable
#elif defined(cl_amd_fp64)
#  pragma OPENCL EXTENSION cl_amd_fp64: enable
#else
#  error double precision is not supported
#endif

//These two would probably be better expressed by:
// {uint, uchar8} and {double4} respectively
// but I've left them like this so it's easier to read

struct __attribute__ ((packed)) _sdl_pf_cl
{
    uint Amask;
    uchar Rloss;
    uchar Gloss;
    uchar Bloss;
    uchar Aloss;
    uchar Rshift;
    uchar Gshift;
    uchar Bshift;
    uchar Ashift;
};

struct __attribute__ ((packed)) Region
{
  double Rmin;
  double Rmax;
  double Imin;
  double Imax;
};

uchar4 getColour(unsigned int it)
{
    uchar4 colour;
    colour.s0 = it == 25600? 0 : min(it, 255u)*.95;
    colour.s1 = it == 25600? 0 : min(it, 255u)*.6;
    colour.s2 = it == 25600? 0 : min(it, 255u)*.25;
    colour.s3 = 255u;
    return colour;
}

uint MapSDLRGBA(uchar4 colour,  struct _sdl_pf_cl format)
{
    return  (   colour.s0 >> format.Rloss) << format.Rshift
            | ( colour.s1 >> format.Gloss) << format.Gshift
            | ( colour.s2 >> format.Bloss) << format.Bshift
            | ((colour.s3 >> format.Aloss) << format.Ashift & format.Amask  );
}

kernel void calculateIterations( __global uint* data,
                                int width,
                                int height,
                                struct Region r,
                                struct _sdl_pf_cl format)
{
    int row = get_global_id (1);
    int col = get_global_id (0);
    int index = ((row*width)+col);
    uchar Red = 0;
    uchar Green = 0;
    uchar Blue = 0;
    uchar Alpha = 255;
    if (index > width*height)
    {
        return;
    }
    uint iteration_factor = 100;
    uint max_iteration = 256 * iteration_factor;

    double incX = (r.Rmax - r.Rmin)/width;
    double incY = (r.Imax - r.Imin)/height;
    incX = incX < 0 ? -incX : incX;
    incY = incY < 0 ? -incY : incY;

    double x = r.Rmin+(col*incX);
    double y = r.Imax-(row*incY);
    double x0 = x;
    double y0 = y;

    uint iteration = 0;

    while ( (x*x + y*y <= 4)  &&  (iteration < max_iteration) )
    {
        double xtemp = x*x - y*y + x0;
        y = 2*x*y + y0;

        x = xtemp;

        iteration++;
    }

    data[index] = MapSDLRGBA(getColour(iteration), format);

}
