#include "FracGen.hpp"

#include <iostream>
#include <cfloat>
#include <CL/sycl.hpp>

inline RGBA getColour(unsigned int it)
{
  RGBA colour;
  colour.r = it == 25600? 0 : std::min(it, 255u)*0.7;
  colour.g = it == 25600? 0 : std::min(it, 255u)*0;
  colour.b = it == 25600? 0 : std::min(it, 255u)*0.5;
  return colour;
}

inline RGBA::operator uint32_t() const
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

struct pxFmt
{
    pxFmt(SDL_PixelFormat format)
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

    uint32_t Amask;
    uint8_t Rloss;
    uint8_t Gloss;
    uint8_t Bloss;
    uint8_t Aloss;
    uint8_t Rshift;
    uint8_t Gshift;
    uint8_t Bshift;
    uint8_t Ashift;
};

inline uint32_t MapSDLRGBA(RGBA colour,  pxFmt format)
{
    return  ( colour.r >> format.Rloss) << format.Rshift
            | (colour.g >> format.Gloss) << format.Gshift
            | (colour.b >> format.Bloss) << format.Bshift
            | ((colour.a >> format.Aloss) << format.Ashift & format.Amask  );
}
template <typename Acc>
void calculateIterations(Acc data,
                         int width,
                         int height,
                         Region r,
                         pxFmt format,
                         cl::sycl::id<1> tid)
{
    int row = tid.get(0)/width;
    int col = tid.get(0)%width;
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

    data[tid.get(0)] = MapSDLRGBA(getColour(iteration), format);

}

void FracGen::Generate(uint32_t* v, SDL_PixelFormat* format, int width, int height, Region r)
{
    if(format == nullptr)
    {
        return;
    }


    try
    {
        cl::sycl::range<1> pixels(width*height);
        cl::sycl::buffer<uint32_t,1> buff (v, pixels);
        cl::sycl::buffer<SDL_PixelFormat,1> fmt (format, 1);
        q.submit([&](cl::sycl::handler& cgh)
                        {
                            auto access_v = buff.get_access<cl::sycl::access::mode::write>(cgh);
                            cgh.parallel_for<class syclbrotkernel>
                                    (  pixels,
                                       [=, fmt = pxFmt(*format)] (cl::sycl::id<1> tid)
                                       { calculateIterations(access_v, width, height, r, fmt, tid);}
                                    );
                        });
        q.wait_and_throw();
    }
    catch(cl::sycl::exception const& e)
    {
        std::cout << "SYCL sync exception -> " << e.what() << std::endl;
    }
    catch(...)
    {
        std::cout << " Exception caught! " << std::endl;
    }
}

FracGen::FracGen(bool benching)
{
    static bool once = false;

    cl::sycl::default_selector device_selector;

    cl::sycl::async_handler sycl_err_handler =  [] (cl::sycl::exception_list exceptions)
                                                {
                                                    for (std::exception_ptr const& e : exceptions)
                                                    {
                                                        try
                                                        {
                                                            std::rethrow_exception(e);
                                                        }
                                                        catch(cl::sycl::exception const& e)
                                                        {
                                                            std::cout << "SYCL async exception -> " << e.what() << std::endl;
                                                        }
                                                    }
                                                };

    q = cl::sycl::queue{device_selector, sycl_err_handler};

    if(!once || !benching )
    {
        std::cout << "Running on "
                  << q.get_device().get_info<cl::sycl::info::device::name>()
                  << std::endl ;
        once = true;
    }

}

FracGen::~FracGen()
{}

bool operator==(const Region &r1, const Region &r2)
{
    return (   (r1.Imax - r2.Imax <= DBL_EPSILON) && (r1.Imin - r2.Imin <= DBL_EPSILON)
            && (r1.Rmax - r2.Rmax <= DBL_EPSILON) && (r1.Rmin - r2.Rmin <= DBL_EPSILON) );
}
