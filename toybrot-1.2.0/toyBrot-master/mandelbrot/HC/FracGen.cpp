#include "FracGen.hpp"

#include <hc.hpp>
#include <iostream>
#include <cfloat>

RGBA getColour(unsigned int it) [[hc]]
{
  RGBA colour;
  colour.r = it == 25600? 0 : static_cast<uint8_t>(std::min(it, 255u));
  colour.g = it == 25600? 0 : 0;
  colour.b = it == 25600? 0 : 0;
//  colour.r = static_cast<uint8_t>((255u - colour.r) > 0 ? (255u - colour.r) : (colour.r - 255u));
//  colour.g = static_cast<uint8_t>((255u - colour.g) > 0 ? (255u - colour.g) : (colour.g - 255u));
//  colour.b = static_cast<uint8_t>((255u - colour.b) > 0 ? (255u - colour.b) : (colour.b - 255u));
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

uint32_t MapSDLRGBA(RGBA colour,  pxFmt format) [[hc]]
{
    return    (colour.r >> format.Rloss) << format.Rshift
            | (colour.g >> format.Gloss) << format.Gshift
            | (colour.b >> format.Bloss) << format.Bshift
            | ((colour.a >> format.Aloss) << format.Ashift & format.Amask  );
}

void calculateIterations(hc::array_view<uint32_t,1> data,
                         int width,
                         int height,
                         Region r,
                         pxFmt format,
                         hc::index<1> idx) [[hc]]
{
    int row = idx[0]/width;
    int col = idx[0]%width;
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
    data[idx] = MapSDLRGBA(getColour(iteration), format);
}

void FracGen::Generate(uint32_t* v, SDL_PixelFormat* format, int width, int height, Region r)
{
    if(format == nullptr)
    {
        return;
    }
    hc::array_view<uint32_t, 1>av(width*height, v);
    hc::parallel_for_each(hc::extent<1>(width*height),
                          [=, fmt = pxFmt(*format)]
                          (hc::index<1> i) [[hc]]
                          {calculateIterations(av, width, height, r, fmt, i);} );
}

FracGen::FracGen(bool benching)
{
    static bool once = false;

    auto devices = hc::accelerator::get_all();
    for (auto dev : devices)
    {
        if(!once || !benching )
        {
            std::cout << "Found HC device: " << dev.get_description().c_str() << std::endl;
            std::cout << "At: " << dev.get_device_path().c_str() << std::endl;
            std::cout << "Display: " << dev.get_has_display() << std::endl;
            std::cout << "Is HSA: " << dev.is_hsa_accelerator() << std::endl;
            std::cout << "Version: " << dev.get_version() << std::endl;
            once = true;
        }
        if(dev.is_hsa_accelerator())
        {
            hc::accelerator::set_default(dev.get_device_path());
        }
    }
}

FracGen::~FracGen()
{}

bool operator==(const Region &r1, const Region &r2)
{
    return (   (r1.Imax - r2.Imax <= DBL_EPSILON) && (r1.Imin - r2.Imin <= DBL_EPSILON)
            && (r1.Rmax - r2.Rmax <= DBL_EPSILON) && (r1.Rmin - r2.Rmin <= DBL_EPSILON) );
}
