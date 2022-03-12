#ifndef FRACGEN_HPP_DEFINED
#define FRACGEN_HPP_DEFINED

#include <vector>
#include <cstdint>
#include <SDL_pixels.h>


struct Region{double Imin,Imax,Rmin,Rmax;}; //This struct delimits a region in the Argand-Gauss Plane (R X I)

bool operator==(const Region& r1, const Region& r2);

struct RGBA{uint8_t r = 0,g = 0,b = 0,a = 255;
            operator uint32_t() const;};

class FracGen
{
public:

    FracGen(bool benching = false);
    ~FracGen();

    bool Generate(uint32_t* img, SDL_PixelFormat* format, int width, int height, Region r);


private:

    uint32_t getColour();
    bool bench;
};

#endif //FRACGEN_HPP_DEFINED
