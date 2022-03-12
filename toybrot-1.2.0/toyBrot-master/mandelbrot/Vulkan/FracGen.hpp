#ifndef FRACGEN_HPP_DEFINED
#define FRACGEN_HPP_DEFINED

#include <vector>
#include <cstdint>
#include <SDL_pixels.h>



struct Region{double Imin,Imax,Rmin,Rmax;}; //This struct delimits a region in the Argand-Gauss Plane (R X I)

bool operator==(const Region& r1, const Region& r2);

class FracGen
{
public:

    FracGen(bool bench = false);
    ~FracGen();

    void Generate(uint32_t* img, SDL_PixelFormat* format, int width, int height, Region r);


private:


};

#endif //FRACGEN_HPP_DEFINED
