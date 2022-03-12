#ifndef PNGWRITER_H_DEFINED
#define PNGWRITER_H_DEFINED
#include <png.h>
#include <vector>
#include <cstdint>

using pngColourType = png_uint_16;

struct pngRGB
{
    pngColourType r;
    pngColourType g;
    pngColourType b;
};

using pngData = std::vector<pngRGB>;
using pngData2d = std::vector<pngData>;


class pngWriter
{
    public:

        pngWriter(uint32_t width = 0, uint32_t height = 0);

        void setWidth(unsigned int width)   noexcept {w = width;}
        void setHeight(unsigned int height) noexcept {h = height;}

        bool Init();
        bool Write(const pngData& data);
        bool Write(const pngData2d& rows);
        void Alloc(pngData& data);
        void Alloc(pngData2d& rows);

    private:

        uint32_t w, h;

        png_byte bit_depth;
        png_structp writePtr;
        png_infop infoPtr;

        FILE* output;

};

#endif //PNGWRITER_H_DEFINED
