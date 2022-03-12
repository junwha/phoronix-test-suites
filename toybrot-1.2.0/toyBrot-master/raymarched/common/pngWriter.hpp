#ifndef PNGWRITER_H_DEFINED
#define PNGWRITER_H_DEFINED
#include <png.h>
#include <vector>
#include <cstdint>
#include <string>
#include "FracGen.hpp"

using pngColourType = png_uint_16;

struct pngRGBA
{
    pngColourType r;
    pngColourType g;
    pngColourType b;
    pngColourType a;
};

using pngData = std::vector<pngRGBA>;
using pngData2d = std::vector<pngData>;


class pngWriter
{
    public:

        pngWriter(CameraPtr c, ParamPtr p, std::string outname = "");

        void setFractal(FracPtr f) noexcept;

        bool Init();
        bool Write();

        const std::string FileName() const noexcept {return filename;}

    private:
        void Alloc(pngData& data);
        void Alloc(pngData2d& rows);
        bool Write(const pngData2d& rows);
        /**
         * Params currently unused but I plan on writing to
         * (and maybe reading from) png metadata
         */
        CameraPtr cam;
        ParamPtr  params;

        png_byte bit_depth;
        png_structp writePtr;
        png_infop infoPtr;

        FracPtr fractal;
        pngData2d outData;

        std::string filename;
        std::string strippedName;
        std::string extensionName;
        std::string nameTweak;
        FILE* output;

};

#endif //PNGWRITER_H_DEFINED
