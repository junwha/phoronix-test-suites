#include "pngWriter.hpp"

#include <string>
#include <cstdio>
#include <iostream>

pngWriter::pngWriter(uint32_t width, uint32_t height)
    :w{width},
     h{height},
     bit_depth{32},
     writePtr{nullptr},
     infoPtr{nullptr},
     output{nullptr}
{}

bool pngWriter::Init()
{
    //std::string filename("/mnt/pandora/storage/users/jehferson/FracGenOut/FracGenMPI");
    //std::string filename("FracGenOut/FracGenMPI");
    std::string filename("FracGenMPI.png");

    output = fopen(filename.c_str(), "wb");
    if (output == nullptr)
    {
        std::cerr << "Could not open file " << filename << " for writing" << std::endl;
        return 1;
    }


    // Initialize PNG write structure
    writePtr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (writePtr == nullptr)
    {
        std::cerr << "Could not allocate PNG write struct" << std::endl;
        return 2;
    }

    // Initialize info structure
    infoPtr = png_create_info_struct(writePtr);
    if (infoPtr == nullptr)
    {
        std::cerr << "Could not allocate PNG info struct" << std::endl;
        return 2;
    }

    png_init_io(writePtr, output);

    // Write header (8 bit colour depth)
    png_set_IHDR(writePtr, infoPtr, w, h,
    16, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
    PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);


    std::string title = "FracGenMPI Mandelbrot";
    char ctitle[256];
    for(char& c: ctitle)
    {
        c = 0;
    }
    title.copy(ctitle,title.length(),0);

    png_text title_text;
    title_text.compression = PNG_TEXT_COMPRESSION_NONE;
    title_text.key = "Title";
    title_text.text = ctitle;
    title_text.text_length = title.size();
    png_set_text(writePtr, infoPtr, &title_text, 1);


    png_write_info(writePtr, infoPtr);

    return 0;
}

bool pngWriter::Write(const pngData& data)
{
    // Write image data
    for (unsigned int row = 0; row < h; row ++)
    {
       png_write_row(writePtr, reinterpret_cast<png_const_bytep>(data.data()+(row*w)) );
    }

    // End write
    png_write_end(writePtr, nullptr);
    //check for error here
    return true;
}

bool pngWriter::Write(const pngData2d& rows)
{
    // Write image data
    for (auto row : rows)
    {
       png_write_row(writePtr, reinterpret_cast<png_const_bytep>(row.data()) );
    }

    // End write
    png_write_end(writePtr, nullptr);
    //check for error here
    return true;
}

void pngWriter::Alloc(pngData& data)
{
    data.resize(h*w);
}

void pngWriter::Alloc(pngData2d& rows)
{
    rows.resize(h);
    for(auto& v: rows)
    {
        v.resize(w);
    }
}


