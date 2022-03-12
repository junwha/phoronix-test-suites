#include "pngWriter.hpp"

#include <string>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <thread>
#include <future>
#include <limits>

#ifdef __EMSCRIPTEN__
    #include <emscripten.h>
#endif


pngWriter::pngWriter(CameraPtr c, ParamPtr p, std::string outname)
    : cam{c}
    , params{p}
    , bit_depth{static_cast<png_byte>(sizeof(pngColourType)*4)}
    , writePtr{nullptr}
    , infoPtr{nullptr}
    , fractal{nullptr}
    , outData{}
    , filename{outname.empty() ?"toybrotOut.png" : outname }
    , output{nullptr}
{
    auto pos = filename.find(".");
    strippedName = filename.substr(0, pos);
    extensionName = filename.substr(pos, filename.size());
}

void pngWriter::setFractal(FracPtr f) noexcept
{
    fractal = f;
    outData.resize(cam->ScreenHeight());
    for(auto& row: outData)
    {
        row.resize(cam->ScreenWidth());
    }
}

bool pngWriter::Init()
{
    size_t tweak = 0;
    if(std::ifstream(filename))
    {
        do
        {
            filename = strippedName + nameTweak + extensionName;
            nameTweak = "_" + std::to_string(tweak++);
        }
        while(std::ifstream(filename));
    }

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

    // Write header (16 bit colour depth)
    png_set_IHDR(writePtr, infoPtr, cam->ScreenWidth(), cam->ScreenHeight(),
    16, PNG_COLOR_TYPE_RGB_ALPHA, PNG_INTERLACE_NONE,
    PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    std::string title = "Toybrot Mandelbox";
    char ctitle[256];
    for(char& c: ctitle)
    {
        c = 0;
    }
    title.copy(ctitle,title.length(),0);
    char titleKey[] = "Title";
    png_text title_text;
    title_text.compression = PNG_TEXT_COMPRESSION_NONE;
    title_text.key = titleKey;
    title_text.text = ctitle;
    title_text.text_length = title.size();
    png_set_text(writePtr, infoPtr, &title_text, 1);


    png_write_info(writePtr, infoPtr);

    return 0;
}


bool pngWriter::Write()
{
    if(fractal == nullptr)
    {
        return false;
    }
    // Write image data

    auto convertPixel = [&](size_t idx)
            {
                #ifndef TB_SINGLETHREAD
                    size_t numTasks = std::thread::hardware_concurrency();
                #else
                    constexpr const size_t numTasks = 1;
                #endif
                for (size_t i = idx; i < this->cam->ScreenWidth()*this->cam->ScreenHeight(); i+= numTasks)
                {
                    size_t row = i / cam->ScreenWidth();
                    size_t col = i % cam->ScreenWidth();
                    png_save_uint_16(reinterpret_cast<png_bytep>(&outData[row][col].r), (*fractal)[row*cam->ScreenWidth() + col].R()*std::numeric_limits<uint16_t>::max());
                    png_save_uint_16(reinterpret_cast<png_bytep>(&outData[row][col].g), (*fractal)[row*cam->ScreenWidth() + col].G()*std::numeric_limits<uint16_t>::max());
                    png_save_uint_16(reinterpret_cast<png_bytep>(&outData[row][col].b), (*fractal)[row*cam->ScreenWidth() + col].B()*std::numeric_limits<uint16_t>::max());
                    png_save_uint_16(reinterpret_cast<png_bytep>(&outData[row][col].a), (*fractal)[row*cam->ScreenWidth() + col].A()*std::numeric_limits<uint16_t>::max());
                }
            };
    #ifndef TB_SINGLETHREAD
        std::vector<std::future<void>> tasks (std::thread::hardware_concurrency());
        // "classic" for loop because we need to know the idx here
        for(size_t idx = 0; idx < std::thread::hardware_concurrency(); idx++)
        {
            tasks[idx] = std::async(convertPixel, idx);
        }
        for( auto& t : tasks)
        {
            t.get();
        }
    #else

        convertPixel(0);

    #endif
    return Write(outData);
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
    fclose(output);

    #ifdef __EMSCRIPTEN__
        std::string callScript ("window.offerFileAsDownload(\"" + filename +"\", \"image/png\")");
        emscripten_run_script(callScript.c_str());
    #else
        std::cout << "Wrote "<< filename << std::endl;
    #endif
    return true;
}

void pngWriter::Alloc(pngData& data)
{
    data.resize(cam->ScreenWidth()*cam->ScreenHeight());
}

void pngWriter::Alloc(pngData2d& rows)
{
    rows.resize(cam->ScreenHeight());
    for(auto& v: rows)
    {
        v.resize(cam->ScreenWidth());
    }
}


