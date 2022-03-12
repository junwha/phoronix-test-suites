#define NOMINMAX

#include <algorithm>
#include <sstream>
#include <chrono>
#include <cfloat>
#include <numeric>
#include <iostream>
#include <fstream>
#include <thread>
#include "FracGenWindow.hpp"
#include "FracGen.hpp"


static std::shared_ptr<int> ColourScheme(new int(0));
static auto genTime (std::chrono::high_resolution_clock::now());
static int windowWidth = 1280;
static int windowHeight = 720;

int runProgram(bool benching) noexcept
{

    std::shared_ptr<Region> reg(new Region());

    reg->Imax = 1.5;
    reg->Imin = -1.5;
    reg->Rmax = 1;
    reg->Rmin = -2;

    std::shared_ptr<bool> redraw(new bool(true));
    std::shared_ptr<bool> exit(new bool(false));

    FracGenWindow mainWindow(windowWidth,windowHeight, redraw, exit);

    std::thread eventCapture([&mainWindow, exit, benching](){while( !*exit ){ if(!benching){mainWindow.captureEvents();}}});

    FracGen gpuFrac(benching);
    mainWindow.registerColourFlag(ColourScheme);
    mainWindow.setRegion(reg);

    SurfPtr frac = mainWindow.getFrac();
    std::stringstream stream;

    genTime = std::chrono::high_resolution_clock::now();
    bool clockReset = false;

    eventCapture.detach();
    while(!*exit)
    {
        if(*redraw)
        {
            if(clockReset)
            {
                clockReset = false;
                genTime = std::chrono::high_resolution_clock::now();
            }
            SDL_LockSurface(frac.get());

            gpuFrac.Generate(reinterpret_cast<uint32_t*>(frac->pixels), frac->format, windowWidth, windowHeight, *reg.get());

            SDL_UnlockSurface(frac.get());

            *redraw = false;
            auto d = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - genTime).count();

            stream << "Fractal Generation Took " << d << " milliseconds";
            mainWindow.setTitle(stream.str());
            stream.str("");
            clockReset = true;
            if(benching)
            {
                *exit = true;
            }
        }
        mainWindow.paint();
	}
    if(eventCapture.joinable())
    {
        eventCapture.join();
    }
    return 0;
}

void printUsage()
{
    std::vector<std::string> help
    {
        "Fracgen is a toy mandelbrot fractal generator you can use for silly benchmarks",
        "If you just want to look at some fractals, just run it plain",
        "Drag boxes with Mouse1 to select region of interest, Mouse2 switches colour scheme",
        "Mouse 3 resets the image to the original area",
        "",
        "Run from the cli for toy benchmarking",
        "Available options",
        "    -i X",
        "        Number of interactions to run",
        "    -o X",
        "        Output results to a file"

    };
    for(std::string h: help)
    {
        std::cout << h << std::endl;
    }
}

int main (int argc, char** argv) noexcept
{
    int res = 0;
    size_t iterations = 1;
    std::ofstream outlog;

    enum class setting{NONE, ITERATIONS, OUTPUT};

    if(argc > 1)
    {
        auto op = setting::NONE;
        for(int a = 1; a < argc; a++)
        {
            std::string token(argv[a]);
            if(token == "-i")
            {
                    op = setting::ITERATIONS;
                    continue;
            }
            if(token == "-o")
            {
                    op = setting::OUTPUT;
                    continue;
            }
            if((token == "-h") || (token == "--h"))
            {
                printUsage();
                return 0;
            }

            //No exceptions here, only undefined behaviour
            int n = atoi(argv[a]);
            switch(op)
            {
                case setting::ITERATIONS:
                    iterations = static_cast<size_t>(n);
                    op = setting::NONE;
                    break;
                case setting::OUTPUT:
                    outlog.open(argv[a]);
                    if(outlog.fail())
                    {
                        std::cout << "Could not open file " << argv[a] << " for output";
                        return 1;
                    }
                    op = setting::NONE;
                    break;
                default:
                    break;
            }
        }
    }
    if(iterations > 1)
    {
        std::vector<size_t> results;
        for(size_t i = 0; i < iterations; i++)
        {
            res = runProgram(true);
            if(res != 0)
            {
                return res;
            }
            size_t d = static_cast<size_t>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - genTime).count());
            std::cout << "Iteration " << i << " took " << d << " milliseconds" << std::endl;
            if(outlog.is_open())
            {
                outlog << "Iteration " << i << " took " << d << " milliseconds" << std::endl;
            }
            results.push_back(d);
        }
        auto avg = std::accumulate(results.begin(), results.end(), 0u)/ results.size();

        std::cout << std::endl << "Average time of " << avg << " milliseconds (over " << results.size()<< " tests)"<< std::endl;
        if(outlog.is_open())
        {
            outlog << std::endl << "Average time of " << avg << " milliseconds (over " << results.size()<< " tests)"<< std::endl;
        }
    }
    else
    {
        res = runProgram(false);
    }

    if(outlog.is_open())
    {
        outlog.flush();
        outlog.close();
    }

    return res;
}

