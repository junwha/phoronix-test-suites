#define NOMINMAX

#include <algorithm>
#include <assert.h>
#include <sstream>
#include <array>
#include <future>
#include <chrono>
#include <cfloat>
#include <numeric>
#include <vector>
#include <iostream>
#include <fstream>
#include <thread>

/*
 * Some preprocessor shenanigans to make both
 * the SDL2 and libPNG dependencies optional
 */
#ifdef TOYBROT_ENABLE_GUI
    #include "FracGenWindow.hpp"
#else
    class FracGenWindow;
#endif
#ifdef TOYBROT_ENABLE_PNG
    #include "pngWriter.hpp"
#else
    class pngWriter;
#endif
#include "FracGen.hpp"

#ifdef TOYBROT_HIPSYCL_FLAVOUR
    static std::string flavourName = TOYBROT_HIPSYCL_FLAVOUR ;
#else
    static std::string flavourName = "SYCL";
#endif



static auto genTime (std::chrono::high_resolution_clock::now());
static constexpr const int defaultWidth  = 1820;
static constexpr const int defaultHeight = 980;
static int windowWidth = defaultWidth;
static int windowHeight = defaultHeight;
static std::vector<size_t> results;
static size_t setupTime;
static size_t lastRunTime;
static size_t iterations = 1;
static bool forceHeadless = false;
static bool pngExport = false;

int runProgram(bool benching) noexcept
{
    std::shared_ptr<bool> redraw(new bool(true));
    std::shared_ptr<bool> exit(new bool(false));

    // I need to use raw pointers here because of the
    // optional dependency shenanigans
    FracGenWindow* mainWindow = nullptr;
    pngWriter* exportWriter = nullptr;

    #ifdef TOYBROT_ENABLE_GUI
    if(!forceHeadless)
    {
        mainWindow = new FracGenWindow(windowWidth,windowHeight, flavourName, redraw, exit);
    }
    #endif

    genTime = std::chrono::high_resolution_clock::now();

    FracGen gpuFrac(benching, windowWidth, windowHeight);

    setupTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - genTime).count();

    #ifdef TOYBROT_ENABLE_PNG
    if(pngExport)
    {
        std::string exportName("toyBrot_");
        exportName += flavourName;
        exportName += ".png";
        exportWriter = new pngWriter(windowWidth, windowHeight, exportName);
        exportWriter->setFractal(gpuFrac.getBuffer());
        exportWriter->Init();
    }
    #endif


    std::stringstream stream;

    genTime = std::chrono::high_resolution_clock::now();
    bool clockReset = false;

    if(mainWindow != nullptr)
    {
    #ifdef TOYBROT_ENABLE_GUI
        std::thread eventCapture([&mainWindow, exit, benching](){while( !*exit ){ if(!benching){mainWindow->captureEvents();}}});
        mainWindow->setFractal(gpuFrac.getBuffer());

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
                gpuFrac.Generate(windowWidth, windowHeight);

                lastRunTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - genTime).count();

                *redraw = false;

                stream << flavourName << ": Fractal Generation Took " << lastRunTime << " milliseconds";

                results.push_back(lastRunTime);

                mainWindow->updateFractal();

                mainWindow->setTitle(stream.str());
                stream.str("");
                clockReset = true;
                if(benching)
                {
                    *exit = true;
                }
            }
            mainWindow->paint();
        }
        if(eventCapture.joinable())
        {
            eventCapture.join();
        }
    #endif
    }
    //headless version
    else
    {
        genTime = std::chrono::high_resolution_clock::now();

        gpuFrac.Generate(windowWidth, windowHeight);

        lastRunTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - genTime).count();
        stream << flavourName << ": Fractal Generation Took " << lastRunTime << " milliseconds";

        results.push_back(lastRunTime);
        if(iterations == 1)
        {
            std::cout << "Fractal Generation Took " << lastRunTime << " milliseconds" << std::endl;
        }
    }

    #ifdef TOYBROT_ENABLE_PNG
    if(pngExport)
    {
        exportWriter->Write();
    }
    #endif

    #ifdef TOYBROT_ENABLE_GUI
    if(mainWindow != nullptr)
    {
        delete mainWindow;
    }
    #endif
    #ifdef TOYBROT_ENABLE_PNG
    if(exportWriter != nullptr)
    {
        delete exportWriter;
    }
    #endif

    return 0;
}
void printUsage()
{
    std::vector<std::string> help
    {
        "Fracgen is a toy fractal generator you can use for silly CPU benchmarks",
        "And making some fancy mandelboxes",
        "",
        "Run from the cli for toy benchmarking",
        "Available options",
        "    -i X",
        "        Number of interactions to run",
        "    -o X",
        "        Output timingresults to a file",
        "    -d X Y",
        "        Define custom width and height. Default -> " + std::to_string(defaultWidth) + " x " + std::to_string(defaultHeight)
#ifdef TOYBROT_ENABLE_PNG
       ,"    -g",
        "        Disable gui"
#endif
#ifdef TOYBROT_ENABLE_PNG
       ,"    -p",
        "        Save image to a png"
#endif

    };
    for(std::string h: help)
    {
        std::cout << h << std::endl;
    }
}

int main (int argc, char** argv) noexcept
{
    int res = 0;
    std::ofstream outlog;

    enum class setting{NONE, ITERATIONS, JOBS, DIMENSIONS, HEADLESS, OUTPUT, EXPORT};
    uint8_t dimSet = 0;

    bool benching = false;
    if(argc > 1)
    {
        auto op = setting::NONE;
        for(int a = 1; a < argc; a++)
        {
            std::string token(argv[a]);
            if(token == "-i")
            {
                    op = setting::ITERATIONS;
                    benching = true;
                    continue;
            }
            if(token == "-o")
            {
                    op = setting::OUTPUT;
                    continue;
            }
            if(token == "-d")
            {
                    op = setting::DIMENSIONS;
            }
            #ifdef TOYBROT_ENABLE_GUI
            if(token == "-g")
            {
                    op = setting::HEADLESS;
            }
            #endif
            #ifdef TOYBROT_ENABLE_PNG
            if(token == "-p")
            {
                    op = setting::EXPORT;
            }
            #endif
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
                case setting::DIMENSIONS:
                    /*
                     * I COULD just advance through arguments here and do this
                     * in one go. Doing it in this way, however, makes it easier
                     * to catch the specific error case
                     */
                    if(dimSet == 0)
                    {
                        windowWidth = static_cast<size_t>(n);
                        dimSet = 1;
                        break;
                    }
                    else
                    {
                        if(dimSet == 1)
                        {
                            windowWidth = static_cast<size_t>(n);
                            dimSet = 2;
                            break;
                        }
                        else
                        {
                            if(dimSet == 2)
                            {
                                windowHeight = static_cast<size_t>(n);
                                dimSet = 0;
                                op = setting::NONE;
                                break;
                            }
                            else
                            {
                                std::cout << "Failure in setting Width and Height" << std::endl;
                                return 1;
                            }
                        }
                    }
                    break;
                case setting::HEADLESS:
                    forceHeadless = true;
                    op = setting::NONE;
                    break;
                case setting::EXPORT:
                    pngExport = true;
                    op = setting::NONE;
                    break;
                default:
                    break;
            }


        }
    }

    if(benching)
    {
        for(size_t i = 0; i < iterations; i++)
        {
            res = runProgram(true);
            if(res != 0)
            {
                return res;
            }
            std::cout << "Iteration " << i << " took " << lastRunTime << " milliseconds. Setup time: " << setupTime << "ms" << std::endl;
            if(outlog.is_open())
            {
                outlog << "Iteration " << i << " took " << lastRunTime << " milliseconds. Setup time: " << setupTime << "ms" << std::endl;
            }
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

