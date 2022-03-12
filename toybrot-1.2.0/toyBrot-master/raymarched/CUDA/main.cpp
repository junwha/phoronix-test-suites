#define NOMINMAX

#include <sstream>
#include <chrono>
#include <cfloat>
#include <numeric>
#include <vector>
#include <iostream>
#include <fstream>
#include <thread>

#include "defines.hpp"
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

#if TOYBROT_CUDA_CLANG
    #if TOYBROT_USE_DOUBLES
        static std::string flavourName = "CUDA-CLANG-db";
        static std::string fileName    = "cuda-clang-db.c";
    #else
        static std::string flavourName = "CUDA-CLANG";
        static std::string fileName    = "cuda-clang.c";
    #endif
#else
    #if TOYBROT_USE_DOUBLES
        static std::string flavourName = "CUDA-NVCC-db";
        static std::string fileName    = "cuda-nvcc-db.c";
    #else
        static std::string flavourName = "CUDA-NVCC";
        static std::string fileName    = "cuda-nvcc.c";
    #endif
#endif


static auto genTime (std::chrono::high_resolution_clock::now());
static uint32_t windowWidth = 0;
static uint32_t windowHeight = 0;
static std::vector<size_t> results;
static size_t setupTime;
static size_t lastRunTime;
static size_t iterations = 1;
static bool forceHeadless = false;
static bool pngExport = false;
static std::shared_ptr<Camera<tbFPType>> camera;
static std::shared_ptr<Parameters<tbFPType>> params;

/******************************************************************************
 *
 * Default parameters
 *
 ******************************************************************************/

// Camera parameters
static constexpr const tbFPType cameraX         =  0.0;
static constexpr const tbFPType cameraY         =  0.0;
static constexpr const tbFPType cameraZ         = -3.8;
static constexpr const tbFPType targetX         =  0.0;
static constexpr const tbFPType targetY         =  0.0;
static constexpr const tbFPType targetZ         =  0.0;
static constexpr const uint32_t screenWidth     =  1820;
static constexpr const uint32_t screenHeight    =  980;
static constexpr const tbFPType fovY            =  45;
static constexpr const tbFPType near            =  0.1;

// Coulouring parameters

static constexpr const tbFPType hueFactor   = -60.0;
static constexpr const int32_t  hueOffset   = 325;
static constexpr const tbFPType valueFactor = 32;
static constexpr const tbFPType valueRange  = 1.0;
static constexpr const tbFPType valueClamp  = 0.9;
static constexpr const tbFPType satValue    = 0.8;
static constexpr const tbFPType bgRed       = 0.05;
static constexpr const tbFPType bgGreen     = 0.05;
static constexpr const tbFPType bgBlue      = 0.05;
static constexpr const tbFPType bgAlpha     = 1.0;

// Raymarching parameters

static constexpr const uint32_t maxRaySteps      = 7500;
static constexpr const tbFPType collisionMinDist = 0.00055;

// Mandelbox parameters
static constexpr const tbFPType fixedRadiusSq = 2.2;
static constexpr const tbFPType minRadiusSq   = 0.8;
static constexpr const tbFPType foldingLimit  = 1.45;
static constexpr const tbFPType boxScale      = -3.5;
static constexpr const uint32_t boxIterations = 30;


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
        mainWindow = new FracGenWindow(camera, flavourName, redraw, exit);
    }
    #endif

    genTime = std::chrono::high_resolution_clock::now();

    FracGen gpuFrac(benching, camera, params);

    setupTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - genTime).count();

    #ifdef TOYBROT_ENABLE_PNG
    if(pngExport)
    {
        std::string exportName("toyBrot_");
        exportName += flavourName;
        exportName += ".png";
        exportWriter = new pngWriter(camera, params, exportName);
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
                gpuFrac.Generate();

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

        gpuFrac.Generate();

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
        "        Define custom width and height. Default -> " + std::to_string(screenWidth) + " x " + std::to_string(screenHeight)
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

    camera = std::make_shared<Camera<tbFPType> >( Vec3<tbFPType>{cameraX, cameraY, cameraZ}
                                                , Vec3<tbFPType>{targetX,targetY,targetZ}
                                                , screenWidth, screenHeight, near, fovY );
    params = std::make_shared< Parameters<tbFPType> >( maxRaySteps, collisionMinDist, hueFactor, hueOffset
                                                  , valueFactor, valueRange, valueClamp, satValue
                                                  , bgRed, bgGreen, bgBlue, bgAlpha, fixedRadiusSq
                                                  , minRadiusSq, foldingLimit, boxScale, boxIterations  );
#ifdef TOYBROT_CUDA_CLANG
    params->setBgValue(0.8f,0.8f,0.8f,1.0f);
#endif

    toyBrot::loadConfig(camera, params, fileName);

    //override config file dimensions with values from CLI
    if( windowWidth != 0)
    {
        camera->setScreenWidth(windowWidth);
    }

    if(windowHeight != 0)
    {
        camera->setScreenHeight(windowHeight);
    }

    if(windowWidth != 0 || windowHeight != 0)
    {
        camera->updateScreen();
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

