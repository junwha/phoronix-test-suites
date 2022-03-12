
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

#include <mpi.h>
#include "FracGenWindow.hpp"



unsigned int iteration_factor = 100;
unsigned int max_iteration = 256 * iteration_factor;
long double Bailout = 2;
long double power = 2;

std::shared_ptr<int> ColourScheme(new int(0));
auto genTime (std::chrono::high_resolution_clock::now());

size_t numDivs = std::thread::hardware_concurrency() * 4;

//std::array<std::future<bool>, numDivs> tasks;
std::vector<std::future<bool>> tasks(numDivs);


uint32_t getColour(unsigned int it, SDL_PixelFormat* format, unsigned int rank) noexcept
{
    RGB colour;

    if (!ColourScheme)
    {
        colour.r = 128 + std::sin((float)it + 1)*128;
        colour.g = 128 + std::sin((float)it)*128;
        colour.b = std::cos((float)it+1.5)*255;
    }
    else
    {
        if(it == max_iteration)
        {
            colour.r = 0;
            colour.g = 0;
            colour.b = 0;
        }
        else
        {
            colour.r = std::min(it,255u);
            colour.g = std::min(it,255u);
            colour.b = std::min(it,255u);

            //let's make it a bit fun
            switch (rank % 7)
            {
                case 0: colour.r = 0;                             break;
                case 1:               colour.g = 0;               break;
                case 2:                             colour.b = 0; break;
                case 3:               colour.g = 0; colour.b = 0; break;
                case 4: colour.r = 0;               colour.b = 0; break;
                case 5: colour.r = 0; colour.g = 0;               break;
                case 6: break;
            }
        }
    }

    return SDL_MapRGB(format, colour.r, colour.g, colour.b);
}


auto fracGen = [](Region r, uint32_t width, uint32_t height, int rank, int numTasks, size_t index, SDL_PixelFormat* format, PixelData* pixels) noexcept
{
    if(pixels == nullptr)
    {
        return false;
    }

    long double incX = std::abs((r.Rmax - r.Rmin)/width);
    long double incY = std::abs((r.Imax - r.Imin)/height);
    long double offsetY = incY * rank;
    int rowStep = height/numDivs;
    int rowZero =  rowStep * index;

    for(int i = rowZero; i < rowZero+rowStep; i++)
    {
        if(i > height)
        {
            return true;
        }

        for(int j = 0; j < width; j++)
        {

            long double x = r.Rmin+(j*incX);
            long double y = (r.Imax-(i*incY));
            long double x0 = x;
            long double y0 = y;

            unsigned int iteration = 0;

            while ( (x*x + y*y <= 4)  &&  (iteration < max_iteration) )
            {
                long double xtemp = x*x - y*y + x0;
                y = 2*x*y + y0;

                x = xtemp;

                iteration++;
            }

            pixels->at((i*width)+j) = getColour(iteration, format, rank);
        }
    }
    return false;
};

void spawnTasks(Region reg, uint32_t width, uint32_t height, int rank, int procs, SDL_PixelFormat* format, PixelData& pixels) noexcept
{
//    std::cout << "Task " << myrank << " of " << nprocs << " drawing region: ";
//    std::cout << myReg.Imin << "i -> " << myReg.Imax << "i // " << myReg.Rmin << " -> " << myReg.Rmax << std::endl;

    for(unsigned int i = 0; i < tasks.size(); i++)
    {
        tasks[i] = std::async(std::launch::async, fracGen, reg, width, height, rank, procs, i, format, &pixels);
    }

    for(unsigned int i = 0; i < tasks.size(); i++)
    {
        //block until all tasks are done
        tasks[i].get();
    }

}

Region defineRegion(Region r, int rank, int procs)
{
    long double ImLength = r.Imax - r.Imin;
    long double ImStep = ImLength/procs;
    Region myReg = r;
    //bit wonky due to images being drawn top to bottom
    myReg.Imax = myReg.Imax - (ImStep*rank);
    myReg.Imin = myReg.Imax - ImStep;
    return myReg;
}


int main (int argc, char** argv) noexcept
{
    int nprocs, myrank;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);


    auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - genTime).count();


    if(nprocs == 0 )
    {
        myrank = 0;
        nprocs = 1;
    }


    bool keepRunning = true;


    std::shared_ptr<Region> reg(new Region());

    reg->Imax = 1.5;
    reg->Imin = -1.5;
    reg->Rmax = 1;
    reg->Rmin = -2;

    std::shared_ptr<bool> redraw(new bool);
    *redraw = true;

    uint32_t width  = 1280;
    uint32_t height = 720;
//    uint32_t height = 320;
//    uint32_t width  = 480;

    int res = 0;
    std::ofstream outlog;


    Region myReg;


    std::unique_ptr<FracGenWindow> mainWindow;
    SurfPtr frac;
    PixelData workRows;
    workRows.resize(width*(height/nprocs),0u);
    SDL_PixelFormat format;
    std::cout <<"Process "<< myrank << " preparing to broadcast format" << std::endl;

    if(myrank == 0)
    {
        mainWindow = std::make_unique<FracGenWindow>(width,height,32, redraw);
        mainWindow->registerColourFlag(ColourScheme);
        mainWindow->setRegion(reg);
        frac = mainWindow->getFrac();
        format = *(frac->format);
    }
    MPI_Bcast( reinterpret_cast<void*>(&format),
               sizeof(SDL_PixelFormat),
               MPI_BYTE,
               0,
               MPI_COMM_WORLD
               );
    std::stringstream stream;

    std::cout <<"Process "<< myrank << " initialized format" << std::endl;
    genTime = std::chrono::high_resolution_clock::now();
    bool clockReset = false;

    while(keepRunning)
    {
        if(*redraw)
        {

            MPI_Bcast(  reinterpret_cast<void*>(reg.get()),
                        4,
                        MPI_LONG_DOUBLE,
                        0,
                        MPI_COMM_WORLD
                        );

            myReg = defineRegion(*reg, myrank, nprocs);

            std::cout <<"Process "<< myrank << " initialized region" << std::endl;


            spawnTasks(myReg, width, height/static_cast<unsigned int>(nprocs), myrank, nprocs, &format, workRows);

            std::cout <<"Process "<< myrank << " completed tasks" << std::endl;


            MPI_Gather(reinterpret_cast<void*>(workRows.data()),
                       static_cast<int>(workRows.size()),
                       MPI_UINT32_T,
                       frac->pixels,
                       static_cast<int>(workRows.size()),
                       MPI_UINT32_T,
                       0,
                       MPI_COMM_WORLD
                       );

            std::cout <<"Process "<< myrank << " completed gather" << std::endl;


            if(myrank == 0)
            {




                *redraw = false;
                auto d = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - genTime).count();

                genTime = std::chrono::high_resolution_clock::now();

                stream << "Fractal Generation Took " << d << " milliseconds";
                mainWindow->setTitle(stream.str());
                stream.str("");
                clockReset = true;
            }

        }


        if(myrank == 0)
        {
            keepRunning = mainWindow->captureEvents();
            mainWindow->paint();
        }

    }


    if(outlog.is_open())
    {
        outlog.flush();
        outlog.close();
    }


    MPI_Finalize();

    return res;
}

