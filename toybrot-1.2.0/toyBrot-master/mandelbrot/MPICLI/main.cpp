
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
#include <cmath>
#include <cstdint>

#include <mpi.h>
#include "pngWriter.hpp"



struct region{long double Imin,Imax,Rmin,Rmax;}; //This struct delimits a region in the Argand-Gauss Plane (R X I)
bool operator==(const region& r1, const region& r2){return ( (r1.Imax - r2.Imax <= LDBL_EPSILON) && (r1.Imin - r2.Imin <= LDBL_EPSILON)
														  && (r1.Rmax - r2.Rmax <= LDBL_EPSILON) && (r1.Rmin - r2.Rmin <= LDBL_EPSILON) );}

bool endProgram;
unsigned int iteration_factor = 100;
unsigned int max_iteration = 256 * iteration_factor;
long double Bailout = 2;
long double power = 2;
int lastI = 0;
bool ColourScheme = false;
auto genTime (std::chrono::high_resolution_clock::now());

int myrank = 1;
int nprocs = 1;
size_t numDivs = 3;
int realDivs = 0;
int imDivs = 0;



png_bytep row = nullptr;
//std::array<std::future<bool>, numDivs> tasks;
std::vector<std::future<bool>> tasks(numDivs);


pngRGB getColour(unsigned int it, unsigned int rank) noexcept
{
    pngRGB colour;

	if (ColourScheme)
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
    return colour;
}


auto fracGen = [](region r, uint32_t width, uint32_t height, int rank, int numTasks, size_t index, pngData* pixels) noexcept
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

            pixels->at((i*width)+j) = getColour(iteration, rank);
		}
	}
	return false;
};

void spawnTasks(region reg, uint32_t width, uint32_t height, int rank, int procs, pngData& pixels) noexcept
{
//    std::cout << "Task " << myrank << " of " << nprocs << " drawing region: ";
//    std::cout << myReg.Imin << "i -> " << myReg.Imax << "i // " << myReg.Rmin << " -> " << myReg.Rmax << std::endl;

    for(unsigned int i = 0; i < tasks.size(); i++)
	{
        tasks[i] = std::async(std::launch::async, fracGen, reg, width, height, rank, procs, i, &pixels);
	}

    for(unsigned int i = 0; i < tasks.size(); i++)
	{
        //block until all tasks are done
        tasks[i].get();
	}

}

region defineRegion(region r, int rank, int procs)
{
    long double ImLength = r.Imax - r.Imin;
    long double ImStep = ImLength/nprocs;
    region myReg = r;
    //bit wonky due to images being drawn top to bottom
    myReg.Imax = myReg.Imax - (ImStep*rank);
    myReg.Imin = myReg.Imax - ImStep;
    return myReg;
}

int main (int argc, char** argv) noexcept
{

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);


    auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - genTime).count();


    if(nprocs == 0 )
    {
        myrank = 0;
        nprocs = 1;
    }

//    int closestSqrt = nprocs;
//    int r = static_cast<int>(sqrt(closestSqrt));
//    while( (r*r) != closestSqrt )
//    {
//        closestSqrt -= 1;
//        r = static_cast<int>(sqrt(closestSqrt));
//    }

    uint32_t height = 1920 * 6;
    uint32_t width  = 4096 * 6;
//    uint32_t height = 720 * 1;
//    uint32_t width  = 1024 * 1;



    pngData pngRows;
    pngData workRows;



    pngWriter writer(0,0);

    //root process behaviour
    if(myrank == 0)
    {
        writer.setHeight(height);
        writer.setWidth(width);
        writer.Init();
        writer.Alloc(pngRows);


    }

    workRows.resize(width*(height/static_cast<unsigned int>(nprocs)));


    region reg {-1.5l,1.5l,-2,1};



    MPI_Bcast(  reinterpret_cast<void*>(&reg),
                4,
                MPI_LONG_DOUBLE,
                0,
                MPI_COMM_WORLD
                );

    int res = 0;
    std::ofstream outlog;


    region myReg = defineRegion(reg, myrank, nprocs);


    spawnTasks(myReg, width, height/static_cast<unsigned int>(nprocs), myrank, nprocs, workRows);

    MPI_Gather(reinterpret_cast<void*>(workRows.data()),
               static_cast<int>(workRows.size()*3),
               MPI_UINT16_T,
               reinterpret_cast<void*>(pngRows.data()),
               static_cast<int>(workRows.size()*3),
               MPI_UINT16_T,
               0,
               MPI_COMM_WORLD
               );



    if(myrank == 0)
    {

        auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - genTime).count();
        std::cout << "Generating fractal of " << width << "x" << height << " using " << nprocs << " workers took ";
        std::cout << (end-start)/1000 << "." << (end-start)%1000 << " seconds" << std::endl;

        writer.Write(pngRows);
    }


    //defineRegion(myrank,nprocs);






    if(outlog.is_open())
    {
        outlog.flush();
        outlog.close();
    }

    if(res == 0)
    {
        writer.Write(pngRows);
    }

    MPI_Finalize();

    return res;
}

