
#define NOMINMAX

#ifdef WIN32
	#include <Windows.h>
	#include "SDL.h"
	#include "SDL_opengl.h"
#else

#include "SDL2/SDL.h"
#include "SDL2/SDL_opengl.h"

#endif


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


struct region{long double Imin,Imax,Rmin,Rmax;}; //This struct delimits a region in the Argand-Gauss Plane (R X I)
bool operator==(const region& r1, const region& r2){return ( (r1.Imax - r2.Imax <= LDBL_EPSILON) && (r1.Imin - r2.Imin <= LDBL_EPSILON)
														  && (r1.Rmax - r2.Rmax <= LDBL_EPSILON) && (r1.Rmin - r2.Rmin <= LDBL_EPSILON) );}

region reg;
SDL_Window* mainwindow;
SDL_Renderer* render;
SDL_Surface* screen;
SDL_Texture* texture;
SDL_Surface* frac;
SDL_Surface* highlight;
bool endProgram;
unsigned int iteration_factor = 100;
unsigned int max_iteration = 256 * iteration_factor;
long double Bailout = 2;
long double power = 2;
int w =	1280;
int h = 720;
int bpp = 32;
float aspect = (float)w/(float)h;
bool drawRect = false;
int rectX = 0;
int rectY = 0;
int MouseX = 0;
int MouseY = 0;
int lastI = 0;
bool creating = false;
bool ColourScheme = false;
auto genTime (std::chrono::high_resolution_clock::now());
std::stringstream stream;

size_t numDivs = 24;

//std::array<std::future<bool>, numDivs> tasks;
std::vector<std::future<bool>> tasks(numDivs);



void createHighlight() noexcept
{
    highlight = SDL_CreateRGBSurface(0,w,h,bpp,0,0,0,0);
	SDL_SetSurfaceBlendMode(highlight, SDL_BLENDMODE_BLEND);
	void* pix = highlight->pixels;
	for(int i = 0; i < frac->h; i++)
	{
		for(int j = 0; j< frac->w; j++)
		{
			
			Uint8* p = (Uint8*)pix + (i * highlight->pitch) + j*highlight->format->BytesPerPixel;
			*(Uint32*) p = SDL_MapRGB(frac->format, 255, 255, 255);
		}
	}
    SDL_SetSurfaceAlphaMod(highlight,128);
}

void DrawHL() noexcept
{
	SDL_Rect r;
	r.x = (rectX<MouseX?rectX:MouseX);
	r.y = (rectY<MouseY?rectY:MouseY);
	r.w = abs(MouseX - rectX);
	r.h = abs(MouseY - rectY);
	SDL_BlitSurface(highlight,&r,screen,&r);
}



Uint32 getColour(unsigned int it/*, double x*/) noexcept
{
	

	if (ColourScheme)
	{
        //GO GO DUMB MATHS!!
		//Aproximate range: From 0.3 to 1018 and then infinity (O.o)
//		long double index = it + (log(2*(log(Bailout))) - (log(log(std::abs(x)))))/log(power);
//		return SDL_MapRGB(frac->format, (sin(index))*255, (sin(index+50))*255, (sin(index+100))*255);
        return SDL_MapRGB(frac->format, 128+ sin((float)it + 1)*128, 128 + sin((float)it)*128 ,  cos((float)it+1.5)*255);
	}
	else
	{
        //std::cout<< it <<std::endl;

        //return SDL_MapRGB(frac->format, 128+ sin((float)it + 1)*128, 128 + sin((float)it)*128 ,  cos((float)it+1.5)*255);
        if(it == max_iteration)
        {
            return SDL_MapRGB(frac->format, 0, 0 , 0);
        }
        else
        {
			//auto b = std::min(it,255u);
            return SDL_MapRGB(frac->format, std::min(it,255u) , 0, std::min(it,255u) );
            //return SDL_MapRGB(frac->format, 128+ sin((float)it + 1)*128, 128 + sin((float)it)*128 ,  cos((float)it+1.5)*255);
        }
	}

}


void CreateFractal(region r) noexcept
{
	//this legacy unused function is here for reference
	if(creating == false)
	{
		lastI = 0;
		creating = true;
	}
	SDL_LockSurface(frac);
	Uint32* pix = (Uint32*)frac->pixels;
	long double incX = std::abs((r.Rmax - r.Rmin)/frac->w);
	long double incY = std::abs((r.Imax - r.Imin)/frac->h);
	for(int i = lastI; i < (lastI+10); i++)
	{
		if (i == frac->h)
		{
			break;
		}
		for(int j = 0; j< frac->w; j++)
		{
			
			Uint8* p = (Uint8*)pix + (i * frac->pitch) + j*frac->format->BytesPerPixel;//Don't ask u.u
			long double x = r.Rmin+(j*incX);
			long double y = r.Imax-(i*incY);
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

			*(Uint32*) p = getColour(iteration/*, x*/);
		}
	}
	lastI +=10;
	if (lastI == frac->h)
	{
		creating = false;
	}
	SDL_UnlockSurface(frac);
	
}

void paint() noexcept
{



	SDL_BlitSurface(frac,0,screen,0);
	if(drawRect)
	{
		DrawHL();
	}

	SDL_UpdateTexture(texture, NULL, screen->pixels, screen->pitch);
	SDL_RenderClear(render);
	SDL_RenderCopy(render, texture, NULL, NULL);
	SDL_RenderPresent(render);

}

auto fracGen = [](region r,int index, Uint32* pix, int h0) noexcept
{
    //std::cout << "tid: " << std::this_thread::get_id() << std::endl;

	//Uint32* pix = (Uint32*)frac->pixels;
	long double incX = std::abs((r.Rmax - r.Rmin)/frac->w);
	long double incY = std::abs((r.Imax - r.Imin)/frac->h);
	for(int i = h0;i < h0+10; i++)
	{
		if (i == frac->h)
		{
			return true;
		}
        //Initially intuitive/illustrative division
        //for(int j = (index%numDivs)*(frac->w/numDivs); j< ((index%numDivs)+1)*(frac->w/numDivs); j++)
        //Newer prefetcher-friendly version
        for(int j = 0 + index; j< frac->w; j+=numDivs)
		{

			Uint8* p = (Uint8*)pix + (i * frac->pitch) + j*frac->format->BytesPerPixel;//Set initial pixel
			long double x = r.Rmin+(j*incX);
			long double y = r.Imax-(i*incY);
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

			*(Uint32*) p = getColour(iteration/*, x*/);
		}
	}
	return false;
};

void spawnTasks(region reg, bool bench) noexcept
{
	creating = true;
	static std::atomic<int> h {0};

	SDL_LockSurface(frac);
    for(unsigned int i = 0; i < tasks.size(); i++)
	{
		tasks[i] = std::async(std::launch::async, fracGen,reg, i, /*tasks.size(),*/ (Uint32*)frac->pixels,h.load());
	}

    h+= 10;

    for(unsigned int i = 0; i < tasks.size(); i++)
	{
		if(tasks[i].get())
		{
			h.store(0);
			creating = false;
			auto d = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - genTime).count();

			stream << "Fractal Generation Took " << d << " milliseconds";
			SDL_SetWindowTitle(mainwindow, stream.str().c_str());
			stream.str("");
            if(bench)
            {
                endProgram = true;
            }
		}
	}
	SDL_UnlockSurface(frac);




}


void onKeyboardEvent(const SDL_KeyboardEvent& ) noexcept
{


}

void onMouseMotionEvent(const SDL_MouseMotionEvent& e) noexcept
{
	MouseX = e.x;
	MouseY = e.y;
	int rw = MouseX - rectX;
	int rh = MouseY - rectY;
	if (rh == 0)
	{
		return;
	}
	float ra = abs(rw/rh);
	if (ra == aspect)
	{
		return;
	}
	if (ra < aspect)
	{

		MouseX = rectX + rh*aspect;

	}
	else
	{
		MouseY = rectY + rw/aspect;
	}
}

void onMouseButtonEvent(const SDL_MouseButtonEvent& e ) noexcept
{
	if (creating)
	{
		//ignore
		return;
	}
	if (e.button == 4)
	{
		//M_WHEEL_UP
		//std::cout<< "button 4" <<std::endl;
		return;
	}
	if (e.button == 5)
	{
		//M_WHEEL_DOWN
		//std::cout<< "button 5" <<std::endl;
		return;
	}
	if (e.button == 2)
	{
		//Middle Button
		ColourScheme = !ColourScheme;
		//CreateFractal(reg);
        genTime = std::chrono::high_resolution_clock::now();
        spawnTasks(reg, false);
		return;
	}
	if(e.button == 3)
	{
		//Right Button
		reg.Imax = 1.5;
		reg.Imin = -1.5;
		reg.Rmax = 1;
		reg.Rmin = -2;
		//CreateFractal(reg);
        genTime = std::chrono::high_resolution_clock::now();
        spawnTasks(reg, false);
		return;
	}
	if(e.type == SDL_MOUSEBUTTONDOWN)
	{
		rectX = e.x;
		rectY = e.y;
		drawRect = true;
	}
	else
	{
		int rx = MouseX;
		int ry = MouseY;
//		int rw = std::abs(MouseX - rectX);
//		int rh = std::abs(MouseY - rectY);
		
	


		double x0 = reg.Rmin + ((reg.Rmax - reg.Rmin)/w) * rectX;
		double x1 = reg.Rmin + ((reg.Rmax - reg.Rmin)/w) * rx;
		
		double y0 = reg.Imax - ((reg.Imax - reg.Imin)/h) * rectY;
		double y1 = reg.Imax - ((reg.Imax - reg.Imin)/h) * ry;

		reg.Rmax = (x0>x1?x0:x1);
		reg.Rmin = (x0>x1?x1:x0);

		
		reg.Imax = (y0>y1?y0:y1);
		reg.Imin = (y0>y1?y1:y0);

		drawRect = false;
		//CreateFractal(reg);
        genTime = std::chrono::high_resolution_clock::now();
        spawnTasks(reg, false);
	}

}

void capture() noexcept
{
	SDL_Event event;
	while (SDL_PollEvent(&event))
	{
		switch (event.type)
		{
		case SDL_KEYDOWN:
		case SDL_KEYUP:
			onKeyboardEvent(event.key);
			break;

		case SDL_MOUSEBUTTONDOWN:
		case SDL_MOUSEBUTTONUP:
			onMouseButtonEvent(event.button);
			break;

		case SDL_QUIT:
			endProgram = true;
			break;

		case SDL_MOUSEMOTION:
			onMouseMotionEvent(event.motion);
			break;

		case SDL_JOYAXISMOTION:
		case SDL_JOYBUTTONDOWN:
		case SDL_JOYBUTTONUP:
		case SDL_JOYHATMOTION:
		case SDL_JOYBALLMOTION:
//		case SDL_ACTIVEEVENT:
//		case SDL_VIDEOEXPOSE:
//		case SDL_VIDEORESIZE:
			break;

		default:
			// Unexpected event type!
			//assert(0);
			break;
		}
	}
}




int runProgram(bool benching) noexcept
{
	endProgram = false;
	SDL_Init(SDL_INIT_EVERYTHING);
//	screen = SDL_SetVideoMode(w,h,bpp, SDL_HWSURFACE|SDL_DOUBLEBUF|SDL_ASYNCBLIT);
	mainwindow = SDL_CreateWindow("Mandelbrot Fractal Explorer - Use Mouse1 to zoom in and Mouse2 to zoom out. Press Mouse 3 to change colouring scheme",
							  SDL_WINDOWPOS_UNDEFINED,
							  SDL_WINDOWPOS_UNDEFINED,
							  w, h,
							  SDL_WINDOW_SHOWN);

	render = SDL_CreateRenderer(mainwindow, -1, 0);

	screen = SDL_CreateRGBSurface(0, w, h, bpp,
									0x00FF0000,
									0x0000FF00,
									0x000000FF,
									0xFF000000);

	texture = SDL_CreateTexture(render,
								SDL_PIXELFORMAT_ARGB8888,
								SDL_TEXTUREACCESS_STREAMING,
								w, h);
	assert(screen);
//	SDL_WM_SetCaption("Mandelbrot Fractal Explorer - Use Mouse1 to zoom in and Mouse2 to zoom out. Press Mouse 3 to change colouring scheme",0);
	SDL_SetWindowTitle(mainwindow, "Mandelbrot Fractal Explorer - Use Mouse1 to zoom in and Mouse2 to zoom out. Press Mouse 3 to change colouring scheme");

	//frac =	SDL_CreateRGBSurface(SDL_HWSURFACE|SDL_DOUBLEBUF|SDL_ASYNCBLIT,w,h,bpp,0,0,0,0);
	frac = SDL_CreateRGBSurface(0, w, h, bpp,
								0x00FF0000,
								0x0000FF00,
								0x000000FF,
								0xFF000000);

	
	reg.Imax = 1.5;
	reg.Imin = -1.5;
	reg.Rmax = 1;
	reg.Rmin = -2;


	//CreateFractal(reg);
    genTime = std::chrono::high_resolution_clock::now();
    spawnTasks(reg, benching);
	createHighlight();

	while(!endProgram)
	{
		if(creating)
		{
            spawnTasks(reg, benching);
		}

        if(!benching)
        {
            capture();
        }
		paint();
	}

	return 0;
}

void printUsage()
{
    std::vector<std::string> help
    {
        "Fracgen is a toy mandelbrot fractal generator you can use for silly CPU benchmarks",
        "If you just want to look at some fractals, just run it plain",
        "Drag boxes with Mouse1 to select region of interest, Mouse2 switches colour scheme",
        "Mouse 3 resets the image to the original area",
        "",
        "Run from the cli for toy benchmarking",
        "Available options",
        "    -i X",
        "        Number of interactions to run",
        "    -j X",
        "        Number of parallel tasks to run",
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
    auto numThreads = std::thread::hardware_concurrency();
    if(numThreads > 0)
    {
        // This 4 is an experimental factor. It feels like a sweet spot
        // I don't know why and never chased this but it's been true
        // across Bulldozer, Skylake and Threadripper
        numDivs=numThreads*4;
        tasks.resize(numDivs);
    }

    enum class setting{NONE, ITERATIONS, JOBS, OUTPUT};

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
            if(token == "-j")
            {
                    op = setting::JOBS;
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
                    iterations = n;
                    op = setting::NONE;
                    break;
                case setting::JOBS:
                    numDivs = n;
                    tasks.resize(numDivs);
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

        std::cout << "Preparing to run with "<< numDivs << " parallel tasks" << std::endl;
        if(outlog.is_open())
        {
            outlog << "Preparing to run with "<< numDivs << " parallel tasks" << std::endl;
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
            auto d = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - genTime).count();
            std::cout << "Iteration " << i << " took " << d << " milliseconds" << std::endl;
            if(outlog.is_open())
            {
                outlog << "Iteration " << i << " took " << d << " milliseconds" << std::endl;
            }
            results.push_back(d);
        }
        auto avg = std::accumulate(results.begin(), results.end(), 0)/ results.size();

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

