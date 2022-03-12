#include "FracGenWindow.hpp"


FracGenWindow::FracGenWindow(int w, int h, int bpp, std::shared_ptr<bool> redrawFlag)
    :width{w},
     height{h},
     aspectratio{h == 0 ? 0.f : w/h},
     colourDepth{bpp},
     drawRect{false},
     rectX{0},
     rectY{0},
     mouseX{0},
     mouseY{0},
     redrawRequired{redrawFlag}
{
    SDL_Init(SDL_INIT_EVERYTHING);
    mainwindow = SDL_CreateWindow("MPI Toybrot - Use Mouse1 to zoom in and Mouse2 to zoom out",
                              SDL_WINDOWPOS_UNDEFINED,
                              SDL_WINDOWPOS_UNDEFINED,
                              width, height,
                              SDL_WINDOW_SHOWN);

    render = SDL_CreateRenderer(mainwindow, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_TARGETTEXTURE );

    screen = SurfUnq(SDL_CreateRGBSurface(  0, width, height, colourDepth,
                                            0x00FF0000,
                                            0x0000FF00,
                                            0x000000FF,
                                            0xFF000000));

    texture = SDL_CreateTexture(    render,
                                    SDL_PIXELFORMAT_ARGB8888,
                                    SDL_TEXTUREACCESS_STREAMING,
                                    width, height);
    SDL_SetWindowTitle(mainwindow, "MPI Toybrot - Use Mouse1 to zoom in and Mouse2 to zoom out");

    frac = SurfPtr(SDL_CreateRGBSurface(0, width, height, colourDepth,
                                        0x00FF0000,
                                        0x0000FF00,
                                        0x000000FF,
                                        0xFF000000) );

    highlight = SurfUnq(SDL_CreateRGBSurface(0,w,h,colourDepth,
                                             0xFF000000,
                                             0x00FF0000,
                                             0x0000FF00,
                                             0x000000FF));

    SDL_SetSurfaceBlendMode(highlight.get(), SDL_BLENDMODE_BLEND);
    void* pix = highlight->pixels;
    for(int i = 0; i < frac->h; i++)
    {
        for(int j = 0; j< frac->w; j++)
        {

           auto p = reinterpret_cast<uint32_t*>(pix) +
                    (i * highlight->w)
                    + j;
            *p = SDL_MapRGB(frac->format, 255u, 255u, 255u);
        }
    }
    SDL_SetSurfaceAlphaMod(highlight.get(), 128u);
}

FracGenWindow::~FracGenWindow()
{
    SDL_DestroyWindow(mainwindow);
    SDL_DestroyRenderer(render);
    SDL_DestroyTexture(texture);
}

void FracGenWindow::paint()
{

    SDL_BlitSurface(frac.get(),nullptr,screen.get(),nullptr);
    if(drawRect)
    {
        drawHighlight();
    }

    SDL_UpdateTexture(texture, nullptr, screen->pixels, screen->pitch);
    SDL_RenderClear(render);
    SDL_RenderCopy(render, texture, nullptr, nullptr);
    SDL_RenderPresent(render);


}

bool FracGenWindow::captureEvents()
{
    SDL_Event event;
    while (SDL_PollEvent(&event))
    {
        switch (event.type)
        {
        case SDL_KEYDOWN:
        case SDL_KEYUP:
            return onKeyboardEvent(event.key);

        case SDL_MOUSEBUTTONDOWN:
        case SDL_MOUSEBUTTONUP:
            return onMouseButtonEvent(event.button);

        case SDL_QUIT:
            return false;

        case SDL_MOUSEMOTION:
            return onMouseMotionEvent(event.motion);

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
    return true;
}

void FracGenWindow::setTitle(std::string title)
{
    SDL_SetWindowTitle(mainwindow, title.c_str());
}

void FracGenWindow::drawHighlight()
{
    SDL_Rect r;
    r.x = (rectX<mouseX?rectX:mouseX);
    r.y = (rectY<mouseY?rectY:mouseY);
    r.w = abs(mouseX - rectX);
    r.h = abs(mouseY - rectY);
    SDL_BlitSurface(highlight.get(), &r, screen.get(),&r);
}

bool FracGenWindow::onKeyboardEvent(const SDL_KeyboardEvent &e) noexcept
{
    if(e.type == SDL_KEYDOWN)
    {
        if(e.keysym.sym == SDLK_ESCAPE)
        {
           return false;
        }
    }
    return true;
}

bool FracGenWindow::onMouseMotionEvent(const SDL_MouseMotionEvent &e) noexcept
{
    mouseX = e.x;
    mouseY = e.y;
    int rw = mouseX - rectX;
    int rh = mouseY - rectY;
    if (rh == 0)
    {
        return true;
    }
    float ra = abs(rw/rh);
    if ( (ra - aspectratio) <= FLT_EPSILON)
    {
        return true;
    }
    if (ra < aspectratio)
    {
        mouseX = static_cast<int>(rectX + rh*aspectratio);
    }
    else
    {
        mouseY = static_cast<int>(rectY + rw/aspectratio);
    }
    return true;
}

bool FracGenWindow::onMouseButtonEvent(const SDL_MouseButtonEvent &e) noexcept
{
    if (e.button == 4)
    {
        //M_WHEEL_UP
        //std::cout<< "button 4" <<std::endl;
        return true;
    }
    if (e.button == 5)
    {
        //M_WHEEL_DOWN
        //std::cout<< "button 5" <<std::endl;
        return true;
    }
    if (e.button == 2)
    {
        //Middle Button
        if(e.type == SDL_MOUSEBUTTONDOWN)
        {
            if(colourScheme != nullptr)
            {
                *colourScheme = (*colourScheme) +1;
                *redrawRequired = true;
            }
        }
        return true;
    }
    if(e.button == 3)
    {
        //Right Button
        if(ROI != nullptr)
        {
            ROI->Imax = 1.5l;
            ROI->Imin = -1.5l;
            ROI->Rmax = 1;
            ROI->Rmin = -2;
            if(e.type == SDL_MOUSEBUTTONUP)
            {
                *redrawRequired = true;
            }
        }
        return true;
    }
    if(e.type == SDL_MOUSEBUTTONDOWN)
    {
        rectX = e.x;
        rectY = e.y;
        drawRect = true;
    }
    else
    {
        int rx = mouseX;
        int ry = mouseY;

        long double x0 = ROI->Rmin + ((ROI->Rmax - ROI->Rmin)/width) * rectX;
        long double x1 = ROI->Rmin + ((ROI->Rmax - ROI->Rmin)/width) * rx;

        long double y0 = ROI->Imax - ((ROI->Imax - ROI->Imin)/height) * rectY;
        long double y1 = ROI->Imax - ((ROI->Imax - ROI->Imin)/height) * ry;

        ROI->Rmax = (x0>x1?x0:x1);
        ROI->Rmin = (x0>x1?x1:x0);


        ROI->Imax = (y0>y1?y0:y1);
        ROI->Imin = (y0>y1?y1:y0);

        drawRect = false;
        *redrawRequired = true;
    }
    return true;
}

bool operator==(const Region &r1, const Region &r2)
{
    return (   (r1.Imax - r2.Imax <= LDBL_EPSILON) && (r1.Imin - r2.Imin <= LDBL_EPSILON)
            && (r1.Rmax - r2.Rmax <= LDBL_EPSILON) && (r1.Rmin - r2.Rmin <= LDBL_EPSILON) );
}
