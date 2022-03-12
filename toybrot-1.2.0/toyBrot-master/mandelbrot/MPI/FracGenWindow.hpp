#ifndef TOYBROT_FRACGENWINDOW_DEFINED
#define TOYBROT_FRACGENWINDOW_DEFINED

#ifdef WIN32
    #include <Windows.h>
    #include "SDL.h"
    #include "SDL_opengl.h"
#else

#include "SDL2/SDL.h"
#include "SDL2/SDL_opengl.h"

#endif

#include <memory>
#include <atomic>
#include <cfloat>
#include <vector>

struct Region{long double Imin,Imax,Rmin,Rmax;}; //This struct delimits a region in the Argand-Gauss Plane (R X I)

bool operator==(const Region& r1, const Region& r2);

struct RGB { uint8_t r,g,b;};
using PixelData = std::vector<uint32_t>;

using SurfPtr = std::shared_ptr<SDL_Surface>;
using SurfUnq = std::unique_ptr<SDL_Surface>;


class FracGenWindow
{
public:
    FracGenWindow(int width, int height, int bpp, std::shared_ptr<bool> redrawFlag);
    ~FracGenWindow();

    void draw(std::shared_ptr<SDL_Surface> surface);
    void paint();
    float AspectRatio() const {return aspectratio;}
    bool captureEvents();
    void registerRedrawFlag(std::shared_ptr<bool> b) {redrawRequired = b;}
    void registerColourFlag(std::shared_ptr<int> i) {colourScheme = i;}
    void setRegion(std::shared_ptr<Region> r) {ROI = r;}
    SurfPtr getFrac() {return frac;}
    SDL_PixelFormat* Format() {return screen->format;}
    void setTitle(std::string title);


private:

    void drawHighlight();

    bool onKeyboardEvent(const SDL_KeyboardEvent& e) noexcept;
    bool onMouseMotionEvent(const SDL_MouseMotionEvent& e) noexcept;
    bool onMouseButtonEvent(const SDL_MouseButtonEvent& e ) noexcept;

    int     width;
    int     height;
    float   aspectratio;
    int     colourDepth;
    bool    drawRect;
    int     rectX;
    int     rectY;
    int     mouseX;
    int     mouseY;

    SDL_Window*     mainwindow;
    SDL_Renderer*   render;
    SurfUnq         screen;
    SDL_Texture*    texture;
    SurfPtr         frac;
    SurfUnq         highlight;

    std::shared_ptr<bool>   redrawRequired;
    std::shared_ptr<Region> ROI;
    std::shared_ptr<int>    colourScheme;
};

#endif //TOYBROT_FRACGENWINDOW_DEFINED
