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

#include "FracGen.hpp"

using SurfPtr = std::shared_ptr<SDL_Surface>;
using SurfUnq = std::unique_ptr<SDL_Surface>;


class FracGenWindow
{
public:
    FracGenWindow(int width, int height, std::shared_ptr<bool> redrawFlag, std::shared_ptr<bool> exit);
    ~FracGenWindow();

    void draw(std::shared_ptr<SDL_Surface> surface);
    void paint();
    float AspectRatio() const {return AR;}
    bool captureEvents();
    void registerRedrawFlag(std::shared_ptr<bool> b) {redrawRequired = b;}
    void registerColourFlag(std::shared_ptr<int> i) {colourScheme = i;}
    void setRegion(std::shared_ptr<Region> r) {ROI = r;}
    SurfPtr getFrac() {return frac;}
    void setTitle(std::string title);


private:

    void drawHighlight();

    bool onKeyboardEvent(const SDL_KeyboardEvent& e) noexcept;
    bool onMouseMotionEvent(const SDL_MouseMotionEvent& e) noexcept;
    bool onMouseButtonEvent(const SDL_MouseButtonEvent& e ) noexcept;

    void resetHL(int x0, int y0);

    int         width;
    int         height;
    int         colourDepth;
    bool        drawRect;
    SDL_Rect    HL;
    int         mouseX;
    int         mouseY;
    float       AR;

    SDL_Window* mainwindow;
    SDL_Renderer* render;
    SurfPtr screen;
    SDL_Texture* texture;
    SurfPtr frac;
    SurfUnq highlight;

    std::shared_ptr<bool> redrawRequired;
    std::shared_ptr<Region> ROI;
    std::shared_ptr<int> colourScheme;
    std::shared_ptr<bool> exitNow;
};

#endif //TOYBROT_FRACGENWINDOW_DEFINED
