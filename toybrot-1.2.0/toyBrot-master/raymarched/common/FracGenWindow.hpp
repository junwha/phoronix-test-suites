#ifndef TOYBROT_FRACGENWINDOW_DEFINED
#define TOYBROT_FRACGENWINDOW_DEFINED

#ifdef WIN32
    #include <Windows.h>
    #include "SDL.h"
    #include "SDL_opengl.h"
#else

    #include "SDL2/SDL.h"
    #ifdef TB_OPENGL
        #include "SDL2/SDL_opengl.h"
        #ifdef TB_WEBGL_COMPAT
            #include <GLES3/gl3.h>
            #include <array>
            #include <vector>
        #else
            #include <GLES3/gl31.h>
        #endif
        #ifdef __EMSCRIPTEN__
            #include <emscripten.h>
            #include <emscripten/html5.h>
        #endif
    #endif

#endif

#include <memory>
#include "FracGen.hpp"

/**
  * The main reason we include FracGen.hpp here is due to its alias
  * declarations. Making use of them, we can treat FracGenWindows as
  * a "fake template class", where whether it's using doubles or floats
  * depends on the alias definition in FracGen, making it easy to code
  * for both
  */


//Defining some convenience types

using SurfPtr = std::shared_ptr<SDL_Surface>;
using SurfUnq = std::unique_ptr<SDL_Surface>;

class FracGenWindow
{
public:
    FracGenWindow(CameraPtr camera, std::string &flavourDesc, std::shared_ptr<bool> redrawFlag, std::shared_ptr<bool> exit);
    ~FracGenWindow();

    void draw(SurfPtr surface);
    void paint();

    double AspectRatio() const noexcept;
    bool captureEvents();
    void registerRedrawFlag(std::shared_ptr<bool> b) noexcept {redrawRequired = b;}
    void registerColourFlag(std::shared_ptr<int> i) noexcept {colourScheme = i;}
    //SurfPtr getSurf() noexcept {return surface;}
    void setFractal(FracPtr f) noexcept {fractal = f;}
    void setTitle(std::string title);
    void setCamPos(Vec3<tbFPType> p) noexcept;
    void setCamTarget(Vec3<tbFPType> t) noexcept;
    Vec3<tbFPType> CamPos() const noexcept;
    Vec3<tbFPType> CamTarget() const noexcept;

    void updateFractal();

private:

    void drawHighlight();
    void convertPixels(size_t total_threads, size_t idx);

    bool onKeyboardEvent(const SDL_KeyboardEvent& e) noexcept;
    bool onMouseMotionEvent(const SDL_MouseMotionEvent& e) noexcept;
    bool onMouseButtonEvent(const SDL_MouseButtonEvent& e ) noexcept;
#ifdef TB_OPENGL
    void initGL() noexcept;
#endif

    void resetHL(int x0, int y0);

    CameraPtr   cam;
    int         colourDepth;
    bool        drawRect;
    SDL_Rect    HL;
    int         mouseX;
    int         mouseY;


    SDL_Window* mainwindow;
#ifdef TB_OPENGL
    SDL_GLContext ctx;
    #ifdef __EMSCRIPTEN__
        EMSCRIPTEN_WEBGL_CONTEXT_HANDLE  emCtx;
    #endif
    GLuint windowProgram;
    GLuint glTex;
    GLuint vao;
    GLuint vertbuff;
    GLint texLoc;
    std::vector<GLfloat> verts;
    #ifdef TB_WEBGL_COMPAT
        std::vector<std::array<uint8_t, 4>> ubyteTex;
    #endif
#else
    SDL_Renderer* render;
    SurfPtr screen;
    SDL_Texture* texture;
    SurfPtr surface;
    SurfUnq highlight;
#endif


    std::string flavour;
    std::shared_ptr<bool> redrawRequired;
    std::shared_ptr<int> colourScheme;
    std::shared_ptr<bool> exitNow;
    FracPtr fractal;
};

#endif //TOYBROT_FRACGENWINDOW_DEFINED
