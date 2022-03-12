#include "FracGenWindow.hpp"
#include "FracGen.hpp"
#include <cfloat>
#include <cstdint>
#ifndef TB_SINGLETHREAD
    #include <thread>
    #include <future>
#endif
#include <iostream>
#include <limits>

#ifdef TB_OPENGL
static void checkGlError(const char* op)
{
    #ifndef NDEBUG
     for(GLint error = glGetError(); error; error= glGetError())
     {
        switch(error)
        {
            case GL_INVALID_ENUM:
            {
                printf("after %s() glError (GL_INVALID_ENUM)\n", op);
                break;
            }
            case GL_INVALID_VALUE:
            {
                printf("after %s() glError (GL_INVALID_VALUE)\n", op);
                break;
            }
            case GL_INVALID_OPERATION:
            {
                printf("after %s() glError (GL_INVALID_OPERATION)\n", op);
                break;
            }
            case GL_OUT_OF_MEMORY:
            {
                printf("after %s() glError (GL_OUT_OF_MEMORY)\n", op);
                break;
            }
            #ifdef __EMSCRIPTEN__
            case 0x9242:
            {
                printf("after %s() glError (CONTEXT_LOST_WEBGL)\n", op);
                break;
            }

            #endif
            default:
            {
                printf("after %s() glError (0x%x)\n", op, error);
            }
        }
     }
    #endif
}
#endif

FracGenWindow::FracGenWindow(CameraPtr c, std::string& flavourDesc, std::shared_ptr<bool> redrawFlag, std::shared_ptr<bool> exitFlag)
    : cam{c}
    , colourDepth{32}
    , drawRect{false}
    , HL{0,0,0,0}
    , mouseX{0}
    , mouseY{0}
    , redrawRequired{redrawFlag}
    , exitNow{exitFlag}
{


    SDL_Init(SDL_INIT_VIDEO);

    //#ifdef HAVE_OPENGLES
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
    //#endif

    SDL_GL_SetAttribute( SDL_GL_DOUBLEBUFFER, 1 );
    SDL_GL_SetAttribute( SDL_GL_ACCELERATED_VISUAL, 1 );

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
#ifdef TB_WEBGL_COMPAT
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
#else
    //We want 3.1 ES for compute shaders in the regular OpenGL project
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
#endif

    std::string title = flavourDesc + " Toybrot Mandelbox";
    mainwindow = SDL_CreateWindow(title.c_str(),
                              SDL_WINDOWPOS_UNDEFINED,
                              SDL_WINDOWPOS_UNDEFINED,
                              cam->ScreenWidth(), cam->ScreenHeight(),
                              SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL);
#ifdef TB_OPENGL

    #ifdef TB_WEBGL_COMPAT
        ubyteTex.resize(cam->ScreenWidth()*cam->ScreenHeight());
    #endif
    ctx = nullptr;
    initGL();
#else

    render = SDL_CreateRenderer(mainwindow, -1, 0);
    //SDL_RenderClear(render);

    screen = SurfPtr(SDL_CreateRGBSurface(  0, cam->ScreenWidth(), cam->ScreenHeight(), colourDepth,
                                            0xFF000000,
                                            0x00FF0000,
                                            0x0000FF00,
                                            0x000000FF));

    texture = SDL_CreateTexture(    render,
                                    SDL_PIXELFORMAT_RGBA8888,
                                    SDL_TEXTUREACCESS_STREAMING,
                                    cam->ScreenWidth(), cam->ScreenHeight());

    surface = SurfPtr(SDL_CreateRGBSurface(0, cam->ScreenWidth(), cam->ScreenHeight(), colourDepth,
                                        0xFF000000,
                                        0x00FF0000,
                                        0x0000FF00,
                                        0x000000FF));

    highlight = SurfUnq(SDL_CreateRGBSurface(0,cam->ScreenWidth(), cam->ScreenHeight(),colourDepth,
                                             0xFF000000,
                                             0x00FF0000,
                                             0x0000FF00,
                                             0x000000FF));

    SDL_SetSurfaceBlendMode(highlight.get(), SDL_BLENDMODE_BLEND);
    void* pix = highlight->pixels;
    for(int i = 0; i < surface->h; i++)
    {
        for(int j = 0; j< surface->w; j++)
        {

           auto p = reinterpret_cast<uint32_t*>(pix) +
                    (i * highlight->w)
                    + j;
            *p = SDL_MapRGB(surface->format, 255u, 255u, 255u);
        }
    }
    SDL_SetSurfaceAlphaMod(highlight.get(), 128u);
#endif
}

FracGenWindow::~FracGenWindow()
{
    SDL_DestroyWindow(mainwindow);
#ifdef TB_OPENGL
#else
    SDL_DestroyRenderer(render);
    //SDL_DestroyTexture(texture);
#endif
}

void FracGenWindow::paint()
{
#ifdef TB_OPENGL
    #ifdef __EMSCRIPTEN__
        if(emscripten_is_webgl_context_lost(emCtx))
        {
            initGL();
        }
    #endif
    glClear(GL_COLOR_BUFFER_BIT);
    checkGlError("glClear(GL_COLOR_BUFFER_BIT)");

    glDisable(GL_DEPTH_TEST);
    checkGlError("glDisable(GL_DEPTH_TEST)");
    glUseProgram(windowProgram);
    checkGlError("glUseProgram(windowProgram)");
    glEnableVertexAttribArray(0);
    checkGlError("glEnableVertexAttribArray(mainWindow)");
    glBindVertexArray(vao);
    checkGlError("glBindVertexArray(mainWindow)");

    glActiveTexture(GL_TEXTURE0);
    checkGlError("glActiveTexture(GL_TEXTURE0)");
    glBindTexture(GL_TEXTURE_2D, glTex);
    checkGlError("glBindTexture(GL_TEXTURE_2D, glTex)");
    glUniform1i(texLoc, 0);
    checkGlError("glUniform1i(texLoc, 0)");
    //glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glBindBuffer(GL_ARRAY_BUFFER, vertbuff);
    checkGlError("glBindBuffer(GL_ARRAY_BUFFER, vertbuff)");

    glDrawArrays(GL_TRIANGLES, 0, 6);
    checkGlError("glDrawArrays(mainWindow)");
    SDL_GL_SwapWindow(mainwindow);

#else
    SDL_SetRenderTarget(render, texture);
    SDL_RenderClear(render);

    SDL_BlitSurface(surface.get(),nullptr,screen.get(),nullptr);
//    if(drawRect)
//    {
//        drawHighlight();
//    }

    SDL_UpdateTexture(texture, nullptr, screen->pixels, screen->pitch);
    SDL_RenderClear(render);
    SDL_RenderCopy(render, texture, nullptr, nullptr);
    SDL_RenderPresent(render);
#endif

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
                break;
//            if(*redrawRequired)
//            {
//                return true;
//            }
//            else
//            {
//                return onMouseButtonEvent(event.button);
//            }
        case SDL_QUIT:
            *exitNow = true;
            return false;

        case SDL_MOUSEMOTION:
//            if(*redrawRequired)
//            {
//                return true;
//            }
//            else
//            {
//                return onMouseMotionEvent(event.motion);
//            }
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
#ifndef TB_SINGLETHREAD
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
#endif
    return true;
}

void FracGenWindow::setTitle(std::string title)
{
    SDL_SetWindowTitle(mainwindow, title.c_str());
}

void FracGenWindow::drawHighlight()
{
#ifdef TB_OPENGL
#else
    SDL_Rect r;
    r.x = (HL.x<mouseX?HL.x:mouseX);
    r.y = (HL.y<mouseY?HL.y:mouseY);
    r.w = abs(mouseX - HL.x);
    r.h = abs(mouseY - HL.y);
    SDL_BlitSurface(highlight.get(), &r, screen.get(),&r);
#endif
}

void FracGenWindow::updateFractal()
{

    if(fractal == nullptr)
    {
        return;
    }
    if(fractal->size() != static_cast<size_t>(cam->ScreenWidth() * cam->ScreenHeight()) )
    {
        std::cout << "Fractal and ScreenSize mismatch!" << std::endl;
        exit(2);
    }
#ifdef TB_OPENGL
    #ifdef TB_WEBGL_COMPAT
        #ifndef TB_SINGLETHREAD
            size_t num_threads = std::min(8ul, toyBrot::MaxThreads > 0 ? toyBrot::MaxThreads : std::thread::hardware_concurrency() );
            std::vector<std::thread> tasks (num_threads);
            // "classic" for loop because we need to know the idx here
            for(size_t idx = 0; idx < num_threads; idx++)
            {
                tasks[idx] = std::thread([this, idx, num_threads](){this->convertPixels(num_threads, idx);});
            }


            //Wait here and block until all threads are done

            for( auto& t : tasks)
            {
                if(t.joinable())
                {
                    t.join();
                }
            }

        #else
            convertPixels(1, 0);
        #endif
    #endif

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, glTex);
#ifdef TB_WEBGL_COMPAT
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, cam->ScreenWidth(), cam->ScreenHeight(), 0, GL_RGBA, GL_UNSIGNED_BYTE, ubyteTex.data());
#else
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, cam->ScreenWidth(), cam->ScreenHeight(), 0, GL_RGBA, GL_FLOAT, fractal->data());
#endif
    checkGlError("updateFractal");

#else


    /*
     * multi-threading the conversion to the SDL type
     * using similar logic/mechanisms to the STD::THREADS
     * version of the generation itself
     */
    size_t num_threads = std::min(8ul, toyBrot::MaxThreads > 0 ? toyBrot::MaxThreads : std::thread::hardware_concurrency() );
    std::vector<std::thread> tasks (num_threads);
    SDL_LockSurface(surface.get());
    // "classic" for loop because we need to know the idx here
    for(size_t idx = 0; idx < num_threads; idx++)
    {
        tasks[idx] = std::thread([this, idx, num_threads](){this->convertPixels(num_threads, idx);});
    }


    //Wait here and block until all threads are done

    for( auto& t : tasks)
    {
        if(t.joinable())
        {
            t.join();
        }
    }

    SDL_UnlockSurface(surface.get());
#endif
}

void FracGenWindow::convertPixels(size_t total_threads, size_t idx)
{
#if !defined(TB_OPENGL) || defined(TB_WEBGL_COMPAT)
    size_t surfLength = cam->ScreenWidth() * cam->ScreenHeight();
    for(size_t i = idx; i < surfLength; i += total_threads)
    {
        Vec4<uint8_t> colour8( static_cast<uint8_t>( (*fractal)[i].R() * std::numeric_limits<uint8_t>::max()),
                               static_cast<uint8_t>( (*fractal)[i].G() * std::numeric_limits<uint8_t>::max()),
                               static_cast<uint8_t>( (*fractal)[i].B() * std::numeric_limits<uint8_t>::max()),
                               static_cast<uint8_t>( (*fractal)[i].A() * std::numeric_limits<uint8_t>::max())   );

    #ifdef TB_OPENGL
        ubyteTex[i] = *reinterpret_cast<std::array<uint8_t,4>*>(&colour8);
    #else
        reinterpret_cast<uint32_t*>(surface->pixels)[i] = SDL_MapRGBA(surface->format, colour8.R(), colour8.G(), colour8.B(), colour8.A());
    #endif
    }    
#endif
}

/******************************************************************************
 *
 * SDL event handling below here
 *
 *****************************************************************************/


bool FracGenWindow::onKeyboardEvent(const SDL_KeyboardEvent &e) noexcept
{
    if(e.type == SDL_KEYDOWN)
    {
        if(e.keysym.sym == SDLK_ESCAPE)
        {
            *exitNow = true;
           return false;
        }
    }
    return true;
}

bool FracGenWindow::onMouseMotionEvent(const SDL_MouseMotionEvent &e) noexcept
{
    mouseX = e.x;
    mouseY = e.y;
    //int rw = mouseX - HL.x;
    int rh = mouseY - HL.y;
    if (rh == 0)
    {
        return true;
    }
    //double ra = abs(rw/rh);
//    if (ra < cam->AR)
//    {

//        mouseX = HL.x + static_cast<int>(rh*cam->AR);

//    }
//    else
//    {
//        mouseY = HL.x + static_cast<int>(rw/cam->AR);
//    }
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
        return true;
    }
    if(e.button == 1)
    {
        if(e.type == SDL_MOUSEBUTTONDOWN)
        {
            HL.x = e.x;
            HL.y = e.y;
            HL.w = 0;
            HL.h = 0;
            drawRect = true;
        }
        else
        {

            drawRect = false;
            *redrawRequired = true;
        }
        return true;
    }
    return false;
}

#ifdef TB_OPENGL
void FracGenWindow::initGL() noexcept
{

    if(ctx != nullptr)
    {
        SDL_GL_DeleteContext(ctx);
        ctx = nullptr;
    }
    ctx = SDL_GL_CreateContext(mainwindow);

#ifdef __EMSCRIPTEN__
    emCtx = emscripten_webgl_get_current_context();
    emscripten_webgl_enable_extension(emCtx, "OES_texture_float");
    emscripten_webgl_enable_extension(emCtx, "OES_texture_float_linear");
#endif
    //Let's get the shaders out of the way

    windowProgram = glCreateProgram();

    /**
     * Minimal shaders to just render some tex into a quad
     */
    std::stringstream vertSrc;
    vertSrc << "#version 300 es" << std::endl;
    vertSrc << "precision mediump float;" << std::endl;
    vertSrc << "layout(location = 0) in vec3 vertPos;" << std::endl;
    vertSrc << "out vec2 UV;" << std::endl;
    vertSrc << "void main(){" << std::endl;
    vertSrc << "gl_Position =  vec4(vertPos,1);" << std::endl;
    vertSrc << "UV = (vertPos.xy+vec2(1,1))/2.0;" << std::endl;
    vertSrc << "}" << std::endl;
    vertSrc << "" << std::endl;

    std::stringstream fragSrc;
    fragSrc << "#version 300 es" << std::endl;
    fragSrc << "precision mediump float;" << std::endl;
    fragSrc << "uniform sampler2D tex;" << std::endl;
    fragSrc << "in vec2 UV;" << std::endl;
    fragSrc << "out vec4 color;" << std::endl;
    fragSrc << "void main(){" << std::endl;
    fragSrc << "color = texture(tex, UV);" << std::endl;
    fragSrc << "}" << std::endl;
    fragSrc << "" << std::endl;

    std::string vertStr = vertSrc.str();
    std::string fragStr = fragSrc.str();
    const GLchar* glVertSrc = vertStr.c_str();
    const GLchar* glFragSrc = fragStr.c_str();
    GLint vertLength[1] = {static_cast<GLint>(vertStr.length())};
    GLint fragLength[1] = {static_cast<GLint>(fragStr.length())};

    GLint success = 0;
    GLchar info[512];

    GLuint vertID = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragID = glCreateShader(GL_FRAGMENT_SHADER);

    glShaderSource(vertID, 1, &glVertSrc, vertLength);
    glCompileShader(vertID);
    glGetShaderiv(vertID, GL_COMPILE_STATUS, &success);
    if(success == 0)
    {
        glGetShaderInfoLog(vertID, 512, NULL, info);
        std::cerr << "Error compiling window vertex shader: " << info << std::endl;
        exit(12);
    }

    glShaderSource(fragID, 1, &glFragSrc, fragLength);
    glCompileShader(fragID);
    glGetShaderiv(fragID, GL_COMPILE_STATUS, &success);
    if(success == 0)
    {
        glGetShaderInfoLog(fragID, 512, NULL, info);
        std::cerr << "Error compiling window frag shader: " << info << std::endl;
        exit(12);
    }


    glAttachShader(windowProgram, vertID);
    glAttachShader(windowProgram, fragID);
    glLinkProgram(windowProgram);
    checkGlError("glLinkProgram(windowProgram)");

    //We're not going to manipulate this further, so we're good with what's loaded on the program
    glDeleteShader(vertID);
    glDeleteShader(fragID);
    glUseProgram(windowProgram);
    checkGlError("glUseProgram(windowProgram)");

    texLoc = glGetUniformLocation(windowProgram, "tex");

    //And now set up the data we need for OpenGL

    glGenTextures(1, &glTex);
    glBindTexture(GL_TEXTURE_2D, glTex);
    #ifdef TB_WEBGL_COMPAT
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, cam->ScreenWidth(), cam->ScreenHeight(), 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    #else
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, cam->ScreenWidth(), cam->ScreenHeight(), 0, GL_RGBA, GL_FLOAT, 0);
    #endif
    checkGlError("glTexImage2D(mainWindow)");
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    verts =
    {
        -1.0f, -1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
        1.0f,  1.0f, 0.0f,
    };

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &vertbuff);
    glBindBuffer(GL_ARRAY_BUFFER, vertbuff);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * verts.size(), verts.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vertbuff);
    glVertexAttribPointer(
        0,                  // attribute 0
        3,                  // size
        GL_FLOAT,           // type
        GL_FALSE,           // normalized?
        0,                  // stride
        (void*)0            // array buffer offset
    );

    // Draw the triangles !
    glDrawArrays(GL_TRIANGLES, 0, 6); // 2*3 indices starting at 0 -> 2 triangles
    SDL_GL_SwapWindow(mainwindow);

    glDisableVertexAttribArray(0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindVertexArray(0);
    glUseProgram(0);

}
#endif
