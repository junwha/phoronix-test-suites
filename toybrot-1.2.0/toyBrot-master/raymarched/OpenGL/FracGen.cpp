#include "FracGen.hpp"

#include <iostream>
#include <fstream>
#include <cfloat>
#include <cstring>
#include <string>
#include <GLES3/gl31.h>


constexpr const tbFPType WORKGROUP_SIZE = 16;

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
            default:
            {
                printf("after %s() glError (0x%x)\n", op, error);
            }
        }
     }
    #endif
}

static void debugprint(const char* msg)
{
    #ifndef NDEBUG
        std::cout << msg << std::endl;
    #endif
}

void FracGen::Generate()
{
    /*
     * Time to set up the I/O
     */

    glUseProgram(glProgram);

    /*
     * Thanks, in part, to this shader having been written for Vulkan, which
     * is very strict, we know these bindings to all have been very well defined
     */
    if(outBuffer->size() != cam->ScreenHeight() * cam->ScreenWidth())
    {
        outBuffer->assign(cam->ScreenHeight() * cam->ScreenWidth(), RGBA{0,0,0,0});
    }

    debugprint("Generating");

    GLuint vao = 0;
    GLuint outBuffVBO = 0;
    GLuint camVBO = 0;
    GLuint paramsVBO = 0;

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &outBuffVBO);
    checkGlError("genBuffers(outBuff)");
    glGenBuffers(1, &camVBO);
    checkGlError("genBuffers(cam)");
    glGenBuffers(1, &paramsVBO);
    checkGlError("genBuffers(params)");

    debugprint("Buffers generated");

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, outBuffVBO);
    checkGlError("bindBuffer(outBuff)");
    debugprint("outBuff bound successfully");
    glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(outSize()), outBuffer->data(), GL_STATIC_COPY);
    checkGlError("bufferData(outBuff)");
    debugprint("outBuff data transferred successfully");

    glBindBuffer(GL_UNIFORM_BUFFER, camVBO);
    glBufferData(GL_UNIFORM_BUFFER, static_cast<GLsizeiptr>(sizeof(*cam)), cam.get(), GL_STATIC_READ);

    glBindBuffer(GL_UNIFORM_BUFFER, paramsVBO);
    glBufferData(GL_UNIFORM_BUFFER, static_cast<GLsizeiptr>(sizeof(*parameters)), parameters.get(), GL_STATIC_READ);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    debugprint("Buffers data transfered");

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, outBuffLocation, outBuffVBO);
    glBindBufferBase(GL_UNIFORM_BUFFER, cameraLocation,  camVBO);
    glBindBufferBase(GL_UNIFORM_BUFFER, paramsLocation,  paramsVBO);


    glDispatchCompute(static_cast<uint32_t>( ceil(cam->ScreenWidth()  / WORKGROUP_SIZE))
                      , static_cast<uint32_t>( ceil(cam->ScreenHeight() / WORKGROUP_SIZE))
                      , 1);
    checkGlError("glDispatchCompute");
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, outBuffVBO);
    auto ptr = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, static_cast<GLsizeiptr>(outSize()), GL_MAP_READ_BIT);
    checkGlError("glMapBufferRange");
    if(ptr == nullptr)
    {
        std::cout << "Error mapping OpenGL buffer!" << std::endl;
        exit(12);
    }

    memcpy( outBuffer->data(), ptr, outSize());

    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    checkGlError("glUnmapBuffer");
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    checkGlError("glUnbindBuffer");

    #ifndef NDEBUG
            std::cout << "Values from the OpenGL shader" << std::endl;
            std::cout << std::endl << "Camera:" << std::endl << std::endl;
            std::cout << "camPos        -> " << (*outBuffer)[0] << std::endl;
            std::cout << "camUp         -> " << (*outBuffer)[1] << std::endl;
            std::cout << "camRight      -> " << (*outBuffer)[2] << std::endl;
            std::cout << "camTarget     -> " << (*outBuffer)[3] << std::endl;
            std::cout << std::endl;
            std::cout << "camNear       -> " << (*outBuffer)[4].X() << std::endl;
            std::cout << "camFovY       -> " << (*outBuffer)[4].Y() << std::endl;
            std::cout << std::endl;
            std::cout << "screenWidth   -> " << (*outBuffer)[4].Z() << std::endl;
            std::cout << "screenHeight  -> " << (*outBuffer)[4].W() << std::endl;;
            std::cout << std::endl;
            std::cout << "screenTopLeft -> " << (*outBuffer)[5] << std::endl;
            std::cout << "screenUp      -> " << (*outBuffer)[6] << std::endl;
            std::cout << "screenRight   -> " << (*outBuffer)[7] << std::endl;
            std::cout << std::endl << std::endl << "Parameters:" << std::endl << std::endl;
            std::cout << "hueFactor        -> " << (*outBuffer)[8].X() << std::endl;
            std::cout << "hueOffset        -> " << (*outBuffer)[8].Y() << std::endl;
            std::cout << "valueFactor      -> " << (*outBuffer)[8].Z() << std::endl;
            std::cout << "valueRange       -> " << (*outBuffer)[8].W() << std::endl;
            std::cout << "valueClamp       -> " << (*outBuffer)[9].X() << std::endl;
            std::cout << "satValue         -> " << (*outBuffer)[9].Y() << std::endl;
            std::cout << "bgRed            -> " << (*outBuffer)[9].Z() << std::endl;
            std::cout << "bgGreen          -> " << (*outBuffer)[9].W() << std::endl;
            std::cout << "bgBlue           -> " << (*outBuffer)[10].X() << std::endl;
            std::cout << "bgAlpha          -> " << (*outBuffer)[10].Y() << std::endl;
            std::cout << "maxRaySteps      -> " << (*outBuffer)[10].Z() << std::endl;
            std::cout << "collisionMinDist -> " << (*outBuffer)[10].W() << std::endl;
            std::cout << "fixedRadiusSq    -> " << (*outBuffer)[11].X() << std::endl;
            std::cout << "minRadiusSq      -> " << (*outBuffer)[11].Y() << std::endl;
            std::cout << "foldingLimit     -> " << (*outBuffer)[11].Z() << std::endl;
            std::cout << "boxScale         -> " << (*outBuffer)[11].W() << std::endl;
            std::cout << "boxIterations    -> " << (*outBuffer)[12].X() << std::endl;

    #endif

}


FracGen::FracGen(bool benching, CameraPtr c, ParamPtr p)
    : bench{benching}
    , outBuffLocation{0}
    , cameraLocation{1}
    , paramsLocation{2}
    , cam{c}
    , parameters{p}
{
    outBuffer = std::make_shared< colourVec >(cam->ScreenWidth()*cam->ScreenHeight());
    std::string shaderSrc{"FracGen.comp.glsl"};


#ifndef TOYBROT_ENABLE_GUI
    //Try and initialise the OpenGL stuff here. Otherwise, SDL has got our back
#endif


    static bool once = false;
    if(!once || !bench )
    {
        once = true;
        std::cout << glGetString(GL_VERSION) << std::endl;
    }

    glProgram = glCreateProgram();
    GLuint shaderID = glCreateShader(GL_COMPUTE_SHADER);

    /*
     * All right, so a few "needlessly weird" things about to happen here
     * I want a few things which are not super trivial to juggle together
     *
     * 1 - I want to have preprocessor switches in my shader for debugging and doubles
     * In Vulkan, you can just forward the defines to glslangValidator, you're not really
     * consuming the .glsl file directly, but in OpenGL, you need to manually edit the string
     *
     * 2 - I want to have the shader using openGL version 310 es. The reason for this is I want
     * to later port this to webGPU through emscripten, so I need to be on an ES profile
     *
     * 3 - BUT es profiles don't have double support (at least not dvec3/4 from the validator's complaints)
     *
     * 4 - And I want to have just the one source for both openGL and Vulkan
     *
     * With all of that in mind, I need to do some massaging of the shader source here which makes
     * this part more involved than one'd expect (ifstream rdbuf, done)
     */

    std::string src;
    std::ifstream shaderFile;
    std::string ln;
    std::string additionalDefines{""};
    std::string alternativeVersion{"#version 310 es\n"};
#ifndef NDEBUG
        additionalDefines += "#define TOYBROT_DEBUG\n";
#endif
#ifdef TOYBROT_USE_DOUBLES
        additionalDefines += "#define TOYBROT_USE_DOUBLES\n";
        alternativeVersion = "";
#endif

    try
    {
        shaderFile.open(shaderSrc);
        if(!shaderFile.is_open())
        {
            throw std::ifstream::failure("Couldn't open file "+ shaderSrc);
        }
        while(std::getline(shaderFile,ln))
        {
            if(!ln.empty())
            {
                if(!ln.compare(0,8,"#version"))
                {

                    if(!alternativeVersion.empty())
                    {
                        src += alternativeVersion;
                    }
                    else
                    {
                        (src += ln) += '\n';
                    }
                    src+= additionalDefines;
                }
                else
                {
                    (src += ln) += '\n';
                }
            }
            else
            {
                /*
                 *  I could just remove the else and have this here
                 *  but it would make the defines if present bit fiddlier
                 *  Conversely, not having this here, while functional, makes
                 *  debugging the shader much harder
                 */

                src += '\n';
            }
        }
        shaderFile.close();
    }
    catch (std::ifstream::failure e)
    {
        std::cerr << "Error reading shader file: " << e.what() << std::endl;
        exit(12);
    }
    const GLchar* glsrc = src.c_str();
    GLint success = 0;
    GLint length[1] ={static_cast<GLint>(src.length())};
    GLchar info[512];
    glShaderSource(shaderID, 1, &glsrc, length);
    glCompileShader(shaderID);
    glGetShaderiv(shaderID, GL_COMPILE_STATUS, &success);
    if(success == 0)
    {
        glGetShaderInfoLog(shaderID, 512, NULL, info);
        std::cerr << "Error compiling shader: " << info << std::endl;
        exit(12);
    }

    glAttachShader(glProgram, shaderID);
    glLinkProgram(glProgram);
    checkGlError("glLinkProgram");

    //We're not going to manipulate this further, so we're good with what's loaded on the program
    glDeleteShader(shaderID);
    checkGlError("DeleteShader");
    debugprint("Constructor done");
}


FracGen::~FracGen()
{}
