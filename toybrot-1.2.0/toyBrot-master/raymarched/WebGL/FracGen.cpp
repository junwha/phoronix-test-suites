#include "FracGen.hpp"

#include <iostream>
#include <fstream>
#include <cfloat>
#include <cstring>
#include <string>
#include <sstream>

#include <GLES3/gl32.h>
#include <GLES3/gl2ext.h>

#ifdef __EMSCRIPTEN__
    #include <emscripten.h>
    //#include <emscripten/html5.h>
#endif


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
    /**
     * A lot of the stuff here was adapted and/or copypasted from
     *
     * http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-14-render-to-texture/
     */

    /*
     * Time to set up the I/O
     */

    glUseProgram(glProgram);
    checkGlError("glUseProgram");



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
    GLuint fracFB = 0;
    GLuint fracTex = 0;
    GLenum drawBuff[1] = {GL_COLOR_ATTACHMENT0};
    cameraLocation = glGetUniformLocation(glProgram, "cam.camPos");;
    checkGlError("glGetUniformLocation(cam)");
    paramsLocation = glGetUniformLocation(glProgram, "params.hueFactor");;
    checkGlError("glGetUniformLocation(params)");

    setUniforms();

    glGenFramebuffers(1, &fracFB);
    checkGlError("genFrameBuffers");

    glBindFramebuffer(GL_FRAMEBUFFER, fracFB);
    checkGlError("glBindFramebuffer");

    glGenTextures(1, &fracTex);
    checkGlError("glGenTextures");
    //glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, fracTex);
    checkGlError("glBindTexture");
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, cam->ScreenWidth(), cam->ScreenHeight(), 0, GL_RGBA, GL_FLOAT, 0);
    checkGlError("glTexImage2D");


    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    checkGlError("glTexParameteri");


    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fracTex, 0);
    checkGlError("glFramebufferTexture2D");

    glGenVertexArrays(1, &vao);
    checkGlError("glGenVertexArrays");
    glBindVertexArray(vao);
    checkGlError("glBindVertexArray");

    glBindFramebuffer(GL_FRAMEBUFFER, fracFB);
    checkGlError("glBindFramebuffer");

    /**
      * This bit for the Vertices is just straight up copypasta, didn't even touch it, really
      */

    static const GLfloat g_quad_vertex_buffer_data[] = {
        -1.0f, -1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
        1.0f,  1.0f, 0.0f,
    };

    GLuint quad_vertexbuffer;
    glGenBuffers(1, &quad_vertexbuffer);
    checkGlError("glGenBuffers(quad_vertexbuffer)");
    glBindBuffer(GL_ARRAY_BUFFER, quad_vertexbuffer);
    checkGlError("glBindBuffer(quad_vertexbuffer)");

    glBufferData(GL_ARRAY_BUFFER, sizeof(g_quad_vertex_buffer_data), g_quad_vertex_buffer_data, GL_STATIC_DRAW);
    checkGlError("glBufferData(quad_vertexbuffer)");

    glEnableVertexAttribArray(0);
    checkGlError("glEnableVertexAttribArray(0)");
    glBindBuffer(GL_ARRAY_BUFFER, quad_vertexbuffer);
    checkGlError("glBindBuffer(quad_vertexbuffer)");
    glVertexAttribPointer(
        0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
        3,                  // size
        GL_FLOAT,           // type
        GL_FALSE,           // normalized?
        0,                  // stride
        (void*)0            // array buffer offset
    );
    checkGlError("glVertexAttribPointer(Frac)");

    // Draw the triangles !
    glDrawArrays(GL_TRIANGLES, 0, 6); // 2*3 indices starting at 0 -> 2 triangles
    checkGlError("glDrawArrays(Frac)");

    glDisableVertexAttribArray(0);
    checkGlError("glDisableVertexAttribArray(0)");

    //Do the work
    glDrawBuffers(1, drawBuff);

    checkGlError("glDrawBuffers");
    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        std::cout << "Error drawing OpenGL FrameBuffer!" << std::endl;
        emscripten_force_exit(12);
        exit(12);
    }


    //This is essentially how we tell OpenGL to wait
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    checkGlError("glReadBuffer");

    //And now we copy the data back
    glReadPixels(0,0,cam->ScreenWidth(), cam->ScreenHeight(), GL_RGBA, GL_FLOAT, outBuffer->data());
    checkGlError("glReadPixels");


    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    checkGlError("glBindFramebuffer(0)");
    glBindTexture(GL_TEXTURE_2D, 0);
    checkGlError("glBindTexture(GL_TEXTURE_2D, 0)");

//    glMatrixMode(GL_MODELVIEW);
//    glLoadIdentity();



    #ifndef NDEBUG
        Vec3f camPos;
        Vec3f camUp;
        Vec3f camRight;
        Vec3f camTarget;

        float camNear;
        float camFovY;

        uint32_t screenWidth;
        uint32_t screenHeight;

        Vec3f screenTopLeft;
        Vec3f screenUp;
        Vec3f screenRight;
        glGetUniformfv(glProgram, camLocs["camPos"], reinterpret_cast<GLfloat*>(&camPos));
        glGetUniformfv(glProgram, camLocs["camUp"], reinterpret_cast<GLfloat*>(&camUp));
        glGetUniformfv(glProgram, camLocs["camRight"], reinterpret_cast<GLfloat*>(&camRight));
        glGetUniformfv(glProgram, camLocs["camTarget"], reinterpret_cast<GLfloat*>(&camTarget));

        glGetUniformfv(glProgram, camLocs["camNear"], &camNear);
        glGetUniformfv(glProgram, camLocs["camFovY"], &camFovY);

        glGetUniformuiv(glProgram, camLocs["screenWidth"], &screenWidth);
        glGetUniformuiv(glProgram, camLocs["screenHeight"], &screenHeight);

        glGetUniformfv(glProgram, camLocs["screenTopLeft"], reinterpret_cast<GLfloat*>(&screenTopLeft));
        glGetUniformfv(glProgram, camLocs["screenUp"], reinterpret_cast<GLfloat*>(&screenUp));
        glGetUniformfv(glProgram, camLocs["screenRight"], reinterpret_cast<GLfloat*>(&screenRight));

        float hueFactor;
        int32_t hueOffset;
        float valueFactor;
        float valueRange;
        float valueClamp;
        float satValue;
        float bgRed;
        float bgGreen;
        float bgBlue;
        float bgAlpha;

        uint32_t maxRaySteps;
        float collisionMinDist;

        float fixedRadiusSq;
        float minRadiusSq;
        float foldingLimit;
        float boxScale;
        uint32_t boxIterations;

        glGetUniformfv(glProgram, paramLocs["hueFactor"], &hueFactor);
        glGetUniformiv(glProgram, paramLocs["hueOffset"], &hueOffset);
        glGetUniformfv(glProgram, paramLocs["valueFactor"], &valueFactor);
        glGetUniformfv(glProgram, paramLocs["valueRange"], &valueRange);
        glGetUniformfv(glProgram, paramLocs["valueClamp"], &valueClamp);
        glGetUniformfv(glProgram, paramLocs["satValue"], &satValue);
        glGetUniformfv(glProgram, paramLocs["bgRed"], &bgRed);
        glGetUniformfv(glProgram, paramLocs["bgGreen"], &bgGreen);
        glGetUniformfv(glProgram, paramLocs["bgBlue"], &bgBlue);
        glGetUniformfv(glProgram, paramLocs["bgAlpha"], &bgAlpha);

        glGetUniformuiv(glProgram, paramLocs["maxRaySteps"], &maxRaySteps);
        glGetUniformfv(glProgram, paramLocs["collisionMinDist"], &collisionMinDist);

        glGetUniformfv(glProgram, paramLocs["fixedRadiusSq"], &fixedRadiusSq);
        glGetUniformfv(glProgram, paramLocs["minRadiusSq"], &minRadiusSq);
        glGetUniformfv(glProgram, paramLocs["foldingLimit"], &foldingLimit);
        glGetUniformfv(glProgram, paramLocs["boxScale"], &boxScale);
        glGetUniformuiv(glProgram, paramLocs["boxIterations"], &boxIterations);


        std::cout << "Values from the OpenGL shader" << std::endl;
        std::cout << std::endl << "Camera:" << std::endl << std::endl;
        std::cout << "camPos        -> " << camPos << std::endl;
        std::cout << "camUp         -> " << camUp << std::endl;
        std::cout << "camRight      -> " << camRight << std::endl;
        std::cout << "camTarget     -> " << camTarget << std::endl;
        std::cout << std::endl;
        std::cout << "camNear       -> " << camNear << std::endl;
        std::cout << "camFovY       -> " << camFovY << std::endl;
        std::cout << std::endl;
        std::cout << "screenWidth   -> " << screenWidth << std::endl;
        std::cout << "screenHeight  -> " << screenHeight << std::endl;;
        std::cout << std::endl;
        std::cout << "screenTopLeft -> " << screenTopLeft << std::endl;
        std::cout << "screenUp      -> " << screenUp << std::endl;
        std::cout << "screenRight   -> " << screenRight << std::endl;
        std::cout << std::endl << std::endl << "Parameters:" << std::endl << std::endl;
        std::cout << "hueFactor        -> " << hueFactor << std::endl;
        std::cout << "hueOffset        -> " << hueOffset << std::endl;
        std::cout << "valueFactor      -> " << valueFactor << std::endl;
        std::cout << "valueRange       -> " << valueRange << std::endl;
        std::cout << "valueClamp       -> " << valueClamp << std::endl;
        std::cout << "satValue         -> " << satValue << std::endl;
        std::cout << "bgRed            -> " << bgRed << std::endl;
        std::cout << "bgGreen          -> " << bgGreen << std::endl;
        std::cout << "bgBlue           -> " << bgBlue << std::endl;
        std::cout << "bgAlpha          -> " << bgAlpha << std::endl;
        std::cout << "maxRaySteps      -> " << maxRaySteps << std::endl;
        std::cout << "collisionMinDist -> " << collisionMinDist << std::endl;
        std::cout << "fixedRadiusSq    -> " << fixedRadiusSq << std::endl;
        std::cout << "minRadiusSq      -> " << minRadiusSq << std::endl;
        std::cout << "foldingLimit     -> " << foldingLimit << std::endl;
        std::cout << "boxScale         -> " << boxScale << std::endl;
        std::cout << "boxIterations    -> " << boxIterations << std::endl;

    #endif

    glUseProgram(0);
    checkGlError("glUseProgram(0)");


}

void FracGen::setUniforms()
{
    camLocs["camPos"] = glGetUniformLocation(glProgram,"cam.camPos");
    camLocs["camUp"] = glGetUniformLocation(glProgram,"cam.camUp");
    camLocs["camRight"] = glGetUniformLocation(glProgram,"cam.camRight");
    camLocs["camTarget"] = glGetUniformLocation(glProgram,"cam.camTarget");
    camLocs["camTarget"] = glGetUniformLocation(glProgram,"cam.camTarget");
    camLocs["camNear"] = glGetUniformLocation(glProgram,"cam.camNear");
    camLocs["camFovY"] = glGetUniformLocation(glProgram,"cam.camFovY");
    camLocs["screenWidth"] = glGetUniformLocation(glProgram,"cam.screenWidth");
    camLocs["screenHeight"] = glGetUniformLocation(glProgram,"cam.screenHeight");
    camLocs["screenTopLeft"] = glGetUniformLocation(glProgram,"cam.screenTopLeft");
    camLocs["screenUp"] = glGetUniformLocation(glProgram,"cam.screenUp");
    camLocs["screenRight"] = glGetUniformLocation(glProgram,"cam.screenRight");
//#ifndef NDEBUG
//    std::cout<< std::endl << "****************" << std::endl;

//    std::cout << "Uniform locations for:" <<std::endl;
//#endif
    for(auto& loc: camLocs)
    {
//#ifndef NDEBUG
//        std::cout << "cam." << loc.first<<": "<< loc.second <<std::endl;
//#endif
        if(loc.second == -1)
        {
            std::cerr << "Error acquiring uniform location for cam."<< loc.first << std::endl;
            emscripten_force_exit(12);
            exit(12);
        }
    }

    paramLocs["hueFactor"]          = glGetUniformLocation(glProgram,"params.hueFactor");
    paramLocs["hueOffset"]          = glGetUniformLocation(glProgram,"params.hueOffset");
    paramLocs["valueFactor"]        = glGetUniformLocation(glProgram,"params.valueFactor");
    paramLocs["valueRange"]         = glGetUniformLocation(glProgram,"params.valueRange");
    paramLocs["valueClamp"]         = glGetUniformLocation(glProgram,"params.valueClamp");
    paramLocs["satValue"]           = glGetUniformLocation(glProgram,"params.satValue");
    paramLocs["bgRed"]              = glGetUniformLocation(glProgram,"params.bgRed");
    paramLocs["bgGreen"]            = glGetUniformLocation(glProgram,"params.bgGreen");
    paramLocs["bgBlue"]             = glGetUniformLocation(glProgram,"params.bgBlue");
    paramLocs["bgAlpha"]            = glGetUniformLocation(glProgram,"params.bgAlpha");
    paramLocs["maxRaySteps"]        = glGetUniformLocation(glProgram,"params.maxRaySteps");
    paramLocs["collisionMinDist"]   = glGetUniformLocation(glProgram,"params.collisionMinDist");
    paramLocs["fixedRadiusSq"]      = glGetUniformLocation(glProgram,"params.fixedRadiusSq");
    paramLocs["minRadiusSq"]        = glGetUniformLocation(glProgram,"params.minRadiusSq");
    paramLocs["foldingLimit"]       = glGetUniformLocation(glProgram,"params.foldingLimit");
    paramLocs["boxScale"]           = glGetUniformLocation(glProgram,"params.boxScale");
    paramLocs["boxIterations"]      = glGetUniformLocation(glProgram,"params.boxIterations");


    for(auto& loc: paramLocs)
    {
//        #ifndef NDEBUG
//            std::cout << "params." << loc.first<<": "<< loc.second <<std::endl;
//        #endif
        if(loc.second == -1)
        {
            std::cerr << "Error acquiring uniform location for params."<< loc.first << std::endl;
            emscripten_force_exit(12);
            exit(12);
        }
    }
//    #ifndef NDEBUG
//        std::cout<< std::endl << "****************" << std::endl;
//    #endif


    glUniform3f(camLocs["camPos"],        cam->Pos().X(),cam->Pos().Y(),cam->Pos().Z());
    glUniform3f(camLocs["camUp"],         cam->Up().X(),cam->Up().Y(),cam->Up().Z());
    glUniform3f(camLocs["camRight"],      cam->Right().X(),cam->Right().Y(),cam->Right().Z());
    glUniform3f(camLocs["camTarget"],     cam->Target().X(),cam->Target().Y(),cam->Target().Z());

    glUniform1f(camLocs["camNear"],       cam->Near());
    glUniform1f(camLocs["camFovY"],       cam->FovY());

    glUniform1ui(camLocs["screenWidth"],  cam->ScreenWidth());
    glUniform1ui(camLocs["screenHeight"], cam->ScreenHeight());

    glUniform3f(camLocs["screenTopLeft"], cam->ScreenTopLeft().X(),cam->ScreenTopLeft().Y(),cam->ScreenTopLeft().Z());
    glUniform3f(camLocs["screenUp"],      cam->ScreenUp().X(),cam->ScreenUp().Y(),cam->ScreenUp().Z());
    glUniform3f(camLocs["screenRight"],   cam->ScreenRight().X(),cam->ScreenRight().Y(),cam->ScreenRight().Z());

    glUniform1f(paramLocs["hueFactor"],         parameters->HueFactor());
    glUniform1i(paramLocs["hueOffset"],         parameters->HueOffset());
    glUniform1f(paramLocs["valueFactor"],       parameters->ValueFactor());
    glUniform1f(paramLocs["valueRange"],        parameters->ValueRange());
    glUniform1f(paramLocs["valueClamp"],        parameters->ValueClamp());
    glUniform1f(paramLocs["satValue"],          parameters->SatValue());
    glUniform1f(paramLocs["bgRed"],             parameters->BgRed());
    glUniform1f(paramLocs["bgGreen"],           parameters->BgGreen());
    glUniform1f(paramLocs["bgBlue"],            parameters->BgBlue());
    glUniform1f(paramLocs["bgAlpha"],           parameters->BgAlpha());

    glUniform1ui(paramLocs["maxRaySteps"],      parameters->MaxRaySteps());
    glUniform1f(paramLocs["collisionMinDist"],  parameters->CollisionMinDist());

    glUniform1f(paramLocs["fixedRadiusSq"],     parameters->FixedRadiusSq());
    glUniform1f(paramLocs["minRadiusSq"],       parameters->MinRadiusSq());
    glUniform1f(paramLocs["foldingLimit"],      parameters->FoldingLimit());
    glUniform1f(paramLocs["boxScale"],          parameters->BoxScale());
    glUniform1ui(paramLocs["boxIterations"],    parameters->BoxIterations());

}


FracGen::FracGen(bool benching, CameraPtr c, ParamPtr p)
    : bench{benching}
    , outBuffLocation{0}
    , cameraLocation{1}
    , paramsLocation{2}
    , cam{c}
    , parameters{p}
    , camLocs
        {
            {"camPos",-1},
            {"camUp",-1},
            {"camRight",-1},
            {"camTarget",-1},
            {"camNear",-1},
            {"camFovY",-1},
            {"screenWidth",-1},
            {"screenHeight",-1},
            {"screenTopLeft",-1},
            {"screenUp",-1},
            {"screenRight",-1}
        }
    , paramLocs
        {
            {"hueFactor",-1},
            {"hueOffset",-1},
            {"valueFactor",-1},
            {"valueRange",-1},
            {"valueClamp",-1},
            {"satValue",-1},
            {"bgRed",-1},
            {"bgGreen",-1},
            {"bgBlue",-1},
            {"bgAlpha",-1},
            {"maxRaySteps",-1},
            {"collisionMinDist",-1},
            {"fixedRadiusSq",-1},
            {"minRadiusSq",-1},
            {"foldingLimit",-1},
            {"boxScale",-1},
            {"boxIterations",-1}
        }
{
    outBuffer = std::make_shared< colourVec >(cam->ScreenWidth()*cam->ScreenHeight());
    std::string shaderSrc{"FracGen.frag"};


#ifndef TOYBROT_ENABLE_GUI
    //Try and initialise the OpenGL stuff here. Otherwise, SDL has got our back
#endif


    static bool once = false;
    if(!once || !bench )
    {
        once = true;
        std::cout << glGetString(GL_VERSION) << std::endl;
    }

    //Hands off the SDL internal stuff
    glUseProgram(0);

    glProgram = glCreateProgram();
    GLuint fragID = glCreateShader(GL_FRAGMENT_SHADER);

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
    std::string alternativeVersion{"#version 300 es\n"};
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
        emscripten_force_exit(12);
        exit(12);
    }
    const GLchar* glFragsrc = src.c_str();
    GLint success = 0;
    GLint fragLength[1] = {static_cast<GLint>(src.length())};
    GLchar info[512];
    glShaderSource(fragID, 1, &glFragsrc, fragLength);
    glCompileShader(fragID);
    glGetShaderiv(fragID, GL_COMPILE_STATUS, &success);
    if(success == 0)
    {
        glGetShaderInfoLog(fragID, 512, NULL, info);
        std::cerr << "Error compiling fragment shader: " << info << std::endl;
        emscripten_force_exit(12);
        exit(12);
    }


    /**
     * The vertex shader here is a big old nothingburger. Just assume A quad and
     * output some UVs for the fragments
     */
    std::stringstream vertSrc;
    vertSrc << "#version 300 es" << std::endl;
    vertSrc << "layout(location = 0) in vec3 vertPos;" << std::endl;
    vertSrc << "out vec2 UV;" << std::endl;
    vertSrc << "void main(){" << std::endl;
    vertSrc << "gl_Position =  vec4(vertPos,1);" << std::endl;
    vertSrc << "UV = (vertPos.xy+vec2(1,1))/2.0;" << std::endl;
    vertSrc << "}" << std::endl;
    vertSrc << "" << std::endl;


    GLuint vertID = glCreateShader(GL_VERTEX_SHADER);
    std::string vertStr = vertSrc.str();
    const GLchar* glVertsrc = vertStr.c_str();

    GLint VertLength[1] = {static_cast<GLint>(vertSrc.str().length())};
    glShaderSource(vertID, 1, &glVertsrc, VertLength);
    glCompileShader(vertID);
    glGetShaderiv(vertID, GL_COMPILE_STATUS, &success);
    if(success == 0)
    {
        glGetShaderInfoLog(vertID, 512, NULL, info);
        std::cerr << "Error compiling vertex shader: " << info << std::endl;
        emscripten_force_exit(12);
        exit(12);
    }

    glAttachShader(glProgram, vertID);
    glAttachShader(glProgram, fragID);
    checkGlError("glAttachShader");


    glLinkProgram(glProgram);
    checkGlError("glLinkProgram");

    //We're not going to manipulate this further, so we're good with what's loaded on the program
    glDeleteShader(vertID);
    checkGlError("DeleteShader(Vert)");
    glDeleteShader(fragID);
    checkGlError("DeleteShader(Frag)");
    glUseProgram(0);

    debugprint("Constructor done");

}


FracGen::~FracGen()
{}
