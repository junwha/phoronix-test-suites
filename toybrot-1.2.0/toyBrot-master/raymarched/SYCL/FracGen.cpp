#include "FracGen.hpp"

#include <iostream>
#include <cfloat>
#include <atomic>
#include <vector>

#include <CL/sycl.hpp>

/******************************************************************************
 *
 * Tweakable parameters
 *
 ******************************************************************************/


__device__ static constexpr const size_t maxRaySteps      = 7500;
__device__ static constexpr const float  collisionMinDist = 0.00055f;
__device__ static constexpr const float  cameraZ          = -3.8f;


// coulouring parameters

__device__ static constexpr const float hueFactor   = -40.0f;
__device__ static constexpr const int   hueOffset   = 200;
__device__ static constexpr const float valueFactor = 32;
__device__ static constexpr const float valueRange  = 1.0f;
__device__ static constexpr const float valueClamp  = 0.95f;
__device__ static constexpr const float satValue    = 0.7f;
__device__ static constexpr const float bgValue     = 0.05f;
__device__ static constexpr const float bgAlpha     = 1.0f;



// Mandelbox constants
__device__ static constexpr const float  fixedRadiusSq  = 2.2f;
__device__ static constexpr const float  minRadiusSq    = 0.8f;
__device__ static constexpr const float  foldingLimit   = 1.45;
__device__ static constexpr const float  boxScale       = -3.5f;
__device__ static constexpr const size_t boxIterations  = 30;

/******************************************************************************
 *
 * Distance estimator functions and helpers
 *
 ******************************************************************************/

inline void sphereFold(Vec3f& z, float& dz)
{
    float r2 = z.sqMod();
    if ( r2 < minRadiusSq)
    {
        // linear inner scaling
        float temp = (fixedRadiusSq/minRadiusSq);
        z *= temp;
        dz *= temp;
    }
    else if(r2<fixedRadiusSq)
    {
        // this is the actual sphere inversion
        float temp =(fixedRadiusSq/r2);
        z *= temp;
        dz*= temp;
    }
}

inline void boxFold(Vec3f& z)
{
    z = z.clamp(-foldingLimit, foldingLimit)* 2.0f - z;
}

float boxDist(const Vec3f& p)
{
    /**
     * Distance estimator for a mandelbox
     *
     * Distance estimator adapted from
     * https://http://blog.hvidtfeldts.net/index.php/2011/11/distance-estimated-3d-fractals-vi-the-mandelbox/
     */
    const Vec3f& offset = p;
    float dr = boxScale;
    Vec3f z{p};
    for (size_t n = 0; n < boxIterations; n++)
    {
        boxFold(z);       // Reflect
        sphereFold(z,dr);    // Sphere Inversion

        z = z * boxScale + offset;  // Scale & Translate
        dr = dr * cl::sycl::fabs(boxScale) + 1.0f;

    }
    float r = z.mod();
    return r/cl::sycl::fabs(dr);
}


float bulbDist(const Vec3f& p)
{

    /**
     * Distance estimator for a mandelbulb
     *
     * Distance estimator adapted from
     * https://www.iquilezles.org/www/articles/mandelbulb/mandelbulb.htm
     * https://www.shadertoy.com/view/ltfSWn
     */

    Vec3f w = p;
    float m = w.sqMod();

    //vec4 trap = vec4(abs(w),m);
    float dz = 5.0f;


    for( int i=0; i<4; i++ )
    {
#if 1
        float m2 = m*m;
        float m4 = m2*m2;
        dz = 8.0f*sqrt(m4*m2*m)*dz + 1.0f;

        float x = w.X(); float x2 = x*x; float x4 = x2*x2;
        float y = w.Y(); float y2 = y*y; float y4 = y2*y2;
        float z = w.Z(); float z2 = z*z; float z4 = z2*z2;

        float k3 = x2 + z2;
        float k2 = 1/sqrt( k3*k3*k3*k3*k3*k3*k3 );
        float k1 = x4 + y4 + z4 - 6.0f*y2*z2 - 6.0f*x2*y2 + 2.0f*z2*x2;
        float k4 = x2 - y2 + z2;

        w.setX(p.X() +  64.0f*x*y*z*(x2-z2)*k4*(x4-6.0f*x2*z2+z4)*k1*k2);
        w.setY(p.Y() + -16.0f*y2*k3*k4*k4 + k1*k1);
        w.setZ(p.Z() +  -8.0f*y*k4*(x4*x4 - 28.0f*x4*x2*z2 + 70.0f*x4*z4 - 28.0f*x2*z2*z4 + z4*z4)*k1*k2);
#else
        dz = 8.0*pow(sqrt(m),7.0)*dz + 1.0;
        //dz = 8.0*pow(m,3.5)*dz + 1.0;

        float r = w.mod();
        float b = 8.0*acos( w.Y()/r);
        float a = 8.0*atan2( w.X(), w.Z() );
        w = p + Vec3f( sin(b)*sin(a), cos(b), sin(b)*cos(a) ) * pow(r,8.0);
#endif

       // trap = min( trap, vec4(abs(w),m) );

        m = w.sqMod();
        if( m > 256.0f )
            break;
    }

    return 0.25f*cl::sycl::log(m)*cl::sycl::sqrt(m)/dz;
}

float sphereDist(Vec3f p)
{
    float radius = 2.7f;
    return p.mod() - radius;
}

/******************************************************************************
 *
 * Coulouring functions and helpers
 *
 ******************************************************************************/

RGBA HSVtoRGB(int H, float S, float V)
{

    /**
     * adapted from
     * https://gist.github.com/kuathadianto/200148f53616cbd226d993b400214a7f
     */

    RGBA output;
    float C = S * V;
    float X = C * (1 - cl::sycl::fabs(cl::sycl::fmod(H / 60.0f, 2) - 1));
    float m = V - C;
    float Rs, Gs, Bs;

    if(H >= 0 && H < 60)
    {
        Rs = C;
        Gs = X;
        Bs = 0;
    }
    else if(H >= 60 && H < 120)
    {
        Rs = X;
        Gs = C;
        Bs = 0;
    }
    else if(H >= 120 && H < 180)
    {
        Rs = 0;
        Gs = C;
        Bs = X;
    }
    else if(H >= 180 && H < 240)
    {
        Rs = 0;
        Gs = X;
        Bs = C;
    }
    else if(H >= 240 && H < 300)
    {
        Rs = X;
        Gs = 0;
        Bs = C;
    }
    else {
        Rs = C;
        Gs = 0;
        Bs = X;
    }

    output.setR(Rs + m);
    output.setG(Gs + m);
    output.setB(Bs + m);
    output.setA(1.0f);

    return output;
}

RGBA getColour(const Vec4f& steps)
{
    RGBA colour;

    Vec3f position(steps.X(),steps.Y(),steps.Z());

    const RGBA background(bgValue,bgValue,bgValue,bgAlpha);

    if(steps.W() == maxRaySteps)
    {
        return background;
    }

    float saturation = satValue;
    float hueVal = (position.Z() * hueFactor) + hueOffset;
    int hue = static_cast<int>( cl::sycl::trunc(cl::sycl::fmod(hueVal, 360.0f) ) );
    hue = hue < 0 ? 360 + hue: hue;
    float value = valueRange * (1.0f - cl::sycl::min(steps.W()*valueFactor/maxRaySteps, valueClamp));

    colour = HSVtoRGB(hue, saturation, value);

//  Simplest colouring, based only on steps (roughly distance from camera)
    //colour = RGBA(value,value,value,1.0f);

    return colour;
}

/******************************************************************************
 *
 * Ray marching functions and helpers
 *
 ******************************************************************************/

Vec4f trace(const Camera& cam, const Screen& s, int x, int y)
{
    /**
     * This function taken from
     * http://blog.hvidtfeldts.net/index.php/2011/06/distance-estimated-3d-fractals-part-i/
     */

    float totalDistance = 0.0f;
    unsigned int steps;

    Vec3f pixelPosition = s.topLeft + Vec3f{s.pixelWidth*x, s.pixelHeight * y, 0.f};

    Vec3f rayDir = pixelPosition -  static_cast<Vec3f>(cam.pos);
    rayDir.normalise();

    Vec3f p;
    for (steps=0; steps < maxRaySteps; steps++)
    {
        p = cam.pos + (rayDir * totalDistance);
        //float distance = cl::sycl::min(boxDist(p),bulbDist(p));
        float distance = boxDist(p);
        totalDistance += distance;
        if (distance < collisionMinDist) break;
    }
    //return both the steps and the actual position in space for colouring purposes
    return Vec4f{p,static_cast<float>(steps)};
}

template < typename Accessor>
void traceRegion(Accessor data,
                 Camera cam, Screen scr,
                 /*cl::sycl::id<2> tid*/cl::sycl::nd_item<2> tid)
{
    int col = tid.get_global_id(0);
    int row = tid.get_global_id(1);
    int index = ((row*scr.width)+col);

    if (col > scr.width || row > scr.height || index > scr.width*scr.height)
    {
        return;
    }

    data[index] = getColour(trace(cam, scr, col, row));
}

/******************************************************************************
 *
 * Thread spawning section
 *
 ******************************************************************************/

void FracGen::Generate(int width, int height)
{
    /*
     * calculate the rectangle which represents the screen (camera z near) in object space
     * No need to have an actual general camera so I'm just assuming the camera
     * always sits on the Z axis and always has (0,1,0) as it's up vector

     * This allows me to cheat a lot and not have to actually go into the
     * linear algebra side and write something like gluUnproject
     */
    Screen s;

    Vec3f screenPlaneOrigin{cam->pos.X(),cam->pos.Y(),cam->pos.Z() + cam->near};
    float screenPlaneHeight = 2*(cam->near*sin(cam->fovY/2));
    screenPlaneHeight = screenPlaneHeight < 0 ? -screenPlaneHeight : screenPlaneHeight;
    float screenPlaneWidth = screenPlaneHeight * cam->AR;
    // if 0,0 is top left, pixel height needs to be a negative
    s.width  = width;
    s.height = height;
    s.pixelHeight = (-1.f) * screenPlaneHeight / s.height;
    s.pixelWidth = screenPlaneWidth / s.width;
    s.topLeft = Vec3f {screenPlaneOrigin.X() - (screenPlaneWidth/2),
                       screenPlaneOrigin.Y() + (screenPlaneHeight/2),
                       screenPlaneOrigin.Z()                          };
    try
    {
        Camera c (*cam);

        //This is how we'll split the workload
        cl::sycl::range<2> workgroup(16,16);
        const size_t globalWidth = width%workgroup.get(0) == 0? width : ((width/workgroup.get(0))+1)*workgroup.get(0);
        const size_t globalHeight = height%workgroup.get(1) == 0? height : ((height/workgroup.get(1))+1)*workgroup.get(1);
        cl::sycl::range<2> pixels(globalWidth,globalHeight);
        //Create the device-side buffer
        cl::sycl::buffer<RGBA,1> buff (outBuffer->data(), width*height);
        q->submit([&](cl::sycl::handler& cgh)
                        {
                            auto access_v = buff.get_access<cl::sycl::access::mode::write>(cgh);
                            cgh.parallel_for<class syclmarchingkernel>
                                    (  cl::sycl::nd_range<2>(pixels, workgroup),
                                       [=] (/*cl::sycl::id<2> tid*/cl::sycl::nd_item<2> tid)
                                       { traceRegion(access_v, c, s,tid);}
                                    );
                        });
        q->wait_and_throw();
    }
    catch(cl::sycl::exception const& e)
    {
        std::cout << "SYCL sync exception -> " << e.what() << std::endl;
    }
    catch(...)
    {
        std::cout << " Exception caught! " << std::endl;
    }
}

FracGen::FracGen(bool benching, size_t width, size_t height)
    : bench{benching}
    , cam{new Camera}
{
    outBuffer = std::make_shared< colourVec >(width*height);
    cam->AR = static_cast<double>(width)/static_cast<double>(height);
    // Position here is more or less ignored. It's used for initial screen calculation but the
    // rays are launched from a different distance, specified in the .cl file
    cam->pos = Vec3f{0, 0, cameraZ};
    cam->target = Vec3f{0,0,0};
    cam->up = Vec3f{0,1,0};
    cam->near = 0.1f;
    cam->fovY = 45;
    static bool once = false;

    cl::sycl::default_selector device_selector;

    cl::sycl::async_handler sycl_err_handler =  [] (cl::sycl::exception_list exceptions)
                                            {
                                                for (std::exception_ptr const& e : exceptions)
                                                {
                                                    try
                                                    {
                                                        std::rethrow_exception(e);
                                                    }
                                                    catch(cl::sycl::exception const& e)
                                                    {
                                                        std::cout << "SYCL async exception -> " << e.what() << std::endl;
                                                    }
                                                }
                                            };

    q = std::make_unique<cl::sycl::queue>(device_selector, sycl_err_handler);

    if(!once || !bench )
    {
        std::cout << "Running on "
                  << q->get_device().get_info<cl::sycl::info::device::name>()
                  << std::endl ;
        once = true;
    }
}

FracGen::~FracGen()
{}



