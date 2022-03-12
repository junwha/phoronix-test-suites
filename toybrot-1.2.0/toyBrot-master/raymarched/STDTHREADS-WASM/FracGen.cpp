#include "FracGen.hpp"

#include <iostream>
#include <cfloat>
#include <functional>
#include <thread>
#include <future>
#include <vector>

static size_t oversubscriptionFactor = 20;
static size_t numThreads = std::thread::hardware_concurrency()*oversubscriptionFactor;

using  estimatorFunction = std::function<tbFPType(Vec3<tbFPType>, const Parameters<tbFPType>&)>;

//static Vec3<tbFPType> boxMaxes;
//static Vec3<tbFPType> boxMins;

/******************************************************************************
 *
 * Distance estimator functions and helpers
 *
 ******************************************************************************/

void sphereFold(Vec3<tbFPType>& z, tbFPType& dz, const Parameters<tbFPType>& params)
{
    tbFPType r2 = z.sqMod();
    if ( r2 < params.MinRadiusSq())
    {
        // linear inner scaling
        tbFPType temp = (params.FixedRadiusSq()/params.MinRadiusSq());
        z *= temp;
        dz *= temp;
    }
    else if(r2<params.FixedRadiusSq())
    {
        // this is the actual sphere inversion
        tbFPType temp =(params.FixedRadiusSq()/r2);
        z *= temp;
        dz*= temp;
    }
}

void boxFold(Vec3<tbFPType>& z, const Parameters<tbFPType>& params)
{
    z = z.clamp(-params.FoldingLimit(), params.FoldingLimit()) * static_cast<tbFPType>(2.0) - z;
}

tbFPType boxDist(const Vec3<tbFPType>& p, const Parameters<tbFPType>& params)
{
    /**
     * Distance estimator for a mandelbox
     *
     * Distance estimator adapted from
     * https://http://blog.hvidtfeldts.net/index.php/2011/11/distance-estimated-3d-fractals-vi-the-mandelbox/
     */
    const Vec3<tbFPType>& offset = p;
    tbFPType dr = params.BoxScale();
    Vec3<tbFPType> z{p};
    for (size_t n = 0; n < params.BoxIterations(); n++)
    {
        boxFold(z, params);       // Reflect
        sphereFold(z,dr, params);    // Sphere Inversion

        z = z * params.BoxScale() + offset;  // Scale & Translate
        dr = dr * std::abs(params.BoxScale()) + 1.0f;


    }
    tbFPType r = z.mod();
    return r/std::abs(dr);
}

tbFPType bulbDist(const Vec3<tbFPType>& p, const Parameters<tbFPType>&)
{

    /**
     * Distance estimator for a mandelbulb
     *
     * Distance estimator adapted from
     * https://www.iquilezles.org/www/articles/mandelbulb/mandelbulb.htm
     * https://www.shadertoy.com/view/ltfSWn
     */

    Vec3<tbFPType> w = p;
    tbFPType m = w.sqMod();

    //vec4 trap = vec4(abs(w),m);
    tbFPType dz = 3.0f;


    for( int i=0; i<4; i++ )
    {
#if 1
        tbFPType m2 = m*m;
        tbFPType m4 = m2*m2;
        dz = 8.0f*sqrt(m4*m2*m)*dz + 1.0f;

        tbFPType x = w.X(); tbFPType x2 = x*x; tbFPType x4 = x2*x2;
        tbFPType y = w.Y(); tbFPType y2 = y*y; tbFPType y4 = y2*y2;
        tbFPType z = w.Z(); tbFPType z2 = z*z; tbFPType z4 = z2*z2;

        tbFPType k3 = x2 + z2;
        tbFPType k2 = 1/sqrt( k3*k3*k3*k3*k3*k3*k3 );
        tbFPType k1 = x4 + y4 + z4 - 6.0f*y2*z2 - 6.0f*x2*y2 + 2.0f*z2*x2;
        tbFPType k4 = x2 - y2 + z2;

        w.setX(p.X() +  64.0f*x*y*z*(x2-z2)*k4*(x4-6.0f*x2*z2+z4)*k1*k2);
        w.setY(p.Y() + -16.0f*y2*k3*k4*k4 + k1*k1);
        w.setZ(p.Z() +  -8.0f*y*k4*(x4*x4 - 28.0f*x4*x2*z2 + 70.0f*x4*z4 - 28.0f*x2*z2*z4 + z4*z4)*k1*k2);
#else
        dz = 8.0*pow(sqrt(m),7.0)*dz + 1.0;
        //dz = 8.0*pow(m,3.5)*dz + 1.0;

        tbFPType r = w.mod();
        tbFPType b = 8.0*acos( w.Y()/r);
        tbFPType a = 8.0*atan2( w.X(), w.Z() );
        w = p + Vec3<tbFPType>( sin(b)*sin(a), cos(b), sin(b)*cos(a) ) * pow(r,8.0);
#endif

       // trap = min( trap, vec4(abs(w),m) );

        m = w.sqMod();
        if( m > 256.0f )
            break;
    }

    return 0.25f*log(m)*sqrt(m)/dz;
}

tbFPType sphereDist(Vec3<tbFPType> p, const Parameters<tbFPType>&)
{
    tbFPType radius = 2.f;
    return p.mod() - radius;
}

/******************************************************************************
 *
 * Coulouring functions and helpers
 *
 ******************************************************************************/

RGBA HSVtoRGB(int H, tbFPType S, tbFPType V)
{

    /**
     * adapted from
     * https://gist.github.com/kuathadianto/200148f53616cbd226d993b400214a7f
     */

    RGBA output;
    tbFPType C = S * V;
    tbFPType X = C * (1 - std::abs(std::fmod(H / 60.0, 2) - 1));
    tbFPType m = V - C;
    tbFPType Rs, Gs, Bs;

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
    output.setA( 1.0f );

    return output;
}

RGBA getColour(const Vec4<tbFPType>& steps, const Parameters<tbFPType>& params )
{
    RGBA colour;

    Vec3<tbFPType> position(steps.X(),steps.Y(),steps.Z());


    if(steps.W() >= params.MaxRaySteps())
    {
        return RGBA(params.BgRed(),params.BgGreen(),params.BgBlue(),params.BgAlpha());
    }

//        This is a good place to tbFPType check your bounds if you need to

//        boxMins.setX(std::min(boxMins.X(),position.X()));
//        boxMins.setY(std::min(boxMins.Y(),position.Y()));
//        boxMins.setZ(std::min(boxMins.Z(),position.Z()));

//        boxMaxes.setX(std::max(boxMaxes.X(),position.X()));
//        boxMaxes.setY(std::max(boxMaxes.Y(),position.Y()));
//        boxMaxes.setZ(std::max(boxMaxes.Z(),position.Z()));

    tbFPType saturation = params.SatValue();
    tbFPType hueVal = (position.Z() * params.HueFactor()) + params.HueOffset();
    int hue = static_cast<int>( trunc(fmod(hueVal, 360.0f) ) );
    hue = hue < 0 ? 360 + hue: hue;
    tbFPType value = params.ValueRange()*(1.0f - std::min(steps.W()*params.ValueFactor()/params.MaxRaySteps(), params.ValueClamp()));

    colour = HSVtoRGB(hue, saturation, value);

//    Simplest colouring, based only on steps (roughly distance from camera)
    //colour = RGBA(value,value,value,1.0f);

    return colour;
}

/******************************************************************************
 *
 * Ray marching functions and helpers
 *
 ******************************************************************************/

Vec4<tbFPType> trace(const Camera<tbFPType>& cam, const Parameters<tbFPType>& params, size_t x, size_t y, const estimatorFunction& f)
{
    /**
     * This function taken from
     * http://blog.hvidtfeldts.net/index.php/2011/06/distance-estimated-3d-fractals-part-i/
     */

    tbFPType totalDistance = 0.0f;
    unsigned int steps;

    Vec3<tbFPType> pixelPosition = cam.ScreenTopLeft() + (cam.ScreenRight() * static_cast<tbFPType>(x)) + (cam.ScreenUp() * static_cast<tbFPType>(y));

    Vec3<tbFPType> rayDir = pixelPosition - cam.Pos();
    rayDir.normalise();

    Vec3<tbFPType> p;
    for (steps=0; steps < params.MaxRaySteps(); steps++)
    {
        p = cam.Pos() + (rayDir * totalDistance);
        tbFPType distance = f(p, params);
        totalDistance += distance;
        if (distance < params.CollisionMinDist()) break;
    }
    //return both the steps and the actual position in space for colouring purposes
    return Vec4<tbFPType>{p,static_cast<tbFPType>(steps)};
}

void traceRegion(FracPtr data,
                 const Camera<tbFPType>& cam,
                 const Parameters<tbFPType>& params,
                 const estimatorFunction& f,
                 size_t h0, size_t heightStep,  size_t idx, std::vector<bool>& results)
{
    for(size_t h = h0; h < h0+heightStep; h++)
    {
        if (h >= cam.ScreenHeight())
        {
            results[idx] = true;
            return;
        }

        for(size_t w = 0 + idx; w < cam.ScreenWidth(); w+= numThreads )
        {
            (*data)[(h*cam.ScreenWidth())+w] = getColour(trace(cam, params, w, h, f), params);
        }
    }
}

/******************************************************************************
 *
 * Thread spawning section
 *
 ******************************************************************************/

bool FracGen::Generate()
{
    if(outBuffer->size() != cam->ScreenWidth()*cam->ScreenHeight())
    {
        outBuffer->resize(cam->ScreenWidth()*cam->ScreenHeight());
    }

    bool finishedGeneration = false;
    int heightStep = bench ? cam->ScreenHeight() : 10;

    estimatorFunction bulb(bulbDist);
    estimatorFunction box(boxDist);
    estimatorFunction sphere(sphereDist);

    std::vector< std::thread > threadPool(numThreads);

    std::vector<bool> results(threadPool.size());

    const auto& estimator = box;

    for(size_t i = 0; i < numThreads; i++)
    {
        threadPool[i] = std::thread([this, &estimator, heightStep, h = lastHeight, idx = i, &results]()
                                    {traceRegion(this->outBuffer, *(this->cam), *(this->parameters), estimator, h, heightStep, idx, results);});

    }
    lastHeight += heightStep;

    for(auto& td : threadPool)
    {
        //wait until all threads complete
        if(td.joinable())
        {
            td.join();
        }
    };

    for(bool b : results)
    {
        if(b)
        {
            //If one of them is done, they all must be
            lastHeight = 0;
            finishedGeneration = true;
//            std::cout << "Minimum bounds = " << boxMins.X() << " " << boxMins.Y() << " " << boxMins.Z() << std::endl;
//            std::cout << "Maximum bounds = " << boxMaxes.X() << " " << boxMaxes.Y() << " " << boxMaxes.Z() << std::endl;
        }
    }
    return finishedGeneration;
}

FracGen::FracGen(bool benching, CameraPtr c, ParamPtr p, const size_t factor)
    : bench{benching}
    , cam{c}
    , parameters{p}
    , lastHeight{0}
{

    outBuffer = std::make_shared< colourVec >(cam->ScreenWidth()*cam->ScreenHeight());

    static bool once = false;
    #ifndef TOYBROT_MAX_THREADS
        if(factor != 0)
        {
            oversubscriptionFactor = factor;
        }
    #else
        oversubscriptionFactor = 1;
    #endif

    numThreads = toyBrot::MaxThreads*oversubscriptionFactor;

    if(!bench || !once )
    {
        once = true;
        #ifdef TOYBROT_MAX_THREADS
            std::cout << "Running on " << toyBrot::MaxThreads << " threads" << std::endl;
            std::cout.flush();
        #else
            std::cout << "System reports " << toyBrot::MaxThreads << " native threads" << std::endl;
            std::cout << "We're going to spawn " << std::thread::hardware_concurrency() * oversubscriptionFactor << " threads in our pool" << std::endl;
        #endif
    }
}

FracGen::~FracGen()
{}



