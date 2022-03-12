#include "FracGen.hpp"

#include <iostream>
#include <cfloat>
#include <vector>

#include <omp.h>

//static Vec3<tbFPType> boxMaxes;
//static Vec3<tbFPType> boxMins;

/******************************************************************************
 *
 * Distance estimator functions and helpers
 *
 ******************************************************************************/

//void sphereFold(Vec3<tbFPType>& z, tbFPType& dz, const ParamPtr params)
//{
//    tbFPType r2 = z.sqMod();
//    if ( r2 < parameters->MinRadiusSq())
//    {
//        // linear inner scaling
//        tbFPType temp = (parameters->FixedRadiusSq()/parameters->MinRadiusSq());
//        z *= temp;
//        dz *= temp;
//    }
//    else if(r2<parameters->FixedRadiusSq())
//    {
//        // this is the actual sphere inversion
//        tbFPType temp =(parameters->FixedRadiusSq()/r2);
//        z *= temp;
//        dz*= temp;
//    }
//}

//void boxFold(Vec3<tbFPType>& z, const ParamPtr params)
//{
//    z = z.clamp(-parameters->FoldingLimit(), parameters->FoldingLimit())* 2.0f - z;
//}

tbFPType FracGen::boxDist(const Vec3<tbFPType>& p) const
{
    /**
     * Distance estimator for a mandelbox
     *
     * Distance estimator adapted from
     * https://http://blog.hvidtfeldts.net/index.php/2011/11/distance-estimated-3d-fractals-vi-the-mandelbox/
     */
    const Vec3<tbFPType>& offset = p;
    tbFPType dr = parameters->BoxScale();
    Vec3<tbFPType> z{p};
    for (size_t n = 0; n < parameters->BoxIterations(); n++)
    {
        //boxFold(z, params);       // Reflect
        z = z.clamp(-parameters->FoldingLimit(), parameters->FoldingLimit())* static_cast<tbFPType>(2.0) - z;

        //sphereFold(z,dr, params);    // Sphere Inversion
        tbFPType r2 = z.sqMod();
        if ( r2 < parameters->MinRadiusSq())
        {
            // linear inner scaling
            tbFPType temp = (parameters->FixedRadiusSq()/parameters->MinRadiusSq());
            z *= temp;
            dr *= temp;
        }
        else if(r2<parameters->FixedRadiusSq())
        {
            // this is the actual sphere inversion
            tbFPType temp =(parameters->FixedRadiusSq()/r2);
            z *= temp;
            dr*= temp;
        }
        z = z * parameters->BoxScale() + offset;  // Scale & Translate
        dr = dr * std::abs(parameters->BoxScale()) + 1.0f;


    }
    tbFPType r = z.mod();
    return r/std::abs(dr);
}

tbFPType FracGen::bulbDist(const Vec3<tbFPType>& p) const
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

tbFPType FracGen::sphereDist(Vec3<tbFPType> p) const
{
    tbFPType radius = 2.f;
    return p.mod() - radius;
}


/******************************************************************************
 *
 * Coulouring functions and helpers
 *
 ******************************************************************************/

RGBA FracGen::HSVtoRGB(int H, tbFPType S, tbFPType V) const
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

RGBA FracGen::getColour(const Vec4<tbFPType>& steps) const
{
    RGBA colour;

    Vec3<tbFPType> position(steps.X(),steps.Y(),steps.Z());

    if(steps.W() >= parameters->MaxRaySteps())
    {
        return RGBA(parameters->BgRed(),parameters->BgGreen(),parameters->BgBlue(),parameters->BgAlpha());
    }

//        This is a good place to tbFPType check your bounds if you need to

//        boxMins.setX(std::min(boxMins.X(),position.X()));
//        boxMins.setY(std::min(boxMins.Y(),position.Y()));
//        boxMins.setZ(std::min(boxMins.Z(),position.Z()));

//        boxMaxes.setX(std::max(boxMaxes.X(),position.X()));
//        boxMaxes.setY(std::max(boxMaxes.Y(),position.Y()));
//        boxMaxes.setZ(std::max(boxMaxes.Z(),position.Z()));

    tbFPType saturation = parameters->SatValue();
    tbFPType hueVal = (position.Z() * parameters->HueFactor()) + parameters->HueOffset();
    int hue = static_cast<int>( trunc(fmod(hueVal, 360.0f) ) );
    hue = hue < 0 ? 360 + hue: hue;
    tbFPType value = parameters->ValueRange()*(1.0f - std::min(steps.W()*parameters->ValueFactor()/parameters->MaxRaySteps(), parameters->ValueClamp()));

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

Vec4<tbFPType> FracGen::trace(const Camera<tbFPType>& cam, size_t x, size_t y) const
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
    for (steps=0; steps < parameters->MaxRaySteps(); steps++)
    {
        p = cam.Pos() + (rayDir * totalDistance);
        tbFPType distance = boxDist(p);
        totalDistance += distance;
        if (distance < parameters->CollisionMinDist()) break;
    }
    //return both the steps and the actual position in space for colouring purposes
    return Vec4<tbFPType>{p,static_cast<tbFPType>(steps)};
}

bool FracGen::traceRegion(colourVec& data,
                 const Camera<tbFPType> &cam,
                 uint32_t h0, uint32_t heightStep) const
{    
    for(size_t h = h0; h < h0+heightStep; h++)
    {
        if (h >= cam.ScreenHeight())
        {
            return true;
        }

        for(size_t w = 0 +  static_cast<size_t>(omp_get_thread_num()); w < cam.ScreenWidth(); w+=  static_cast<size_t>(omp_get_num_threads()) )
        {
            data[(h*cam.ScreenWidth())+w] = getColour(trace(cam, w, h));
        }
    }
    return false;
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

    //estimatorFunction bulb(bulbDist);

    std::vector<bool> results(static_cast<size_t>(omp_get_num_procs()));

    #pragma omp parallel for
    for(size_t i = 0; i < static_cast<size_t>(omp_get_num_procs()); i++)
    {
        results[i]= traceRegion(*outBuffer, *cam, lastHeight, heightStep);
    }

    lastHeight+= heightStep;

    for(bool b: results)
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

FracGen::FracGen(bool benching, CameraPtr c, ParamPtr p)
    : bench{benching}
    , cam{c}
    , parameters{p}
    , lastHeight{0}
{

    outBuffer = std::make_shared< colourVec >(cam->ScreenWidth()*cam->ScreenHeight());

    static bool once = false;
    if(!bench || !once)
    {
      std::cout << "OpenMP reports " << omp_get_num_procs() << " native threads" << std::endl;
      once = true;
    }
}

FracGen::~FracGen()
{}

