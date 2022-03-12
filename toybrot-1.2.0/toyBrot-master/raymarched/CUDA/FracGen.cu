#include "FracGen.hpp"

#include <iostream>
#include <cfloat>
#include <cuda_runtime.h>
#include <nvfunctional>
#include <vector>
#include <stdexcept>

using  estimatorFunction = nvstd::function<tbFPType(Vec3<tbFPType>)>;

/******************************************************************************
 *
 * CUDA helper functions
 *
 ******************************************************************************/

inline void cudaCheck(int line)
{
    auto status = cudaGetLastError();
    if(status != cudaError::cudaSuccess)
    {
        std::cout << "CUDA error @" << line <<" -> " << cudaGetErrorString(status) << std::endl;
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

//specialize to avoid std::sqrt
template<>
__host__ __device__ tbFPType Vec3<tbFPType>::mod()
{
    return sqrtf(sqMod());
}

/**
  * So there's a curious disparity here. Running CUDA on my Maxwell
  * I get the expected result where declaring these __constant__ yields
  * better performance. However, on my Vega, using HIP, the situation is
  * the opposite, which is why these don't match
  */
__device__ __constant__ Parameters<tbFPType> params;
__device__ __constant__ Camera<tbFPType> camera;

/******************************************************************************
 *
 * Distance estimator functions and helpers
 *
 ******************************************************************************/



__device__ inline void sphereFold(Vec3<tbFPType>& z, tbFPType& dz)
{
    tbFPType r2 = z.sqMod();

    /**
      * Traditional GPU coding wisdom dictates you should avoid
      * having branching instructions, such as "if" statements.
      * One way to avoid it is turning them into conditional
      * mathematical expressions, such as the ones under here
      * However, once I tried doing this I actually had massive
      * slowdown compared to the more straighforward if blocks.
      *
      * I left this here to possibly revisit in the future and
      * as a curiosity. (Or heck, maybe I just did it very wrong)
      */

//    int cond1 = r2 < minRadiusSq;
//    int cond2 = (r2 < fixedRadiusSq) * !cond1;
//    //int cond3 = !cond1 * !cond2;
//    //tbFPType temp = ( (fixedRadiusSq/minRadiusSq) * cond1) + ( (fixedRadiusSq/r2) * cond2) + cond3;

//    tbFPType temp[]{1,fixedRadiusSq/minRadiusSq , fixedRadiusSq/r2};
//    int tempidx = cond1 + 2*cond2;
//    z *= temp[tempidx];
//    dz *= temp[tempidx];

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

__device__ inline void boxFold(Vec3<tbFPType>& z)
{
    z = z.clamp(-params.FoldingLimit(), params.FoldingLimit())* static_cast<tbFPType>(2.0) - z;
}

__device__ tbFPType boxDist(const Vec3<tbFPType>& p)
{
    /**
     * Distance estimator for a mandelbox
     *
     * Distance estimator adapted from
     * https://http://blog.hvidtfeldts.net/index.php/2011/11/distance-estimated-3d-fractals-vi-the-mandelbox/
     */
    const Vec3<tbFPType>& offset = p;
    tbFPType dr = params.BoxScale();
    Vec3<tbFPType> z = p;
    for (size_t n = 0; n < params.BoxIterations(); n++)
    {
        boxFold(z);       // Reflect
        sphereFold(z,dr);    // Sphere Inversion

        z = z * params.BoxScale() + offset;  // Scale & Translate
        dr = dr * abs(params.BoxScale()) + 1.0f;

    }

    return z.mod()/abs(dr);
}


__device__ tbFPType bulbDist(const Vec3<tbFPType>& p)
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
        dz = 8.0f*sqrtf(m4*m2*m)*dz + 1.0f;

        tbFPType x = w.X(); tbFPType x2 = x*x; tbFPType x4 = x2*x2;
        tbFPType y = w.Y(); tbFPType y2 = y*y; tbFPType y4 = y2*y2;
        tbFPType z = w.Z(); tbFPType z2 = z*z; tbFPType z4 = z2*z2;

        tbFPType k3 = x2 + z2;
        tbFPType k2 = 1/sqrtf( k3*k3*k3*k3*k3*k3*k3 );
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

    return 0.25f*log(m)*sqrtf(m)/dz;
}

__device__ tbFPType sphereDist(Vec3<tbFPType> p)
{
    tbFPType radius = 2.f;
    return p.mod() - radius;
}

/******************************************************************************
 *
 * Coulouring functions and helpers
 *
 ******************************************************************************/

__device__ RGBA HSVtoRGB(int H, tbFPType S, tbFPType V)
{

    /**
     * adapted from
     * https://gist.github.com/kuathadianto/200148f53616cbd226d993b400214a7f
     */

    RGBA output;
    tbFPType C = S * V;
    tbFPType X = C * (1 - abs(fmodf(H / 60.0, 2) - 1));
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

__device__ RGBA getColour(const Vec4<tbFPType>& steps)
{
    RGBA colour;

    Vec3<tbFPType> position(steps.X(),steps.Y(),steps.Z());

    if(steps.W() == params.MaxRaySteps())
    {
        return RGBA(params.BgRed(),params.BgGreen(),params.BgBlue(),params.BgAlpha());
    }

    tbFPType saturation = params.SatValue();
    tbFPType hueVal = (position.Z() * params.HueFactor()) + params.HueOffset();
    int hue = static_cast<int>( trunc(fmodf(hueVal, 360.0f) ) );
    hue = hue < 0 ? 360 + hue: hue;    tbFPType value = params.ValueRange()*(1.0f - min(steps.W()*params.ValueFactor()/params.MaxRaySteps(), params.ValueClamp()));

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

__device__ Vec4<tbFPType> trace(int x, int y, const estimatorFunction& f)
{
    /**
     * This function taken from
     * http://blog.hvidtfeldts.net/index.php/2011/06/distance-estimated-3d-fractals-part-i/
     */

    tbFPType totalDistance = 0.0f;
    unsigned int steps;

    Vec3<tbFPType> pixelPosition = camera.ScreenTopLeft() + (camera.ScreenRight() * static_cast<tbFPType>(x)) + (camera.ScreenUp() * static_cast<tbFPType>(y));

    Vec3<tbFPType> rayDir = pixelPosition - camera.Pos();
    rayDir.normalise();

    Vec3<tbFPType> p;
    for (steps=0; steps < params.MaxRaySteps(); steps++)
    {
        p = camera.Pos() + (rayDir * totalDistance);
        /**
         * You can enable this and use nvstd::function instead of hardcoding the estimator
         * it works, which is a plus for CUDA but it's quite slower so I thought it fairer to
         * just hard-code as with the other GPU implementations
         */
        //tbFPType distance = f(p);
        tbFPType distance = boxDist(p);
        totalDistance += distance;
        if (distance < params.CollisionMinDist()) break;
    }

    //return both the steps and the actual position in space for colouring purposes
    return Vec4<tbFPType>{p,static_cast<tbFPType>(steps)};
}

__global__ void traceRegion(RGBA* data)
{

    /**
     * nvstd::function will NOT work if the object is not created IN the kernel
     * If YOU ever play with this and tear your hair out with mysterious
     * illegal memory access issues. Now you have a lead to chase
     */

    const estimatorFunction bulb(bulbDist);
    const estimatorFunction box(boxDist);
    const estimatorFunction sphere(sphereDist);


    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int index = ((row*camera.ScreenWidth())+col);
    if (col >= camera.ScreenWidth() || row >= camera.ScreenHeight())
    {
        return;
    }

    data[index] = getColour( trace(col, row, box) );
}

/******************************************************************************
 *
 * Thread spawning section
 *
 ******************************************************************************/

void FracGen::Generate()
{
    if(outBuffer->size() != cam->ScreenWidth()*cam->ScreenHeight())
    {
        outBuffer->resize(cam->ScreenWidth()*cam->ScreenHeight());
    }

    dim3 threadsPerBlock(16,16);
    dim3 numBlocks(std::ceil(static_cast<tbFPType>(cam->ScreenWidth())/threadsPerBlock.x), std::ceil(static_cast<tbFPType>(cam->ScreenHeight())/threadsPerBlock.y));
    RGBA* devVect;
    cudaMalloc(&devVect, outSize());
    traceRegion<<<numBlocks,threadsPerBlock>>>(devVect);
    cudaCheck(__LINE__);
    cudaDeviceSynchronize();
    cudaCheck(__LINE__);
    cudaMemcpy(outBuffer->data(), devVect, outSize(), cudaMemcpyDeviceToHost);
    cudaCheck(__LINE__);
    cudaFree(devVect);
}

FracGen::FracGen(bool benching, CameraPtr c, ParamPtr p)
    : bench{benching}
    , cam{c}
    , parameters{p}
{
    outBuffer = std::make_shared< colourVec >(cam->ScreenWidth()*cam->ScreenHeight());

    static bool once = false;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cudaCheck(__LINE__);
    if(!once || !benching )
    {
        once = true;
        std::cout << "Running CUDA on:       "   << prop.name                      << std::endl;
        std::cout << "Clocked at:            "   << prop.clockRate/1000            << "MHz" << std::endl;
        std::cout << "Max Mem:               "   << prop.totalGlobalMem/1000000000 << "GB" << std::endl;
        std::cout << "Max threads per block: "   << prop.maxThreadsPerBlock        << std::endl;
    }
    cudaMemcpyToSymbol(params, parameters.get(),   sizeof (Parameters<tbFPType>), 0, cudaMemcpyHostToDevice );
    cudaMemcpyToSymbol(camera, cam.get(), sizeof (Camera<tbFPType>),     0, cudaMemcpyHostToDevice );
    cudaCheck(__LINE__);

}

FracGen::~FracGen()
{
    cudaDeviceReset();
    cudaCheck(__LINE__);
}



