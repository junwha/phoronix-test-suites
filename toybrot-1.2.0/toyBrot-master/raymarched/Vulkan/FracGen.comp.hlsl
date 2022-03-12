

#ifdef TOYBROT_USE_DOUBLES
    #define tbFPType   double
    #define tbFPType3  double3
    #define tbFPType4  double4
#else
    #define tbFPType   float
    #define tbFPType3  float3
    #define tbFPType4  float4
#endif


struct Camera
{
    tbFPType3 pos;
    tbFPType3 up;
    tbFPType3 right;
    tbFPType3 target;

    tbFPType padding;

    tbFPType near;
    tbFPType fovY;

    uint width;
    uint height;

    tbFPType3 screenTopLeft;
    tbFPType3 screenUp;
    tbFPType3 screenRight;
};

struct Parameters
{
    tbFPType hueFactor;
    int   hueOffset;
    tbFPType valueFactor;
    tbFPType valueRange;
    tbFPType valueClamp;
    tbFPType satValue;
    tbFPType bgRed;
    tbFPType bgGreen;
    tbFPType bgBlue;
    tbFPType bgAlpha;

    uint  maxRaySteps;
    tbFPType collisionMinDist;

    tbFPType fixedRadiusSq;
    tbFPType minRadiusSq;
    tbFPType foldingLimit;
    tbFPType boxScale;
    uint  boxIterations;
};

[[ vk::binding(0) ]]
RWStructuredBuffer<tbFPType4> data;

[[ vk::binding(1) ]]
ConstantBuffer<Camera> cam;
[[ vk::binding(2) ]]
ConstantBuffer<Parameters> params;


// implementation independent mod
#define mod(x, y) ( x - y * trunc(x / y) )

/******************************************************************************
 *
 * Distance estimator functions
 *
 ******************************************************************************/


void sphereFold(inout tbFPType3 z, inout tbFPType dz)
{
    tbFPType rsq = dot(z,z);
    /** Unlike CUDA and HIP, in Vulkan doing the
      * conditional shenanigans does not increase
      * the function's runtime. However, it doesn't
      * seem to decrease it either so there's not
      * really any point in doing it as all you're
      * left with is less readable code
      */
   // int cond1 = int(rsq < minRadiusSq);
   // int cond2 = int((rsq < fixedRadiusSq) && !bool(cond1));
   // int cond3 = int(!bool(cond1) && !bool(cond2));
   // tbFPType temp = ( (fixedRadiusSq/minRadiusSq) * cond1) + ( (fixedRadiusSq/rsq) * cond2) + cond3;
   // z *= temp;
   // dz *= temp;

    if ( rsq < params.minRadiusSq)
    {
        // linear inner scaling
        tbFPType temp = (params.fixedRadiusSq/params.minRadiusSq);
        z *= temp;
        dz *= temp;
    }
    else if(rsq < params.fixedRadiusSq )
    {
        // this is the actual sphere inversion
        tbFPType temp = (params.fixedRadiusSq/rsq);
        z *= temp;
        dz *= temp;
    }
}

void boxFold(inout tbFPType3 z)
{
    z = clamp(z, -params.foldingLimit, params.foldingLimit)* tbFPType(2) - z;
}

tbFPType boxDist(in tbFPType3 p)
{
    /**
     * Distance estimator for a mandelbox
     *
     * Distance estimator adapted from
     * https://http://blog.hvidtfeldts.net/index.php/2011/11/distance-estimated-3d-fractals-vi-the-mandelbox/
     */
    tbFPType3 offset = p;
    tbFPType dr = params.boxScale;
    tbFPType3 z = p;
    for (uint n = 0; n < params.boxIterations; n++)
    {
        boxFold(z);       // Reflect
        sphereFold(z,dr);    // Sphere Inversion

        z = z * params.boxScale + offset;  // Scale & Translate
        dr = dr * abs(params.boxScale) + tbFPType(1);
    }
    tbFPType r = length(z);
    return r/abs(dr);
}


/******************************************************************************
 *
 * Colouring functions
 *
 ******************************************************************************/



tbFPType4 HSVtoRGB(in int H, in tbFPType S, in tbFPType V)
{

    /**
     * adapted from
     * https://gist.github.com/kuathadianto/200148f53616cbd226d993b400214a7f
     */

    tbFPType4 outcolour;
    tbFPType C = S * V;
    tbFPType X = C * (tbFPType(1) - abs(mod(H / tbFPType(60), tbFPType(2) ) - tbFPType(1) ));
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

    outcolour.x = Rs + m;
    outcolour.y = Gs + m;
    outcolour.z = Bs + m;
    outcolour.w = tbFPType(1);

    return outcolour;
}

tbFPType4 getColour(tbFPType4 steps)
{
    tbFPType4 colour;

    tbFPType3 position = steps.xyz;

    tbFPType4 background = tbFPType4(params.bgRed,params.bgGreen,params.bgBlue,params.bgAlpha);

    if(steps.w == params.maxRaySteps)
    {
        return background;
    }
    else
    {

        tbFPType saturation = params.satValue;
        tbFPType hueVal = (position.z * params.hueFactor) + params.hueOffset;
        int hue = int(mod(hueVal, tbFPType(360) ) );
        hue = hue < 0 ? 360 + hue: hue;
        tbFPType value = params.valueRange*(tbFPType(1) - min( tbFPType( (steps.w*params.valueFactor) /params.maxRaySteps), params.valueClamp));

        colour = HSVtoRGB(hue, saturation, value);

//        colour.x = value;
//        colour.y = value;
//        colour.z = value;
//        colour.w = 1.0f;

        return colour;
    }
}

/******************************************************************************
 *
 * Ray marching function and entry kernel
 *
 ******************************************************************************/

tbFPType4 trace(uint x, uint y)
{
    /**
     * This function taken from
     * http://blog.hvidtfeldts.net/index.php/2011/06/distance-estimated-3d-fractals-part-i/
     */

    tbFPType totalDistance = 0.0;
    uint steps;

    tbFPType3 pixelPosition = cam.screenTopLeft + (cam.screenRight*x) + (cam.screenUp * y);

    tbFPType3 rayDir = pixelPosition - cam.pos;
    rayDir = normalize(rayDir);

    tbFPType3 p;
    for (steps=0; steps < params.maxRaySteps; steps++)
    {
        p = cam.pos + (rayDir * totalDistance);
        tbFPType distance = boxDist(p);
        totalDistance += distance;
        if (distance < params.collisionMinDist) break;
    }
    //return both the steps and the actual position in space for colouring purposes
    return tbFPType4(p, steps);
}

[ numthreads(16,16,1) ]
void traceRegion( uint3 tid : SV_DispatchThreadID)
{
  /*
  In order to fit the work into workgroups, some unnecessary threads are launched.
  We terminate those threads here.
  */
    if(tid.x >= cam.width || tid.y >= cam.height)
    {
        return;
    }

   // This is just here while I hope and wait for it to get implemented
   // if(tid.x == 0 && tid.y == 0)
   // {
   //     printf("Hey, printf works in HLSL as well!");
   // }

    uint col = tid.x;
    uint row = tid.y;
    uint index = ((row*cam.width)+col);

    data[index] = getColour(trace(col, row));
}
