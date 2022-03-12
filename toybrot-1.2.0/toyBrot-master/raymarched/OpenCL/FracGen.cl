
#if defined(TOYBROT_USE_DOUBLES)

    #define tbFPType  double
    #define tbFPType3 double3
    #define tbFPType4 double4

    #if !defined(TOYBROT_VULKAN)
        #if defined(cl_khr_fp64)
        #  pragma OPENCL EXTENSION cl_khr_fp64: enable
        #elif defined(cl_amd_fp64)
        #  pragma OPENCL EXTENSION cl_amd_fp64: enable
        #else
        #  error double precision is not supported
        #endif
    #endif

#else

    #define tbFPType  float
    #define tbFPType3 float3
    #define tbFPType4 float4

#endif

/******************************************************************************
 *
 * Data structs
 *
 ******************************************************************************/

/**
 * I REALLY wish I had found a way to get around all this super clunky
 * manual padding but, although the regular OpenCL compilers were happy
 * CLSPV was just not having a good time aligning this to what Vulkan
 * expects
 */

struct __attribute__ ((packed)) Camera
{
    tbFPType3 pos;
    tbFPType3 up;
    tbFPType3 right;
    tbFPType3 target;

    tbFPType near;
    tbFPType fovY;

    uint width;
    uint height;
    #if defined(TOYBROT_USE_DOUBLES) && defined(TOYBROT_VULKAN)
        uint doublepadding0;
        uint doublepadding1;
    #endif

    tbFPType3 screenTopLeft;
    tbFPType3 screenUp;
    tbFPType3 screenRight;
};


struct __attribute__ ((packed)) Parameters
{
    tbFPType hueFactor;
    int   hueOffset;
    #if defined(TOYBROT_USE_DOUBLES) && defined(TOYBROT_VULKAN)
        uint doublepadding0;
    #endif
    tbFPType valueFactor;
    tbFPType valueRange;
    tbFPType valueClamp;
    tbFPType satValue;
    tbFPType bgRed;
    tbFPType bgGreen;
    tbFPType bgBlue;
    tbFPType bgAlpha;

    uint  maxRaySteps;
    #if defined(TOYBROT_USE_DOUBLES) && defined(TOYBROT_VULKAN)
        uint doublepadding1;
    #endif
    tbFPType collisionMinDist;

    tbFPType fixedRadiusSq;
    tbFPType minRadiusSq;
    tbFPType foldingLimit;
    tbFPType boxScale;
    uint  boxIterations;

    #if defined(TOYBROT_VULKAN)
        uint ubopadding0;
        uint ubopadding1;
        uint ubopadding2;
        uint4 ubopadding3;
    #endif
};

/******************************************************************************
 *
 * Distance estimator functions
 *
 ******************************************************************************/


static void sphereFold(tbFPType3* z, tbFPType* dz, __constant struct Parameters* params)
{

    tbFPType rsq = dot(*z,*z);

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
//    int cond1 = rsq < minRadiusSq;
//    int cond2 = (rsq < fixedRadiusSq) * !cond1;
//    int cond3 = !cond1 * !cond2;
//    tbFPType temp = ( (fixedRadiusSq/minRadiusSq) * cond1) + ( (fixedRadiusSq/rsq) * cond2) + cond3;
//    *z *= temp;
//    *dz *= temp;
    if ( rsq < params->minRadiusSq)
    {
        // linear inner scaling
        tbFPType temp = (params->fixedRadiusSq/params->minRadiusSq);
        *z *= temp;
        *dz *= temp;
    }
    else if(rsq < params->fixedRadiusSq )
    {
        // this is the actual sphere inversion
        tbFPType temp =(params->fixedRadiusSq/rsq);
        *z *= temp;
        *dz *= temp;
    }
}

static void boxFold(tbFPType3* z, __constant struct Parameters* params)
{
    *z = clamp(*z, -params->foldingLimit, params->foldingLimit)* 2 - *z;
}

static tbFPType boxDist(tbFPType3 p, __constant struct Parameters* params)
{
    /**
     * Distance estimator for a mandelbox
     *
     * Distance estimator adapted from
     * https://http://blog.hvidtfeldts.net/index.php/2011/11/distance-estimated-3d-fractals-vi-the-mandelbox/
     */
    tbFPType3 offset = p;
    tbFPType dr = params->boxScale;
    tbFPType3 z = p;
    for (size_t n = 0; n < params->boxIterations; n++)
    {
        boxFold(&z, params);       // Reflect
        sphereFold(&z,&dr, params);    // Sphere Inversion
        z = z * params->boxScale + offset;  // Scale & Translate
        dr = dr * fabs(params->boxScale) + 1;
    }
    tbFPType r = length(z);
    return r/fabs(dr);
}


/******************************************************************************
 *
 * Colouring functions
 *
 ******************************************************************************/



static tbFPType4 HSVtoRGB(int H, tbFPType S, tbFPType V)
{

    /**
     * adapted from
     * https://gist.github.com/kuathadianto/200148f53616cbd226d993b400214a7f
     */

    tbFPType4 output;
    tbFPType C = S * V;
    tbFPType X = C * (1 - fabs(fmod((tbFPType)(H) / 60, 2) - 1));
    tbFPType m = V - C;
    tbFPType Rs, Gs, Bs;

    if(H < 60)
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

    output.x = (Rs + m);
    output.y = (Gs + m);
    output.z = (Bs + m);
    output.w = 1;

    return output;
}

static tbFPType4 getColour(tbFPType4 steps, __constant struct Parameters* params)
{
    tbFPType4 colour;

    tbFPType3 position = (tbFPType3)(steps.x,steps.y,steps.z);

    tbFPType4 background = (tbFPType4)(params->bgRed, params->bgGreen, params->bgBlue, params->bgAlpha);

    if((uint)(steps.w) >= params->maxRaySteps)
    {
        return background;
    }
    else
    {

        tbFPType saturation = params->satValue;
        tbFPType hueVal = (position.z * params->hueFactor) + (tbFPType)(params->hueOffset);
        int hue = (int)( trunc( fmod(hueVal,360 ) ) );
        hue = hue < 0 ? 360 + hue: hue;

        //uint hue = abs( ((int) (position.z * hueFactor) + hueOffset) % 360);
        tbFPType value = params->valueRange*(1 - min( (tbFPType)( (steps.w*params->valueFactor) / (tbFPType)(params->maxRaySteps) ), params->valueClamp));

        colour = HSVtoRGB(hue, saturation, value);

//        Simplest colouring, based only on steps (roughly distance from camera)

//        colour.x = value;
//        colour.y = value;
//        colour.z = value;
//        colour.w = 1.0f;

        return colour;
    }
}


/******************************************************************************
 *
 * Ray marching functions and entry kernel
 *
 ******************************************************************************/

static tbFPType4 trace( __constant struct Camera* cam, __constant struct Parameters* params, uint x, uint y)
{
    /**
     * This function taken from
     * http://blog.hvidtfeldts.net/index.php/2011/06/distance-estimated-3d-fractals-part-i/
     */

    tbFPType totalDistance = 0;
    unsigned int steps;

    tbFPType3 pixelPosition = cam->screenTopLeft + (cam->screenRight * (tbFPType)(x) ) + (cam->screenUp* (tbFPType)(y) );

    tbFPType3 rayDir = pixelPosition - cam->pos;
    rayDir = normalize(rayDir);

    tbFPType3 p = (tbFPType3)(0,0,0);
    for (steps=0; steps < params->maxRaySteps; steps++)
    {
        p = cam->pos + (rayDir * totalDistance);
        tbFPType distance = boxDist(p, params);
        totalDistance += distance;
        if (distance < params->collisionMinDist) break;
    }
    //return both the steps and the actual position in space for colouring purposes
    return (tbFPType4)(p, (tbFPType)(steps));
}


kernel __attribute__((reqd_work_group_size(16, 16, 1)))
void traceRegion(  __global   tbFPType4* data,
                   __constant struct Camera* cam,
                   __constant struct Parameters* params)
{

    uint col = get_global_id (0);
    uint row = get_global_id (1);
    uint index = ((row*cam->width)+col);

#if defined(TOYBROT_DEBUG) && !defined(TOYBROT_VULKAN)
    if (col == 0 && row == 0)
    {
        printf("Kernel-side camera information:\n");
        printf("cam->pos    = %f, %f, %f\n",   cam->pos.x, cam->pos.y, cam->pos.z);
        printf("cam->up     = %f, %f, %f\n",   cam->up.x, cam->up.y, cam->up.z);
        printf("cam->right  = %f, %f, %f\n",   cam->right.x, cam->right.y, cam->right.z);
        printf("cam->target = %f, %f, %f\n\n", cam->target.x, cam->target.y, cam->target.z);

        printf("cam->near   = %f\n",   cam->near);
        printf("cam->fovY   = %f\n\n", cam->fovY);

        printf("cam->width    = %u\n",   (uint)(cam->width) );
        printf("cam->height   = %u\n\n", (uint)(cam->height) );

        printf("cam->screenTL    = %f, %f, %f\n",   cam->screenTopLeft.x, cam->screenTopLeft.y, cam->screenTopLeft.z);
        printf("cam->screenUp    = %f, %f, %f\n",   cam->screenUp.x, cam->screenUp.y, cam->screenUp.z);
        printf("cam->screenRight = %f, %f, %f\n\n", cam->screenRight.x, cam->screenRight.y, cam->screenRight.z);

        printf("Kernel-side Parameter information:\n");

        printf("params->hueFactor   = %f\n",   params->hueFactor);
        printf("params->hueOffset   = %i\n",   (int)(params->hueOffset));
        printf("params->valueFactor = %f\n",   params->valueFactor);
        printf("params->valueRange  = %f\n",   params->valueRange);
        printf("params->valueClamp  = %f\n",   params->valueClamp);
        printf("params->satValue    = %f\n",   params->satValue);
        printf("params->bgRed       = %f\n",   params->bgRed);
        printf("params->bgGreen     = %f\n",   params->bgGreen);
        printf("params->bgBlue      = %f\n",   params->bgBlue);
        printf("params->bgAlpha     = %f\n\n", params->bgAlpha);

        printf("params->maxRaySteps      = %u\n",   (uint)(params->maxRaySteps));
        printf("params->collisionMinDist = %f\n\n", params->collisionMinDist);

        printf("params->fixedRadiusSq = %f\n",   params->fixedRadiusSq);
        printf("params->minRadiusSq   = %f\n",   params->minRadiusSq);
        printf("params->foldingLimit  = %f\n",   params->foldingLimit);
        printf("params->boxScale      = %f\n",   params->boxScale);
        printf("params->boxIterations = %u\n\n", (uint)(params->boxIterations));

    }
#endif

    if (col >= cam->width || index > (cam->width*cam->height))
    {
        return;
    }



    //Not doing the functional shenanigans from STL here
    data[index] = getColour(trace(cam, params, col, row), params);

}
