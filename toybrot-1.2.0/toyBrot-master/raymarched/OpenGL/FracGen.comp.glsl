#version 450

#ifdef TOYBROT_VULKAN
    #extension GL_ARB_separate_shader_objects : enable
    #extension GL_EXT_debug_printf : enable
#endif

#define WORKGROUP_SIZE 16
layout (local_size_x = WORKGROUP_SIZE, local_size_y = WORKGROUP_SIZE, local_size_z = 1 ) in;


#ifdef TOYBROT_USE_DOUBLES
    #define tbFPType   double
    #define tbVecType3 dvec3
    #define tbVecType4 dvec4
#else
    #define tbFPType   float
    #define tbVecType3 vec3
    #define tbVecType4 vec4
#endif


// implementation independent mod
#define mod(x, y) ( x - y * trunc(x / y) )

/******************************************************************************
 *
 * Tweakable parameters
 *
 ******************************************************************************/

layout(binding = 0, std140) buffer outBuf
{
   tbVecType4 data[];
};

layout(binding = 1, std140) uniform Camera
{
    tbVecType3 camPos;
    tbVecType3 camUp;
    tbVecType3 camRight;
    tbVecType3 camTarget;

    tbFPType padding;

    tbFPType camNear;
    tbFPType camFovY;

    uint screenWidth;
    uint screenHeight;

    tbVecType3 screenTopLeft;
    tbVecType3 screenUp;
    tbVecType3 screenRight;
};

layout(binding = 2, std140) uniform Parameters
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

/******************************************************************************
 *
 * Distance estimator functions
 *
 ******************************************************************************/


void sphereFold(inout tbVecType3 z, inout tbFPType dz)
{
    tbFPType rsq = dot(z,z);
    /** Unlike CUDA and HIP, in Vulkan doing the
      * conditional shenanigans does not increase
      * the function's runtime. However, it doesn't
      * seem to decrease it either so there's not
      * really any point in doing it as all you're
      * left with is less readable code
      */
//    int cond1 = int(rsq < minRadiusSq);
//    int cond2 = int((rsq < fixedRadiusSq) && !bool(cond1));
//    int cond3 = int(!bool(cond1) && !bool(cond2));
//    tbFPType temp = ( (fixedRadiusSq/minRadiusSq) * cond1) + ( (fixedRadiusSq/rsq) * cond2) + cond3;
//    z *= temp;
//    dz *= temp;

    if ( rsq < minRadiusSq)
    {
        // linear inner scaling
        tbFPType temp = (fixedRadiusSq/minRadiusSq);
        z *= temp;
        dz *= temp;
    }
    else if(rsq < fixedRadiusSq )
    {
        // this is the actual sphere inversion
        tbFPType temp = (fixedRadiusSq/rsq);
        z *= temp;
        dz *= temp;
    }
}

void boxFold(inout tbVecType3 z)
{
    z = clamp(z, -foldingLimit, foldingLimit)* 2.0f - z;
}

tbFPType boxDist(in tbVecType3 p)
{
    /**
     * Distance estimator for a mandelbox
     *
     * Distance estimator adapted from
     * https://http://blog.hvidtfeldts.net/index.php/2011/11/distance-estimated-3d-fractals-vi-the-mandelbox/
     */
    tbVecType3 offset = p;
    tbFPType dr = boxScale;
    tbVecType3 z = p;
    for (uint n = 0u; n < boxIterations; n++)
    {
        boxFold(z);       // Reflect
        sphereFold(z,dr);    // Sphere Inversion

        z = z * boxScale + offset;  // Scale & Translate
        dr = dr * abs(boxScale) + 1.0f;
    }
    tbFPType r = length(z);
    return r/abs(dr);
}


/******************************************************************************
 *
 * Colouring functions
 *
 ******************************************************************************/



tbVecType4 HSVtoRGB(in int H, in tbFPType S, in tbFPType V)
{

    /**
     * adapted from
     * https://gist.github.com/kuathadianto/200148f53616cbd226d993b400214a7f
     */

    tbVecType4 outcolour;
    tbFPType C = S * V;
    tbFPType X = C * (1.0f - abs( mod(tbFPType(H) / 60.0f, 2.0f) - 1.0f));
    tbFPType m = V - C;
    tbFPType Rs, Gs, Bs;

    if(H >= 0 && H < 60)
    {
        Rs = C;
        Gs = X;
        Bs = 0.0f;
    }
    else if(H >= 60 && H < 120)
    {
        Rs = X;
        Gs = C;
        Bs = 0.0f;
    }
    else if(H >= 120 && H < 180)
    {
        Rs = 0.0f;
        Gs = C;
        Bs = X;
    }
    else if(H >= 180 && H < 240)
    {
        Rs = 0.0f;
        Gs = X;
        Bs = C;
    }
    else if(H >= 240 && H < 300)
    {
        Rs = X;
        Gs = 0.0f;
        Bs = C;
    }
    else {
        Rs = C;
        Gs = 0.0f;
        Bs = X;
    }

    outcolour.x = Rs + m;
    outcolour.y = Gs + m;
    outcolour.z = Bs + m;
    outcolour.w = 1.0f;

    return outcolour;
}

tbVecType4 getColour(tbVecType4 steps)
{
    tbVecType4 colour;

    tbVecType3 position = steps.xyz;

    tbVecType4 background = tbVecType4(bgRed,bgGreen,bgBlue,bgAlpha);

    if(uint(steps.w) == maxRaySteps)
    {
        return background;
    }
    else
    {

        tbFPType saturation = satValue;
        tbFPType hueVal = (position.z * hueFactor) + tbFPType(hueOffset);
        int hue = int(mod(hueVal, 360.0f) );
        hue = hue < 0 ? 360 + hue: hue;
        tbFPType value = valueRange*(1.0 - min( tbFPType( (steps.w*valueFactor) / tbFPType(maxRaySteps) ), valueClamp));

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

tbVecType4 trace(uint x, uint y)
{
    /**
     * This function taken from
     * http://blog.hvidtfeldts.net/index.php/2011/06/distance-estimated-3d-fractals-part-i/
     */

    tbFPType totalDistance = 0.0;
    uint steps;

    tbVecType3 pixelPosition = screenTopLeft + ( screenRight*tbFPType(x) ) + ( screenUp * tbFPType(y));

    tbVecType3 rayDir = pixelPosition - camPos;
    rayDir = normalize(rayDir);

    tbVecType3 p = tbVecType3(0,0,0);
    for (steps=0u; steps < maxRaySteps; steps++)
    {
        p = camPos + (rayDir * totalDistance);
        tbFPType distance = boxDist(p);
        totalDistance += distance;
        if (distance < collisionMinDist) break;
    }
    //return both the steps and the actual position in space for colouring purposes
    return tbVecType4(p, steps);
}


void main()
{

  /*
  In order to fit the work into workgroups, some unnecessary threads are launched.
  We terminate those threads here.
  */
    if(gl_GlobalInvocationID.x >= screenWidth|| gl_GlobalInvocationID.y >= screenHeight)
    {
        return;
    }


    uint col = gl_GlobalInvocationID.x;
    uint row = gl_GlobalInvocationID.y;
    uint index = ((row*uint(screenWidth))+col);

    #ifdef TOYBROT_DEBUG
        if(index == 0u)
        {
            #ifdef TOYBROT_VULKAN
                #ifdef TOYBROT_USE_DOUBLES

                    debugPrintfEXT("Vulkan Printf requires all types to be 32 bit long \n");
                    debugPrintfEXT("However it specifies doubles as 64 so it prints mostly garbage here \n");
                    debugPrintfEXT("See github.com/KhronosGroup/Vulkan-ValidationLayers/blob/master/docs/debug_printf.md for more info \n\n");

                #else

                    debugPrintfEXT("Shader-side camera information:\n");
                    debugPrintfEXT("camPos    = %f, %f, %f\n", camPos.x, camPos.y, camPos.z);
                    debugPrintfEXT("camUp     = %f, %f, %f\n", camUp.x, camUp.y, camUp.z);
                    debugPrintfEXT("camRight  = %f, %f, %f\n", camRight.x, camRight.y, camRight.z);
                    debugPrintfEXT("camTarget = %f, %f, %f\n", camTarget.x, camTarget.y, camTarget.z);

                    debugPrintfEXT("camNear   = %f\n", camNear);
                    debugPrintfEXT("camFovY   = %f\n", camFovY);

                    debugPrintfEXT("screenWidth   = %u\n", screenWidth);
                    debugPrintfEXT("screenHeight  = %u\n", screenHeight);

                    debugPrintfEXT("screenTL    = %f, %f, %f\n",   screenTopLeft.x, screenTopLeft.y, screenTopLeft.z);
                    debugPrintfEXT("screenUp    = %f, %f, %f\n",   screenUp.x, screenUp.y, screenUp.z);
                    debugPrintfEXT("screenRight = %f, %f, %f\n\n", screenRight.x, screenRight.y, screenRight.z);

                    debugPrintfEXT("Shader-side Parameters information:\n");
                    debugPrintfEXT("hueFactor   = %f\n", hueFactor);
                    debugPrintfEXT("hueOffset   = %i\n", hueOffset);
                    debugPrintfEXT("valueFactor = %f\n", valueFactor);
                    debugPrintfEXT("valueRange  = %f\n", valueRange);
                    debugPrintfEXT("valueClamp  = %f\n", valueClamp);
                    debugPrintfEXT("satValue    = %f\n", satValue);
                    debugPrintfEXT("bgRed       = %f\n", bgRed);
                    debugPrintfEXT("bgGreen     = %f\n", bgGreen);
                    debugPrintfEXT("bgBlue      = %f\n", bgBlue);
                    debugPrintfEXT("bgAlpha     = %f\n\n", bgAlpha);


                    debugPrintfEXT("maxRaySteps      = %u\n", maxRaySteps);
                    debugPrintfEXT("collisionMinDist = %f\n\n", collisionMinDist);


                    debugPrintfEXT("fixedRadiusSq = %f\n", fixedRadiusSq);
                    debugPrintfEXT("minRadiusSq   = %f\n", minRadiusSq);
                    debugPrintfEXT("foldingLimit  = %f\n", foldingLimit);
                    debugPrintfEXT("boxScale      = %f\n", boxScale);
                    debugPrintfEXT("boxIterations = %u\n\n", boxIterations);

                #endif //TOYBROT_USE_DOUBLES
            #else //TOYBROT_VULKAN (We're on OpenGL)

                data[0].xyz = camPos;
                data[1].xyz = camUp;
                data[2].xyz = camRight;
                data[3].xyz = camTarget;

                data[4].x = camNear;
                data[4].y = camFovY;

                data[4].z = float(screenWidth);
                data[4].w = float(screenHeight);

                data[5].xyz = screenTopLeft;
                data[6].xyz = screenUp;
                data[7].xyz = screenRight;

                data[8].x = hueFactor;
                data[8].y = float(hueOffset);
                data[8].z = valueFactor;
                data[8].w = valueRange;
                data[9].x = valueClamp;
                data[9].y = satValue;
                data[9].z = bgRed;
                data[9].w = bgGreen;
                data[10].x = bgBlue;
                data[10].y = bgAlpha;

                data[10].z = float(maxRaySteps);
                data[10].w = collisionMinDist;

                data[11].x = fixedRadiusSq;
                data[11].y = minRadiusSq;
                data[11].z = foldingLimit;
                data[11].w = boxScale;
                data[12].x = float(boxIterations);
                data[12].y = 0.0;
                data[12].w = 0.0;
                data[12].z = 0.0;
            #endif //TOYBROT_VULKAN

        }
        #endif //TOYBROT_DEBUG

        #if defined(TOYBROT_DEBUG) && !defined(TOYBROT_VULKAN)
        if(index > 12u)
        {
            data[index] = getColour(trace(col, row));
        }
    #else
        data[index] = getColour(trace(col, row));
    #endif
}
