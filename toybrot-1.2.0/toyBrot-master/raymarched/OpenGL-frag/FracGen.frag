#version 450

//#define WORKGROUP_SIZE 16
//layout (local_size_x = WORKGROUP_SIZE, local_size_y = WORKGROUP_SIZE, local_size_z = 1 ) in;

precision highp float;

#ifdef TOYBROT_USE_DOUBLES
    precision highp double;
    #define tbFPType   double
    #define tbVecType3 dvec3
    #define tbVecType4 dvec4
#else
    #define tbFPType   float
    #define tbVecType3 vec3
    #define tbVecType4 vec4
#endif


// implementation independent mod
//#define mod(x, y) ( x - y * trunc(x / y) )

/******************************************************************************
 *
 * Tweakable parameters
 *
 ******************************************************************************/


struct Camera
{
    tbVecType3 camPos;
    tbVecType3 camUp;
    tbVecType3 camRight;
    tbVecType3 camTarget;

    //tbFPType padding;

    tbFPType camNear;
    tbFPType camFovY;

    uint screenWidth;
    uint screenHeight;

    tbVecType3 screenTopLeft;
    tbVecType3 screenUp;
    tbVecType3 screenRight;
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

out tbVecType4 outColour;

in vec2 UV;

uniform Camera cam;
uniform Parameters params;

/******************************************************************************
 *
 * Distance estimator functions
 *
 ******************************************************************************/


void sphereFold(inout tbVecType3 z, inout tbFPType dz)
{
    tbFPType rsq = dot(z,z);

    if ( rsq < params.minRadiusSq)
    {
        // linear inner scaling
        tbFPType temp = (params.fixedRadiusSq/params.minRadiusSq);
        z *= temp;
        dz *= temp;
    }
    else
    {
        if(rsq < params.fixedRadiusSq )
        {
            // this is the actual sphere inversion
            tbFPType temp = (params.fixedRadiusSq/rsq);
            z *= temp;
            dz *= temp;
        }
    }
}

void boxFold(inout tbVecType3 z)
{
    z = clamp(z, -params.foldingLimit, params.foldingLimit)* 2.0f - z;
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
    tbFPType dr = params.boxScale;
    tbVecType3 z = p;
    for (uint n = 0u; n < params.boxIterations; n++)
    {
        boxFold(z);       // Reflect
        sphereFold(z,dr);    // Sphere Inversion

        z = z * params.boxScale + offset;  // Scale & Translate
        dr = dr * abs(params.boxScale) + 1.0f;
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
    else
    {
        if(H >= 60 && H < 120)
        {
            Rs = X;
            Gs = C;
            Bs = 0.0f;
        }
        else
        {

            if(H >= 120 && H < 180)
            {
                Rs = 0.0f;
                Gs = C;
                Bs = X;
            }
            else
            {
                if(H >= 180 && H < 240)
                {
                    Rs = 0.0f;
                    Gs = X;
                    Bs = C;
                }
                else
                {
                    if(H >= 240 && H < 300)
                    {
                        Rs = X;
                        Gs = 0.0f;
                        Bs = C;
                    }
                    else
                    {
                        Rs = C;
                        Gs = 0.0f;
                        Bs = X;
                    }
                }
            }
        }
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

    tbVecType4 background = tbVecType4(params.bgRed,params.bgGreen,params.bgBlue,params.bgAlpha);

    if(uint(steps.w) == params.maxRaySteps)
    {
        return background;
    }
    else
    {

        tbFPType saturation = params.satValue;
        tbFPType hueVal = (position.z * params.hueFactor) + tbFPType(params.hueOffset);
        int hue = int(mod(hueVal, 360.0f) );
        hue = hue < 0 ? 360 + hue: hue;
        tbFPType value = params.valueRange * (1.0 - min( tbFPType( (steps.w*params.valueFactor) / tbFPType(params.maxRaySteps) ), params.valueClamp ));

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

    tbVecType3 pixelPosition = cam.screenTopLeft + ( cam.screenRight*tbFPType(x) ) + ( cam.screenUp * tbFPType(y));

    tbVecType3 rayDir = pixelPosition - cam.camPos;
    rayDir = normalize(rayDir);

    tbVecType3 p = tbVecType3(0,0,0);
    for (steps=0u; steps < params.maxRaySteps; steps++)
    {
        p = cam.camPos + (rayDir * totalDistance);
        tbFPType distance = boxDist(p);
        totalDistance += distance;
        if (distance < params.collisionMinDist) break;
    }
    //return both the steps and the actual position in space for colouring purposes
    return tbVecType4(p, steps);
}


void main()
{
    outColour = getColour(trace(uint(UV.x * float(cam.screenWidth)),uint( (UV.y) * float(cam.screenHeight))))  ;
}
