
#ifdef TOYBROT_USE_DOUBLES
    #define tbFPType double
#else
    #define tbFPType float
#endif

struct Camera
{
    tbFPType pos[3];
    tbFPType up[3];
    tbFPType right[3];
    tbFPType targ[3];

    tbFPType near;
    tbFPType fov;

    unsigned int width;
    unsigned int height;

    tbFPType screenTopLeft[3];
    tbFPType screenUp[3];
    tbFPType screenRight[3];
};

struct Parameters
{
    tbFPType     hueFactor;
    int          hueOffset;
    tbFPType     valueFactor;
    tbFPType     valueRange;
    tbFPType     valueClamp;
    tbFPType     satValue;
    tbFPType     bgRed;
    tbFPType     bgGreen;
    tbFPType     bgBlue;
    tbFPType     bgAlpha;

    unsigned int maxRaySteps;
    tbFPType     collisionMinDist;

    tbFPType     fixedRadiusSq;
    tbFPType     minRadiusSq;
    tbFPType     foldingLimit;
    tbFPType     boxScale;
    unsigned int boxIterations;
};


static inline tbFPType sqMod(tbFPType v[3])
{
    return v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
}

static inline tbFPType length(tbFPType v[3])
{
    return sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

static inline void normalise(tbFPType v[3])
{
    tbFPType l = length(v);
    if(l == 0)
    {
        return;
    }
    v[0] /= l;
    v[1] /= l;
    v[2] /= l;
}
static inline tbFPType fmod(const tbFPType a, const tbFPType b)
{
    return a - (b*floor(a/b));
}

/******************************************************************************
 *
 * Distance estimator functions and helpers
 *
 ******************************************************************************/


static void sphereFold(tbFPType z[],  tbFPType& dz, uniform const Parameters& params)
{
    tbFPType r2 = sqMod(z);

    if ( r2 < params.minRadiusSq)
    {
        // linear inner scaling
        tbFPType temp = (params.fixedRadiusSq/params.minRadiusSq);
        z[0]  *= temp;
        z[1]  *= temp;
        z[2]  *= temp;
        dz *= temp;
    }
    else if( r2 < params.fixedRadiusSq)
    {
        // this is the actual sphere inversion
        tbFPType temp = ( params.fixedRadiusSq/r2 );
        z[0]  *= temp;
        z[1]  *= temp;
        z[2]  *= temp;
        dz *= temp;
    }
}

static void boxFold(tbFPType z[3], uniform const Parameters& params)
{
    z[0] = clamp(z[0], -params.foldingLimit, params.foldingLimit)* 2.0f - z[0];
    z[1] = clamp(z[1], -params.foldingLimit, params.foldingLimit)* 2.0f - z[1];
    z[2] = clamp(z[2], -params.foldingLimit, params.foldingLimit)* 2.0f - z[2];
}

static tbFPType boxDist(tbFPType z[3], uniform const Parameters& params)
{
    /**
     * Distance estimator for a mandelbox
     *
     * Distance estimator adapted from
     * https://http://blog.hvidtfeldts.net/index.php/2011/11/distance-estimated-3d-fractals-vi-the-mandelbox/
     */
    tbFPType offset[3] = {z[0],z[1],z[2]};
    tbFPType dr = params.boxScale;
    tbFPType p[3] = {z[0],z[1],z[2]};
    for (size_t n = 0; n < params.boxIterations; n++)
    {
        boxFold(p, params);       // Reflect
        sphereFold(p, dr, params);    // Sphere Inversion
        p[0] *= params.boxScale;
        p[1] *= params.boxScale;
        p[2] *= params.boxScale;
        p[0] += offset[0];
        p[1] += offset[1];
        p[2] += offset[2];
        dr = (dr * params.boxScale) + 1.0f;
    }
    tbFPType r = length(p);
    return r/abs(dr);
}


static tbFPType bulbDist(tbFPType p[3], uniform const Parameters& params)
{

    /**
     * Distance estimator for a mandelbulb
     *
     * Distance estimator adapted from
     * https://www.iquilezles.org/www/articles/mandelbulb/mandelbulb.htm
     * https://www.shadertoy.com/view/ltfSWn
     */

    tbFPType w[3] = {p[0],p[1],p[2]};
    tbFPType m = sqMod(w);

    //vec4 trap = vec4(abs(w),m);
    tbFPType dz = 3.0f;


    for( int i=0; i<4; i++ )
    {
#if 1
        tbFPType m2 = m*m;
        tbFPType m4 = m2*m2;
        dz = 8.0f*sqrt(m4*m2*m)*dz + 1.0f;

        tbFPType x = w[0]; tbFPType x2 = x*x; tbFPType x4 = x2*x2;
        tbFPType y = w[1]; tbFPType y2 = y*y; tbFPType y4 = y2*y2;
        tbFPType z = w[2]; tbFPType z2 = z*z; tbFPType z4 = z2*z2;

        tbFPType k3 = x2 + z2;
        tbFPType k2 = 1/sqrt( k3*k3*k3*k3*k3*k3*k3 );
        tbFPType k1 = x4 + y4 + z4 - 6.0f*y2*z2 - 6.0f*x2*y2 + 2.0f*z2*x2;
        tbFPType k4 = x2 - y2 + z2;

        w[0] = p[0] +  64.0f*x*y*z*(x2-z2)*k4*(x4-6.0f*x2*z2+z4)*k1*k2;
        w[1] = p[1] + -16.0f*y2*k3*k4*k4 + k1*k1;
        w[2] = p[2] +  -8.0f*y*k4*(x4*x4 - 28.0f*x4*x2*z2 + 70.0f*x4*z4 - 28.0f*x2*z2*z4 + z4*z4)*k1*k2;
#else
        dz = 8.0*pow(sqrt(m),7.0)*dz + 1.0;
        //dz = 8.0*pow(m,3.5)*dz + 1.0;

        tbFPType r = w.mod();
        tbFPType b = 8.0*acos( w.Y()/r);
        tbFPType a = 8.0*atan2( w.X(), w.Z() );
        w = p + Vec3f( sin(b)*sin(a), cos(b), sin(b)*cos(a) ) * pow(r,8.0);
#endif

       // trap = min( trap, vec4(abs(w),m) );

        m = sqMod(w);
        if( m > 256.0f )
            break;
    }

    return 0.25f*log(m)*sqrt(m)/dz;
}

static inline tbFPType sphereDist(tbFPType p[3], uniform const Parameters& params)
{
    uniform tbFPType radius = 1.f;
    return length(p) - radius;
}

/******************************************************************************
 *
 * Coulouring functions and helpers
 *
 ******************************************************************************/

static void HSVtoRGB(int H, tbFPType S, tbFPType V, tbFPType colour[4])
{

    /**
     * adapted from
     * https://gist.github.com/kuathadianto/200148f53616cbd226d993b400214a7f
     */

    tbFPType C = S * V;
    tbFPType X = C * (1 - abs(fmod(H/(tbFPType)60.0, (tbFPType)2.0) - 1) );
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

    colour[0] = Rs + m;
    colour[1] = Gs + m;
    colour[2] = Bs + m;
    colour[3] = 1.0f ;
}

static void getColour(tbFPType steps[4], tbFPType out[4], uniform const Parameters& params)
{
    if(steps[3] == params.maxRaySteps)
    {
        out[0] = params.bgRed;
        out[1] = params.bgGreen;
        out[2] = params.bgBlue;
        out[3] = params.bgAlpha;
        return;
    }

    tbFPType position[3] = {steps[0],steps[1],steps[2]};

    tbFPType saturation = params.satValue;
    tbFPType hueVal = (position[2] * params.hueFactor) + params.hueOffset;
    int hue = (int)( round(fmod(hueVal, 360.0f) ) );
    hue = hue < 0 ? 360 + hue: hue;
    tbFPType value = params.valueRange*((tbFPType)1.0 - min(steps[3]*params.valueFactor/params.maxRaySteps, params.valueClamp));

    HSVtoRGB(hue, saturation, value, out);
//    out[0] = value;
//    out[1] = value;
//    out[2] = value;
//    out[3] = 1.0f;
}



/******************************************************************************
 *
 * Ray marching functions and helpers
 *
 ******************************************************************************/

static void trace(  uniform const Camera& cam,
                    uniform const Parameters& params,
                    const int x, const int y, tbFPType out[4])
{
    /**
     * This function taken from
     * http://blog.hvidtfeldts.net/index.php/2011/06/distance-estimated-3d-fractals-part-i/
     */

    tbFPType totalDistance = 0.0f;
    unsigned int steps = 0;

    tbFPType pixelPosition[3] = { cam.screenTopLeft[0] + ((cam.screenRight[0] * (tbFPType)x)) + (cam.screenUp[0] * (tbFPType)y) ,
                                  cam.screenTopLeft[1] + ((cam.screenRight[1] * (tbFPType)x)) + (cam.screenUp[1] * (tbFPType)y) ,
                                  cam.screenTopLeft[2] + ((cam.screenRight[2] * (tbFPType)x)) + (cam.screenUp[2] * (tbFPType)y) };

    tbFPType rayDir[3] = { pixelPosition[0] - cam.pos[0],
                           pixelPosition[1] - cam.pos[1],
                           pixelPosition[2] - cam.pos[2]};

    normalise(rayDir);

    varying tbFPType p[3] = {0,0,0};
    for (steps=0; steps < params.maxRaySteps; steps++)
    {
        p[0] = cam.pos[0] + (rayDir[0] * totalDistance);
        p[1] = cam.pos[1] + (rayDir[1] * totalDistance);
        p[2] = cam.pos[2] + (rayDir[2] * totalDistance);
        tbFPType distance = boxDist(p, params);
        totalDistance += distance;
        if (distance < params.collisionMinDist)
        {
            break;
        }
    }

    //return both the steps and the actual position in space for colouring purposes

    out[0] = p[0];
    out[1] = p[1];
    out[2] = p[2];
    out[3] = (tbFPType)((int)steps);
}

export uniform int gangSize()
{
    return programCount;
}

enum LoopType { FOREACH = 0, TILED = 1, STRIDED = 2 };

export uniform bool traceRegion(uniform tbFPType data[][4],
                                uniform const Camera& cam,
                                uniform const Parameters& params,
                                uniform const int h0,
                                uniform const int heightStep,
                                uniform const int idx,
                                uniform const int numTasks,
                                uniform const LoopType mode)
{
    // Even though logically most of the parameters should be
    // unsigned, ISPC is very particular about casting and the
    // indexes in the foreach loop are signed int.
    uniform bool done = false;
    uniform int taskWidth = ceil((tbFPType)cam.width/(tbFPType)numTasks);
    uniform int wStart = taskWidth * idx;
    uniform int wEnd   = min(wStart + taskWidth, (int)cam.width);
    uniform int hEnd   = min(h0 + heightStep , (int)cam.height);
    uniform int idx0   = h0*cam.width + (idx * programCount);
    //bounds check already part of calculation of hEnd
    uniform int idxEnd = hEnd * cam.width;

    /**
      *
      * I've left in three different approaches to splitting
      * each worker's workload within ISPC:
      *
      * FOREACH and TILED are pretty much the same code, except
      * that they change which loop function is called. TILED is
      * the faster and the default. For more information on them
      * please refer to ISPC's documentation. The TL;DR: is that
      * the difference between them is in how to try and split
      * the workload.
      *
      * Finally, STRIDED instead replicates the same strided access
      * I do for my STL implementations, except this time, each
      * instance is now a gang rather than one solitary calculation,
      * so it uses a regular for loop in ISPC's "manual mode"
      */
    switch (mode)
    {
        case FOREACH:
            foreach(h = h0 ... hEnd, w = wStart ... wEnd)
            {
                int pix = ((h*cam.width)+w);

                varying tbFPType traceOut[4] = {0,0,0,0};
                varying tbFPType colour[4] = {0,0,0,0};
                trace(cam, params, w, h, traceOut);
                getColour(traceOut, colour, params);
                data[pix][0] = colour[0];
                data[pix][1] = colour[1];
                data[pix][2] = colour[2];
                data[pix][3] = colour[3];
            }
            break;
        case TILED:
            foreach_tiled(h = h0 ... hEnd, w = wStart ... wEnd)
            {
                int pix = ((h*cam.width)+w);

                varying tbFPType traceOut[4] = {0,0,0,0};
                varying tbFPType colour[4] = {0,0,0,0};
                trace(cam, params, w, h, traceOut);
                getColour(traceOut, colour, params);
                data[pix][0] = colour[0];
                data[pix][1] = colour[1];
                data[pix][2] = colour[2];
                data[pix][3] = colour[3];
            }
            break;
        case STRIDED:
            for(uniform unsigned int i = idx0; i < (unsigned int)idxEnd; i += programCount*numTasks)
            {
                unsigned int pix = i+programIndex;

                if (pix >= idxEnd)
                {
                    continue;
                }

                varying tbFPType traceOut[4] = {0,0,0,0};
                varying tbFPType colour[4] = {0,0,0,0};
                trace(cam, params, pix%cam.width, pix/cam.width, traceOut);
                getColour(traceOut, colour, params);
                data[pix][0] = colour[0];
                data[pix][1] = colour[1];
                data[pix][2] = colour[2];
                data[pix][3] = colour[3];

            }
            break;
    }

    return hEnd == cam.height;
}




