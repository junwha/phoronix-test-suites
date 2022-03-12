#ifndef _TOYBROT_DATA_TYPES_HXX_DEFINED_
#define _TOYBROT_DATA_TYPES_HXX_DEFINED_

#include <cmath>
#include <cfloat>
#include <iostream>

#include <vector>
#include <string>
#include <memory>
#include <atomic>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <typeinfo>
#include <fstream>
#include <sstream>


#if defined(TB_USE_STDFS)
    #include <filesystem>
    namespace tbfs = std::filesystem;
#elif defined(TB_USE_EXPFS)
    #include <experimental/filesystem>
    namespace tbfs = std::experimental::filesystem;
#endif

#include "defines.hpp"
#include "Vec.hxx"

/**
  * This file defines 2 important structs used in ToyBrot
  *
  * These structs are templatised since some of the variants
  * use doubles instead of floats and I want to have those
  * implementations not diverge
  *
  * Camera -> represents a "camera" in space. it also  describes
  *           a rectangle that represents the "screen window" in
  *           space. This rectangle is placed at the near plane
  *           from the camera and is used to calculate the
  *           directions of rays for the raymarching algorithm
  *
  * Parameters -> ToyBrot has a series of tweakable parameters,
  *               these are in three categories:
  *                 - Camera position and direction information
  *                 - Colouring parameters
  *                 - Variables which control how the mandelbox is generated
  *
  *               This struct contains that information so that
  *               it can be easily passed around
  */


template<typename T> class Camera
{
    public:


#ifdef _TB_CUDA_HIP_
        _TB_DEV_ Camera(){}
#else
        Camera()
            : pos{0,0,0}
            , up{0,0,0}
            , right{0,0,0}
            , targ{0,0,0}
            , near{0}
            , fov{0}
            , width{0}
            , height{0}
            , screenTopLeft{0,0,0}
            , screenUp{0,0,0}
            , screenRight{0,0,0} {}
#endif
        _TB_DUAL_ Camera(Vec3<T> position, Vec3<T> target, uint32_t screenWidth, uint32_t screenHeight, T screenDist, T fovY)
            : pos{position}
            , targ{target}
            , width{screenWidth}
            , height{screenHeight}
            , near{screenDist}
            , fov{fovY}
        {
            Vec3<T> camDir = (targ - pos).normalise();
            right = crossProd(Vec3<T>(0,1,0), camDir).normalise();
            up    = crossProd(camDir, right).normalise();
            updateScreen();
        }
        _TB_DUAL_ inline const Vec3<T>& Pos()    const {return pos;  }
        _TB_DUAL_ inline const Vec3<T>& Up()     const {return up;   }
        _TB_DUAL_ inline const Vec3<T>& Right()  const {return right;}
        _TB_DUAL_ inline const Vec3<T>& Target() const {return targ; }

        _TB_DUAL_ inline const T Near() const {return near;}
        _TB_DUAL_ inline const T FovY() const {return fov;}

        _TB_DUAL_ inline uint32_t ScreenWidth() const  {return width;}
        _TB_DUAL_ inline uint32_t ScreenHeight() const {return height;}
        _TB_HOST_ inline void   setScreenWidth(uint32_t w) {width  = w;}
        _TB_HOST_ inline void   setScreenHeight(uint32_t h) {height = h;}

        _TB_DUAL_ inline const Vec3<T>& ScreenTopLeft() const {return screenTopLeft;}
        _TB_DUAL_ inline const Vec3<T>& ScreenUp()      const {return screenUp;     }
        _TB_DUAL_ inline const Vec3<T>& ScreenRight()   const {return screenRight;  }
        _TB_HOST_ void updateScreen()
        {
            Vec3<T> screenPlaneOrigin{pos + ((targ - pos).normalise()*near)};
            T screenPlaneHeight = std::abs(2*(near*sin(fov/2)));
            T screenPlaneWidth = screenPlaneHeight * (static_cast<T>(width)/static_cast<T>(height));

            screenUp = up * (static_cast<T>(-1.0) * screenPlaneHeight / height);
            screenRight = right * (screenPlaneWidth / width);
            screenTopLeft = screenPlaneOrigin + (right * (-screenPlaneWidth/2)) + (up * (screenPlaneHeight/2));
        }

        _TB_HOST_ void Print() const
        {
            std::cout << "Camera.pos    -> " << Pos().X() << ", " << Pos().Y() << ", " << Pos().Z() << std::endl;
            std::cout << "Camera.up     -> " << up.X() << ", " << up.Y() << ", " << up.Z() << std::endl;
            std::cout << "Camera.right  -> " << right.X() << ", " << right.Y() << ", " << right.Z() << std::endl;
            std::cout << "Camera.target -> " << targ.X() << ", " << targ.Y() << ", " << targ.Z() << std::endl;
            std::cout << std::endl;
            std::cout << "Camera.near   -> " << near << std::endl;
            std::cout << "Camera.fovY   -> " << fov << std::endl;
            std::cout << "Screen.width  -> " << width << std::endl;
            std::cout << "Screen.height -> " << height << std::endl;
            std::cout << std::endl;
            std::cout << "ScreenTopLeft -> " << screenTopLeft.X() << ", " << screenTopLeft.Y() << ", " << screenTopLeft.Z() << std::endl;
            std::cout << "Screen.up     -> " << screenUp.X() << ", " << screenUp.Y() << ", " << screenUp.Z() << std::endl;
            std::cout << "Screen.right  -> " << screenRight.X() << ", " << screenRight.Y() << ", " << screenRight.Z() << std::endl << std::endl;
        }

    private:        

        Vec3<T> pos;
        Vec3<T> up;
        Vec3<T> right;
        Vec3<T> targ;

        T near;
        T fov;

        uint32_t width;
        uint32_t height;

        /*
        * Screen Up and Right represent the vertical and horizontal
        * directions on the screen (i.e: screenUp "translates" the
        * Y direction of the screen to the direction that travels "up"
        * on the rectangle in space which represents it)
        */
        Vec3<T> screenTopLeft;
        Vec3<T> screenUp;
        Vec3<T> screenRight;

};

template <typename T>
class Parameters
{
    public:


#ifdef _TB_CUDA_HIP_
        _TB_DEV_ Parameters(){}
#else
        Parameters()
            :maxRaySteps{0}
            ,collisionMinDist{0}
            ,hueFactor{0}
            ,hueOffset{0}
            ,valueFactor{0}
            ,valueRange{0}
            ,valueClamp{0}
            ,satValue{0}
            ,bgRed{0}
            ,bgGreen{0}
            ,bgBlue{0}
            ,bgAlpha{0}
            ,fixedRadiusSq{0}
            ,minRadiusSq{0}
            ,foldingLimit{0}
            ,boxScale{0}
            ,boxIterations{0} {}
#endif
        _TB_DUAL_ Parameters  ( uint32_t steps
                              , T        minDist
                              , T        hueFac
                              , int      hueOff
                              , T        valFac
                              , T        valRng
                              , T        valCmp
                              , T        satVal
                              , T        bgR
                              , T        bgG
                              , T        bgB
                              , T        bgA
                              , T        frSq
                              , T        mrSq
                              , T        fLim
                              , T        bScale
                              , uint32_t bIter )
                    :hueFactor{hueFac}
                    ,hueOffset{hueOff}
                    ,valueFactor{valFac}
                    ,valueRange{valRng}
                    ,valueClamp{valCmp}
                    ,satValue{satVal}
                    ,bgRed{bgR}
                    ,bgGreen{bgG}
                    ,bgBlue{bgB}
                    ,bgAlpha{bgA}
                    ,maxRaySteps{steps}
                    ,collisionMinDist{minDist}
                    ,fixedRadiusSq{frSq}
                    ,minRadiusSq{mrSq}
                    ,foldingLimit{fLim}
                    ,boxScale{bScale}
                    ,boxIterations{bIter} {}


        /*
         * Colouring parameters
         */
        _TB_DUAL_ T HueFactor() const              {return hueFactor;}
        _TB_DUAL_ int32_t HueOffset() const        {return hueOffset;}
        _TB_DUAL_ T ValueFactor() const            {return valueFactor;}
        _TB_DUAL_ T ValueRange() const             {return valueRange;}
        _TB_DUAL_ T ValueClamp() const             {return valueClamp;}
        _TB_DUAL_ T SatValue() const               {return satValue;}
        //_TB_DUAL_ const Vec4<T>& BgValue() const   {return bgValue;}
        _TB_DUAL_ T BgRed() const                  {return bgRed;}
        _TB_DUAL_ T BgGreen() const                {return bgGreen;}
        _TB_DUAL_ T BgBlue() const                 {return bgBlue;}
        _TB_DUAL_ T BgAlpha() const                {return bgAlpha;}

        /*
         * Raymarching parameters
         */
        _TB_DUAL_ uint32_t MaxRaySteps() const     {return maxRaySteps;}
        _TB_DUAL_ T CollisionMinDist() const       {return collisionMinDist;}

        /*
         * Mandelbox parameters
         */
        _TB_DUAL_ T FixedRadiusSq() const          {return fixedRadiusSq;}
        _TB_DUAL_ T MinRadiusSq() const            {return minRadiusSq;}
        _TB_DUAL_ T FoldingLimit() const           {return foldingLimit;}
        _TB_DUAL_ T BoxScale() const               {return boxScale;}
        _TB_DUAL_ uint32_t BoxIterations() const   {return boxIterations;}

        //setters shouldn't be called from device code and are undecorated on purpose
        void setMaxRaySteps(const uint32_t &value)  {maxRaySteps        = value;}
        void setCollisionMinDist(const T &value)    {collisionMinDist   = value;}

        void setHueFactor(const T &value)           {hueFactor          = value;}
        void setHueOffset(int32_t value)            {hueOffset          = value;}
        void setValueFactor(const T &value)         {valueFactor        = value;}
        void setValueRange(const T &value)          {valueRange         = value;}
        void setValueClamp(const T &value)          {valueClamp         = value;}
        void setSatValue(const T &value)            {satValue           = value;}
        //void setBgValue(const Vec4<T> &value)       {bgValue            = value;}
        void setBgValue(T r, T g, T b, T a)         {bgRed = r; bgGreen = g; bgBlue = b; bgAlpha =a;}

        void setFixedRadiusSq(const T &value)        {fixedRadiusSq      = value;}
        void setMinRadiusSq(const T &value)          {minRadiusSq        = value;}
        void setFoldingLimit(const T &value)         {foldingLimit       = value;}
        void setBoxScale(const T &value)             {boxScale           = value;}
        void setBoxIterations(const uint32_t &value) {boxIterations      = value;}

    private:

        // coulouring parameters

        T       hueFactor;
        int32_t hueOffset;
        T       valueFactor;
        T       valueRange;
        T       valueClamp;
        T       satValue;
        T       bgRed;
        T       bgGreen;
        T       bgBlue;
        T       bgAlpha;

        // Mandelbox constants

        uint32_t maxRaySteps;
        T        collisionMinDist;

        T        fixedRadiusSq;
        T        minRadiusSq;
        T        foldingLimit;
        T        boxScale;
        uint32_t boxIterations;
};


namespace toyBrot
{
    template <typename T> std::string valString(T val)
    {
        std::string s = std::to_string(val);
        auto pos = s.find('.');
        auto trimMin = pos + 2;
        if(pos != std::string::npos && (s.length() > trimMin) )
        {
            //We might need to trim some zeroes
            // +1 is because substring is [start,end)
            size_t trimPos = s.find_last_not_of('0')+1;
            if(trimPos < trimMin )
            {
                trimPos = trimMin;
            }
            s = s.substr(0,trimPos);
        }
        return s;
    }
    template <typename T>
    bool loadConfig(std::shared_ptr< Camera<T> > cam, std::shared_ptr < Parameters<T> > params, std::string filename)
    {
        /**
         * In the interest of keeping my assumptions mostly well contained I
         * decided to refrain from doing anything clever like enforcing and
         * relying on a certain memory layout to more straightforwardly structure
         * what is really a (de)serialization problem in a slicker more concise
         * manner. Instead this this is largely done in a pretty frustatringly
         * dumb way with a bunch of ifs for specific things. It's both sad to
         * look at and painful to Ctrl+V over and over but it's kind of the
         * price I've chosen to pay here  to protect me a bit
         * in case I ever decide to change the structs in question
         */

        std::string fullName("toybrot.conf.d/");
        #if defined(TB_USE_STDFS) || defined (TB_USE_EXPFS)
            tbfs::create_directory(fullName);
        #endif
        bool fileIsClean = true;
        fullName.append(filename);
        std::fstream srcFile;
        srcFile.open(fullName, std::fstream::in);
        std::vector<std::string> lines;
        std::string ln;
        /**
         * A bunch of maps here used to reconstruct and manipulate the file
         *  - loadedValues has the values themselves, used to convert to numbers
         *  - valueDefaultStrings has the standard ones used to generate a new file
         *  - valueStringsFromFile has the actual lines from the file
         */
        std::unordered_map<std::string,std::string> loadedValues, valueDefaultStrings, valueStringsFromFile;
        /**
          * keywords are ignored for actual parsing. The split up keys are to make writing the file
          * more straightforward
          */
        std::unordered_set<std::string> keywords{"float", "double", "int", "uint", "uint32_t" , "int32_t" , "size_t", "unsigned", "short", "long", "="};
        std::vector<std::string> camKeys{ "cameraX", "cameraY", "cameraZ", "targetX", "targetY", "targetZ", "width"
                                         , "height", "near", "fovY"};

        std::vector<std::string> colKeys{ "hueFactor", "hueOffset", "valueFactor", "valueRange"
                                        , "valueClamp", "satValue", "bgRed", "bgGreen", "bgBlue", "bgAlpha"};

        std::vector<std::string> rayKeys{"maxRaySteps", "collisionMinDist"};

        std::vector<std::string> boxKeys{"fixedRadiusSq", "minRadiusSq", "foldingLimit", "boxScale", "boxIterations" };

        // spaces intentional to get everything to line up nice
        std::string uintTypeName{"uint32_t"};
        std::string intTypeName {"int32_t "};
        std::string floatTypename{std::string(typeid(T).name()).compare(0,1,"f") ? "double  ": "float   "};

        valueDefaultStrings.emplace("cameraX",          floatTypename + std::string(" cameraX          = ") + valString(cam->Pos().X())      );
        valueDefaultStrings.emplace("cameraY",          floatTypename + std::string(" cameraY          = ") + valString(cam->Pos().Y())      );
        valueDefaultStrings.emplace("cameraZ",          floatTypename + std::string(" cameraZ          = ") + valString(cam->Pos().Z())      );
        valueDefaultStrings.emplace("targetX",          floatTypename + std::string(" targetX          = ") + valString(cam->Target().X())   );
        valueDefaultStrings.emplace("targetY",          floatTypename + std::string(" targetY          = ") + valString(cam->Target().Y())   );
        valueDefaultStrings.emplace("targetZ",          floatTypename + std::string(" targetZ          = ") + valString(cam->Target().Z())   );
        valueDefaultStrings.emplace("width",            uintTypeName  + std::string(" width            = ") + valString(cam->ScreenWidth())  );
        valueDefaultStrings.emplace("height",           uintTypeName  + std::string(" height           = ") + valString(cam->ScreenHeight()) );
        valueDefaultStrings.emplace("near",             floatTypename + std::string(" near             = ") + valString(cam->Near())         );
        valueDefaultStrings.emplace("fovY",             floatTypename + std::string(" fovY             = ") + valString(cam->FovY())         );

        valueDefaultStrings.emplace("hueFactor",        floatTypename + std::string(" hueFactor        = ") + valString(params->HueFactor())        );
        valueDefaultStrings.emplace("hueOffset",        intTypeName   + std::string(" hueOffset        = ") + valString(params->HueOffset())        );
        valueDefaultStrings.emplace("valueFactor",      floatTypename + std::string(" valueFactor      = ") + valString(params->ValueFactor())      );
        valueDefaultStrings.emplace("valueRange",       floatTypename + std::string(" valueRange       = ") + valString(params->ValueRange())       );
        valueDefaultStrings.emplace("valueClamp",       floatTypename + std::string(" valueClamp       = ") + valString(params->ValueClamp())       );
        valueDefaultStrings.emplace("satValue",         floatTypename + std::string(" satValue         = ") + valString(params->SatValue())         );
        valueDefaultStrings.emplace("bgRed",            floatTypename + std::string(" bgRed            = ") + valString(params->BgRed())            );
        valueDefaultStrings.emplace("bgGreen",          floatTypename + std::string(" bgGreen          = ") + valString(params->BgGreen())          );
        valueDefaultStrings.emplace("bgBlue",           floatTypename + std::string(" bgBlue           = ") + valString(params->BgBlue())           );
        valueDefaultStrings.emplace("bgAlpha",          floatTypename + std::string(" bgAlpha          = ") + valString(params->BgAlpha())          );
        valueDefaultStrings.emplace("maxRaySteps",      uintTypeName  + std::string(" maxRaySteps      = ") + valString(params->MaxRaySteps())      );
        valueDefaultStrings.emplace("collisionMinDist", floatTypename + std::string(" collisionMinDist = ") + valString(params->CollisionMinDist()) );
        valueDefaultStrings.emplace("fixedRadiusSq",    floatTypename + std::string(" fixedRadiusSq    = ") + valString(params->FixedRadiusSq())    );
        valueDefaultStrings.emplace("minRadiusSq",      floatTypename + std::string(" minRadiusSq      = ") + valString(params->MinRadiusSq())      );
        valueDefaultStrings.emplace("foldingLimit",     floatTypename + std::string(" foldingLimit     = ") + valString(params->FoldingLimit())     );
        valueDefaultStrings.emplace("boxScale",         floatTypename + std::string(" boxScale         = ") + valString(params->BoxScale())         );
        valueDefaultStrings.emplace("boxIterations",    uintTypeName  + std::string(" boxIterations    = ") + valString(params->BoxIterations())    );



        Vec3<T> camPos;
        Vec3<T> camTarg;
        uint32_t camWidth;
        uint32_t camHeight;
        T camNear;
        T camFoV;

        T        hueFac;
        int32_t  hueOff;
        T        valFac;
        T        valRng;
        T        valCmp;
        T        satVal;
        T        bgR;
        T        bgG;
        T        bgB;
        T        bgA;
        uint32_t maxSteps;
        T        minDist;
        T        frSq;
        T        mrSq;
        T        fLim;
        T        bScale;
        uint32_t bIter;

        lines.push_back(ln);
        long double d;
        if(srcFile.is_open())
        {
            while(std::getline(srcFile,ln))
            {
                if(!ln.empty() && (ln.length() > 2) && ln.compare(0,2,"//") && ln.compare(0,1,"#") )
                {
                    lines.push_back(ln);
                }
            }
            srcFile.close();
            std::cout << "Reading config from " << fullName << std::endl;
            for(auto& l: lines)
            {
                std::stringstream ss(l);
                std::string token{};
                std::string key{};
                while(std::getline(ss, token, ' '))
                {
                    if(!token.empty() && (keywords.find(token) == keywords.end()) )
                    {
                        if(key.empty())
                        {
                            key = token;
                            if(valueStringsFromFile.find(key) != valueStringsFromFile.end())
                            {
                                std::cout << "Found duplicate key (?) " << key << std::endl;
                                std::cout << "valueStringsFromFile[\""<< key <<"\"] = " << valueStringsFromFile[key] << std::endl;
                                throw std::runtime_error("ERROR: Config file has duplicated entries!");
                            }
                            valueStringsFromFile.emplace(key, l);
                        }
                        else
                        {
                            loadedValues.emplace(key, token);
                        }
                    }
                }
            }
            for(const auto& k : valueDefaultStrings)
            {
                auto keyVal = loadedValues.find(k.first);
                if(keyVal != loadedValues.end())
                {
                    //If we've found the value, let's prepare to use it

                    // convert and store camera values
                    if(!keyVal->first.compare("cameraX"))
                    {
                        camPos.setX(static_cast<T>(std::stod(keyVal->second)));
                        continue;
                    }
                    if(!keyVal->first.compare("cameraY"))
                    {
                        camPos.setY(static_cast<T>(std::stod(keyVal->second)));
                        continue;
                    }
                    if(!keyVal->first.compare("cameraZ"))
                    {
                        camPos.setZ(static_cast<T>(std::stod(keyVal->second)));
                        continue;
                    }
                    if(!keyVal->first.compare("targetX"))
                    {
                        camTarg.setX(static_cast<T>(std::stod(keyVal->second)));
                        continue;
                    }
                    if(!keyVal->first.compare("targetY"))
                    {
                       camTarg.setY(static_cast<T>(std::stod(keyVal->second)));
                       continue;
                    }
                    if(!keyVal->first.compare("targetZ"))
                    {
                       camTarg.setZ(static_cast<T>(std::stod(keyVal->second)));
                       continue;
                    }
                    if(!keyVal->first.compare("width"))
                    {
                       camWidth = static_cast<uint32_t>(std::stoul(keyVal->second));
                       continue;
                    }
                    if(!keyVal->first.compare("height"))
                    {
                       camHeight = static_cast<uint32_t>(std::stoul(keyVal->second));
                       continue;
                    }
                    if(!keyVal->first.compare("near"))
                    {
                       camNear = static_cast<T>(std::stod(keyVal->second));
                       continue;
                    }
                    if(!keyVal->first.compare("fovY"))
                    {
                       camFoV = static_cast<T>(std::stod(keyVal->second));
                       continue;
                    }
                    //Store and convert parameter values
                    if(!keyVal->first.compare("hueFactor"))
                    {
                       hueFac = static_cast<T>(std::stod(keyVal->second));
                       continue;
                    }
                    if(!keyVal->first.compare("hueOffset"))
                    {
                       hueOff = static_cast<int32_t>(std::stoi(keyVal->second));
                       continue;
                    }
                    if(!keyVal->first.compare("valueFactor"))
                    {
                       valFac = static_cast<T>(std::stod(keyVal->second));
                       continue;
                    }
                    if(!keyVal->first.compare("valueRange"))
                    {
                       valRng = static_cast<T>(std::stod(keyVal->second));
                       continue;
                    }
                    if(!keyVal->first.compare("valueClamp"))
                    {
                       valCmp = static_cast<T>(std::stod(keyVal->second));
                       continue;
                    }
                    if(!keyVal->first.compare("satValue"))
                    {
                       satVal = static_cast<T>(std::stod(keyVal->second));
                       continue;
                    }
                    if(!keyVal->first.compare("bgRed"))
                    {
                       bgR = static_cast<T>(std::stod(keyVal->second));
                       continue;
                    }
                    if(!keyVal->first.compare("bgGreen"))
                    {
                       bgG = static_cast<T>(std::stod(keyVal->second));
                       continue;
                    }
                    if(!keyVal->first.compare("bgBlue"))
                    {
                       bgB = static_cast<T>(std::stod(keyVal->second));
                       continue;
                    }
                    if(!keyVal->first.compare("bgAlpha"))
                    {
                       bgA = static_cast<T>(std::stod(keyVal->second));
                       continue;
                    }
                    if(!keyVal->first.compare("maxRaySteps"))
                    {
                       maxSteps = static_cast<uint32_t>(std::stoul(keyVal->second));
                       continue;
                    }
                    if(!keyVal->first.compare("collisionMinDist"))
                    {
                       minDist = static_cast<T>(std::stod(keyVal->second));
                       continue;
                    }
                    if(!keyVal->first.compare("fixedRadiusSq"))
                    {
                       frSq = static_cast<T>(std::stod(keyVal->second));
                       continue;
                    }
                    if(!keyVal->first.compare("minRadiusSq"))
                    {
                       mrSq = static_cast<T>(std::stod(keyVal->second));
                       continue;
                    }
                    if(!keyVal->first.compare("foldingLimit"))
                    {
                       fLim = static_cast<T>(std::stod(keyVal->second));
                       continue;
                    }
                    if(!keyVal->first.compare("boxScale"))
                    {
                       bScale = static_cast<T>(std::stod(keyVal->second));
                       continue;
                    }
                    if(!keyVal->first.compare("boxIterations"))
                    {
                       bIter = static_cast<uint32_t>(std::stoul(keyVal->second));
                       continue;
                    }
                    std::cout << "Unexpected value from config file: " << keyVal->first << " = " << keyVal->second << std::endl;
                }
                else
                {
                    //If we haven't found the value, get the default from the provided structs                    
                    fileIsClean = false;
                    std::cout << "Missing value for: " << k.first << std::endl;
                    // convert and store camera values
                    if(!k.first.compare("cameraX"))
                    {
                        camPos.setX(cam->Pos().X());
                        continue;
                    }
                    if(!k.first.compare("cameraY"))
                    {
                        camPos.setY(cam->Pos().Y());
                        continue;
                    }
                    if(!k.first.compare("cameraZ"))
                    {
                        camPos.setZ(cam->Pos().Z());
                        continue;
                    }
                    if(!k.first.compare("targetX"))
                    {
                        camTarg.setX(cam->Target().X());
                        continue;
                    }
                    if(!k.first.compare("targetY"))
                    {
                        camTarg.setY(cam->Target().Y());
                        continue;
                    }
                    if(!k.first.compare("targetZ"))
                    {
                        camTarg.setZ(cam->Target().Z());
                        continue;
                    }
                    if(!k.first.compare("width"))
                    {
                        camWidth = cam->ScreenWidth();
                        continue;
                    }
                    if(!k.first.compare("height"))
                    {
                        camHeight = cam->ScreenHeight();
                        continue;
                    }
                    if(!k.first.compare("near"))
                    {
                        camNear = cam->Near();
                        continue;
                    }
                    if(!k.first.compare("fovY"))
                    {
                        camFoV = cam->FovY();
                        continue;
                    }
                    //Store and convert parameter values
                    if(!k.first.compare("hueFactor"))
                    {
                        hueFac = params->HueFactor();
                        continue;
                    }
                    if(!k.first.compare("hueOffset"))
                    {
                        hueOff = params->HueOffset();
                        continue;
                    }
                    if(!k.first.compare("valueFactor"))
                    {
                        valFac = params->ValueFactor();
                        continue;
                    }
                    if(!k.first.compare("valueRange"))
                    {
                        valRng = params->ValueRange();
                        continue;
                    }
                    if(!k.first.compare("valueClamp"))
                    {
                        valCmp = params->ValueClamp();
                        continue;
                    }
                    if(!k.first.compare("satValue"))
                    {
                        satVal = params->SatValue();
                        continue;
                    }
                    if(!k.first.compare("bgRed"))
                    {
                        bgR = params->BgRed();
                        continue;
                    }
                    if(!k.first.compare("bgGreen"))
                    {
                        bgG = params->BgGreen();
                        continue;
                    }
                    if(!k.first.compare("bgBlue"))
                    {
                        bgB = params->BgBlue();
                        continue;
                    }
                    if(!k.first.compare("bgAlpha"))
                    {
                        bgA = params->BgAlpha();
                        continue;
                    }
                    if(!k.first.compare("maxRaySteps"))
                    {
                        maxSteps = params->MaxRaySteps();
                        continue;
                    }
                    if(!k.first.compare("collisionMinDist"))
                    {
                        minDist = params->CollisionMinDist();
                        continue;
                    }
                    if(!k.first.compare("fixedRadiusSq"))
                    {
                        frSq = params->FixedRadiusSq();
                        continue;
                    }
                    if(!k.first.compare("minRadiusSq"))
                    {
                        mrSq = params->MinRadiusSq();
                        continue;
                    }
                    if(!k.first.compare("foldingLimit"))
                    {
                        fLim = params->FoldingLimit();
                        continue;
                    }
                    if(!k.first.compare("boxScale"))
                    {
                        bScale = params->BoxScale();
                        continue;
                    }
                    if(!k.first.compare("boxIterations"))
                    {
                        bIter = params->BoxIterations();
                        continue;
                    }

                    std::cout << "Unexpected value from config file: " << keyVal->first << " = " << keyVal->second << std::endl;
                }

            }

            if(!fileIsClean)
            {
                std::cout << "Config files missing some values. Inserting defaults" << std::endl;
            }
            /**
              * I use placement new to overwrite the old values for the camera and parameters.
              * Though the parameters are pretty straightforward and have setters I could call
              * the Camera, on the other hand, needs to recalculate the screen coordinates which
              * it would need to do after every setter, if it had them.
              * Placement new allows me to overwrite the data on location with new information and,
              * since I know both the Camera and the Parameters objects are essentally PODs, this is
              * safe, if intense
              */
            new (cam.get()) Camera<T>(camPos, camTarg, camWidth, camHeight, camNear, camFoV);
            new (params.get()) Parameters<T>( maxSteps, minDist, hueFac, hueOff, valFac, valRng, valCmp
                                            , satVal , bgR, bgG, bgB, bgA, frSq, mrSq, fLim, bScale, bIter);
        }
        else
        {
            std::cout << "Config file does not exist, writing defaults" << std::endl;
            fileIsClean = false;
        }

        if(!fileIsClean)
        {
            srcFile.open(fullName, std::fstream::out | std::fstream::trunc);
            //(re)write the file
            std::vector<std::string> lines
            { "// ToyBrot configuration file"
            , "// Type information is merely for user convenience"
            , "// File named .c for ease of highlighting in various editors"
            , "// comments start with either // or #"
            , "// Options available as CLI options have preference over values here"
            , ""
            , "// Camera information"
            };
            //this is slightly over the expected line count
            lines.reserve(60);
            for(auto& key : camKeys)
            {
                if(valueStringsFromFile.find(key) != valueStringsFromFile.end())
                {
                    lines.push_back(valueStringsFromFile[key]);
                }
                else
                {
                    lines.push_back(valueDefaultStrings[key]);
                }
            }
            lines.push_back("");
            lines.push_back("// Colouring parameters ");
            for(auto& key : colKeys)
            {
                if(valueStringsFromFile.find(key) != valueStringsFromFile.end())
                {
                    lines.push_back(valueStringsFromFile[key]);
                }
                else
                {
                    lines.push_back(valueDefaultStrings[key]);
                }
            }
            lines.push_back("");
            lines.push_back("// Raymarching parameters ");
            for(auto& key : rayKeys)
            {
                if(valueStringsFromFile.find(key) != valueStringsFromFile.end())
                {
                    lines.push_back(valueStringsFromFile[key]);
                }
                else
                {
                    lines.push_back(valueDefaultStrings[key]);
                }
            }
            lines.push_back("");
            lines.push_back("// Mandelbox parameters ");
            for(auto& key : boxKeys)
            {
                if(valueStringsFromFile.find(key) != valueStringsFromFile.end())
                {
                    lines.push_back(valueStringsFromFile[key]);
                }
                else
                {
                    lines.push_back(valueDefaultStrings[key]);
                }
            }
            lines.push_back("");

            for(auto& l : lines)
            {
                srcFile << l << std::endl;
            }
            srcFile.flush();
            srcFile.close();

        }

        return fileIsClean;
    }
}

#endif //_TOYBROT_DATA_TYPES_HXX_DEFINED_
