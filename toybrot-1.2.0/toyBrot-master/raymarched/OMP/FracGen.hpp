#ifndef FRACGEN_HPP_DEFINED
#define FRACGEN_HPP_DEFINED

#include <vector>
#include <memory>
#include <functional>

#include "defines.hpp"
#include "Vec.hxx"
#include "dataTypes.hxx"

using RGBA = Vec4<tbFPType>;
using colourVec = std::vector<RGBA>;
using FracPtr = std::shared_ptr< colourVec >;
using CameraPtr = std::shared_ptr< Camera<tbFPType> >;
using ParamPtr = std::shared_ptr< Parameters<tbFPType> >;

class FracGen;
using estimatorFunction = std::function<tbFPType(const FracGen*, Vec3<tbFPType>)>;

class FracGen
{
public:

    FracGen(bool bench = false, CameraPtr c = nullptr, ParamPtr p = nullptr);
    ~FracGen();
    FracPtr getBuffer() noexcept { return outBuffer;}
    size_t outLength() const noexcept { return outBuffer->size();}
    size_t outSize() const noexcept { return sizeof(RGBA) * outBuffer->size();}

    bool Generate();


private:

    tbFPType boxDist(const Vec3<tbFPType>& p) const;
    tbFPType bulbDist(const Vec3<tbFPType>& p) const;
    tbFPType sphereDist(Vec3<tbFPType> p) const;
    RGBA HSVtoRGB(int H, tbFPType S, tbFPType V) const;
    RGBA getColour(const RGBA& steps) const;
    Vec4<tbFPType> trace(const Camera<tbFPType>& cam, size_t x, size_t y) const;
    bool traceRegion(colourVec& data,
                     const Camera<tbFPType> &cam,
                     uint32_t h0, uint32_t heightStep) const;


    bool bench;
    CameraPtr cam;
    ParamPtr parameters;
    FracPtr outBuffer;
    size_t lastHeight;

};


#endif //FRACGEN_HPP_DEFINED
