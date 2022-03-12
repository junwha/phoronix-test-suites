#ifndef FRACGEN_HPP_DEFINED
#define FRACGEN_HPP_DEFINED

#include <vector>
#include <memory>

#include "defines.hpp"
#include "Vec.hxx"
#include "dataTypes.hxx"

using RGBA = Vec4<tbFPType>;
using colourVec = std::vector<RGBA>;
using FracPtr = std::shared_ptr< colourVec >;
using CameraPtr = std::shared_ptr< Camera<tbFPType> >;
using ParamPtr = std::shared_ptr< Parameters<tbFPType> >;

class FracGen
{
public:

    static std::string getFlavour() noexcept;
    FracGen(bool bench = false, CameraPtr c = nullptr, ParamPtr p = nullptr);
    ~FracGen();
    FracPtr getBuffer() noexcept { return outBuffer;}
    size_t outLength() const noexcept { return outBuffer->size();}
    size_t outSize() const noexcept { return sizeof(RGBA) * outBuffer->size();}

    bool Generate();


private:

    bool bench;
    CameraPtr cam;
    ParamPtr parameters;
    FracPtr outBuffer;
    size_t lastHeight;

};


#endif //FRACGEN_HPP_DEFINED
