#ifndef FRACGEN_HPP_DEFINED
#define FRACGEN_HPP_DEFINED

#include <vector>
#include <cstdint>
#include <memory>

#include "Vec.hxx"

struct Camera{ Vec3f pos, up, target; float AR, near, fovY;};
struct Screen{ Vec3f topLeft; int width, height; float pixelWidth, pixelHeight;};
using colourType = float;
using RGBA = Vec4<colourType>;
using colourVec = std::vector<RGBA>;
using FracPtr = std::shared_ptr< colourVec >;
namespace cl
{
    namespace sycl
    {
        class queue;
    }
}

class FracGen
{
public:

    FracGen(bool bench = false, size_t width = 1820, size_t height = 980);
    ~FracGen();
    FracPtr getBuffer() noexcept { return outBuffer;}
    size_t outLength() const noexcept { return outBuffer->size();}
    size_t outSize() const noexcept { return sizeof(RGBA) * outBuffer->size();}

    void Generate(int width, int height);


private:

    bool bench;
    std::shared_ptr<Camera> cam;
    std::unique_ptr<cl::sycl::queue> q;
    FracPtr outBuffer;
};

#endif //FRACGEN_HPP_DEFINED
