#ifndef WD_VEC3_HPP_DEFINED
#define WD_VEC3_HPP_DEFINED

#include <cmath>
#include <cuda_runtime.h>

template < typename T >
class Vec3
{
public:

    __device__ __host__ explicit Vec3() : x(0), y(0), z(0) {}
    __device__ __host__ explicit Vec3(T xVal, T yVal, T zVal): x{xVal}, y{yVal}, z{zVal} {}
    __device__ __host__ Vec3 (const Vec3<T>& ref): x{ref.X()}, y{ref.Y()}, z{ref.Z()} {}
    __device__ __host__ Vec3& operator= (const Vec3<T>& ref) {this->x = ref.X(); this->y = ref.Y(); this->z = ref.Z(); return *this;}

    template < typename U >
    __device__ __host__ operator Vec3<U>() const noexcept
    {
        return Vec3<U>{static_cast<U>(x),static_cast<U>(y),static_cast<U>(z)};
    }

    __device__ __host__ T X() const { return x; }
    __device__ __host__ T Y() const { return y; }
    __device__ __host__ T Z() const { return z; }
    __device__ __host__ void setX(T newX) { x = newX; }
    __device__ __host__ void setY(T newY) { y = newY; }
    __device__ __host__ void setZ(T newZ) { z = newZ; }

    __device__ __host__ T R() const { return x; }
    __device__ __host__ T G() const { return y; }
    __device__ __host__ T B() const { return z; }
    __device__ __host__ void setR(T r) { x = r; }
    __device__ __host__ void setG(T g) { y = g; }
    __device__ __host__ void setB(T b) { z = b; }

    __device__ __host__ Vec3& operator+=(const Vec3& rhs)
    {
        this->x += rhs.x;
        this->y += rhs.y;
        this->z += rhs.z;
        return *this;
    }

    __device__ __host__ Vec3& operator*=(const T& f)
    {
        x *= f;
        y *= f;
        z *= f;
        return *this;
    }
    __device__ __host__ Vec3& operator/=(const T& f)
    {
        x /= f;
        y /= f;
        z /= f;
        return *this;
    }
    __device__ __host__ Vec3& operator*=(const Vec3<T>& rhs)
    {
        x *= rhs.X();
        y *= rhs.Y();
        z *= rhs.Z();
        return *this;
    }

   __device__ __host__ T sqMod() const
    {
        return x*x+y*y+z*z;
    }

    __device__ __host__ T mod()
    {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wfloat-conversion"
#pragma clang diagnostic ignored "-Wconversion"

        return sqrt(sqMod());

#pragma clang diagnostic pop
    }

    __device__ __host__ Vec3<T> clamp(T min, T max)
    {
        Vec3<T> ret{x,y,z};
        ret.x = x < min ? min : (x > max ? max: x);
        ret.y = y < min ? min : (y > max ? max: y);
        ret.z = z < min ? min : (z > max ? max: z);
//        ret.x = x < min ? - min + x : (x > max ? max - x: x);
//        ret.y = y < min ? - min + y : (y > max ? max - y: y);
//        ret.z = z < min ? - min + z : (z > max ? max - z: z);
        return ret;
    }

    __device__ __host__ Vec3<T>& normalise()
    {
        T m = this->mod();
        x /= m;
        y /= m;
        z /= m;

        return *this;
    }

    static constexpr const double flEpsilon = 0.00001;

    __device__ __host__ static bool flEquals(float f1, float f2);
    __device__ __host__ static bool dbEquals(double f1, double f2);

    __device__ __host__ static bool flZero(float f);
    __device__ __host__ static bool dbZero(double f);

protected:


    T x;
    T y;
    T z;

};

template <typename T>
__device__ __host__ bool Vec3<T>::flEquals(float f1, float f2)
{
    return( fabs(static_cast<double>(f1 - f2)) <= Vec3::flEpsilon);
}

template <typename T>
__device__ __host__ bool Vec3<T>::dbEquals(double f1, double f2)
{
    return( fabs(f1 - f2) <= Vec3::flEpsilon);
}

template <typename T>
__device__ __host__ bool Vec3<T>::flZero(float f)
{
    return( fabs(static_cast<double>(f)) <= Vec3::flEpsilon);
}

template <typename T>
__device__ __host__ bool Vec3<T>::dbZero(double f)
{
    return( fabs(f) <= Vec3::flEpsilon);
}

template <typename T>
__device__ __host__ Vec3<T> operator+(const Vec3<T>& a, const Vec3<T>& b)
{
    Vec3<T> res = a;
    res += b;
    return res;
}

template <typename T>
__device__ __host__ Vec3<T> operator*(const Vec3<T>&a, T f)
{
    Vec3<T> res = a;
    res *= f;
    return res;
}

template <typename T>
__device__ __host__ Vec3<T> operator/(const Vec3<T>&a, T f)
{
    Vec3<T> res = a;
    res /= f;
    return res;
}

template <typename T>
__device__ __host__ bool operator == (const Vec3<T>&a, const Vec3<T>& b)
{
    return ( Vec3<T>::flEquals(a.X(),b.X()) &&
             Vec3<T>::flEquals(a.Y(),b.Y()) &&
             Vec3<T>::flEquals(a.Z(),b.Z()) );
}

template <typename T>
__device__ __host__ bool operator != (const Vec3<T>&a, const Vec3<T>& b)
{
    return !(a==b);
}

template <typename T>
__device__ __host__ Vec3<T> operator- (const Vec3<T>& v)
{
    Vec3<T> ret( -(v.X()), -(v.Y()), -(v.Z()) );
    return ret;
}

template <typename T>
__device__ __host__ Vec3<T> operator- (const Vec3<T>& a, const Vec3<T>& b)
{
    return Vec3<T>(a.X() - b.X(), a.Y()-b.Y(), a.Z() - b.Z());
}

template <typename T>
__device__ __host__ static Vec3<T> crossProd (const Vec3<T>& a, const Vec3<T>& b)
{
    return Vec3<T>(a.Y()*b.Z() - a.Z()*b.Y(), a.Z()*b.X() - a.X()*b.Z(), a.X()*b.Y() - a.Y()*b.X());
}

template <typename T>
__device__ __host__ static T dotProd (const Vec3<T>& a, const Vec3<T>& b)
{
    return a.X()*b.X() + a.Y()*b.Y() + a.Z()*b.Z();
}

template <typename T>
__device__ __host__ static Vec3<T> triNormal(const Vec3<T>& a, const Vec3<T>& b, const Vec3<T>& c)
{
    //TODO: winding checks and tuning.
    //ALSO: I feel this is not the best place for this to be so it's kind of temporary for now
    Vec3<T> ab = b-a;
    Vec3<T> ac = c-a;

    Vec3<T> normal = crossProd(ab, ac);
    normal.normalise();
    return normal;
}


template <typename T>
__device__ __host__ static Vec3<T> triNormal(const Vec3<T>* v)
{
    return triNormal(v[0],v[1],v[2]);
}

using Vec3f = Vec3<float>;
using Vec3i = Vec3<int>;
using Vec3d = Vec3<double>;

template<typename T>
class Vec4
{
public:

    __device__ __host__ explicit Vec4() : x(0), y(0), z(0),w(0) {}
    __device__ __host__ explicit Vec4(T xVal, T yVal, T zVal, T wVal): x{xVal}, y{yVal}, z{zVal}, w{wVal} {}
    __device__ __host__ Vec4 (const Vec4<T>& ref): x{ref.X()}, y{ref.Y()}, z{ref.Z()}, w{ref.W()} {}
    __device__ __host__ Vec4 (const Vec3<T>& ref): x{ref.X()}, y{ref.Y()}, z{ref.Z()}, w{0} {}
    __device__ __host__ Vec4 (const Vec3<T>& ref, T wVal): x{ref.X()}, y{ref.Y()}, z{ref.Z()}, w{wVal} {}
    __device__ __host__ Vec4& operator= (const Vec4<T>& ref) {this->x = ref.X(); this->y = ref.Y(); this->z = ref.Z(); this->w = ref.W(); return *this;}

    template < typename U >
    __device__ __host__ operator Vec4<U>() const noexcept
    {
        return Vec4<U>{static_cast<U>(x),static_cast<U>(y),static_cast<U>(z),static_cast<U>(w)};
    }

    __device__ __host__ T X() const { return x; }
    __device__ __host__ T Y() const { return y; }
    __device__ __host__ T Z() const { return z; }
    __device__ __host__ T W() const { return w; }
    __device__ __host__ void setX(T newX) { x = newX; }
    __device__ __host__ void setY(T newY) { y = newY; }
    __device__ __host__ void setZ(T newZ) { z = newZ; }
    __device__ __host__ void setW(T newW) { w = newW; }


    __device__ __host__ T R() const { return x; }
    __device__ __host__ T G() const { return y; }
    __device__ __host__ T B() const { return z; }
    __device__ __host__ T A() const { return w; }
    __device__ __host__ void setR(T r) { x = r; }
    __device__ __host__ void setG(T g) { y = g; }
    __device__ __host__ void setB(T b) { z = b; }
    __device__ __host__ void setA(T a) { w = a; }

private:

    T x,y,z,w;

};


using Vec4f = Vec4<float>;
using Vec4i = Vec4<int>;
using Vec4d = Vec4<double>;

#endif //WD_VEC3_HPP_DEFINED
