#pragma once

#include <complex>

// #if defined(GPU_TARGET_HIP)
// #include <hip/hip_vector_types.h>
// #else
// #include <vector_types.h>
// #endif

namespace contract
{

  template <typename T> struct Complex {
    T x, y;
  };

  template <> struct alignas(16) Complex<double> {
    double x, y;

    Complex() = default;
    __host__ __device__ __forceinline__ Complex(double r, double i = 0.0) : x(r), y(i) { }
    __host__ __device__ __forceinline__ explicit Complex(const double2 &z) : x(z.x), y(z.y) { }
    __host__ __forceinline__ explicit Complex(const std::complex<double> &z) : x(z.real()), y(z.imag()) { }

    __host__ __device__ __forceinline__ operator double2() const { return make_double2(x, y); }
    __host__ __device__ __forceinline__ Complex &operator=(const double2 &z)
    {
      x = z.x;
      y = z.y;
      return *this;
    }
    __host__ __forceinline__ operator std::complex<double>() const { return std::complex<double>(x, y); }
    __host__ __forceinline__ Complex &operator=(const std::complex<double> &z)
    {
      x = z.real();
      y = z.imag();
      return *this;
    }

    __host__ __device__ __forceinline__ double real() const { return x; }
    // __host__ __device__ __forceinline__ double &real() { return x; }
    __host__ __device__ __forceinline__ double imag() const { return y; }
    // __host__ __device__ __forceinline__ double &imag() { return y; }

    __host__ __device__ __forceinline__ Complex operator+() const { return *this; }
    __host__ __device__ __forceinline__ Complex operator-() const { return Complex(-x, -y); }
    __host__ __device__ __forceinline__ Complex conj() const { return Complex(x, -y); }

    __host__ __device__ __forceinline__ Complex operator+(const Complex &b) const { return Complex(x + b.x, y + b.y); }
    __host__ __device__ __forceinline__ Complex operator-(const Complex &b) const { return Complex(x - b.x, y - b.y); }
    __host__ __device__ __forceinline__ Complex operator*(const Complex &b) const
    {
      return Complex(x * b.x - y * b.y, x * b.y + y * b.x);
    }
    __host__ __device__ __forceinline__ Complex operator/(const Complex &b) const
    {
      const double denom = b.x * b.x + b.y * b.y;
      return Complex((x * b.x + y * b.y) / denom, (y * b.x - x * b.y) / denom);
    }

    __host__ __device__ __forceinline__ Complex operator+(double b) const { return Complex(x + b, y); }
    __host__ __device__ __forceinline__ Complex operator-(double b) const { return Complex(x - b, y); }
    __host__ __device__ __forceinline__ Complex operator*(double b) const { return Complex(x * b, y * b); }
    __host__ __device__ __forceinline__ Complex operator/(double b) const { return Complex(x / b, y / b); }

    __host__ __device__ __forceinline__ Complex &operator+=(const Complex &b)
    {
      x += b.x;
      y += b.y;
      return *this;
    }
    __host__ __device__ __forceinline__ Complex &operator-=(const Complex &b)
    {
      x -= b.x;
      y -= b.y;
      return *this;
    }
    __host__ __device__ __forceinline__ Complex &operator*=(const Complex &b)
    {
      const double rx = x * b.x - y * b.y;
      const double ry = x * b.y + y * b.x;
      x = rx;
      y = ry;
      return *this;
    }
    __host__ __device__ __forceinline__ Complex &operator/=(const Complex &b)
    {
      const double denom = b.x * b.x + b.y * b.y;
      const double rx = (x * b.x + y * b.y) / denom;
      const double ry = (y * b.x - x * b.y) / denom;
      x = rx;
      y = ry;
      return *this;
    }
    __host__ __device__ __forceinline__ Complex &operator+=(double b)
    {
      x += b;
      return *this;
    }
    __host__ __device__ __forceinline__ Complex &operator-=(double b)
    {
      x -= b;
      return *this;
    }
    __host__ __device__ __forceinline__ Complex &operator*=(double b)
    {
      x *= b;
      y *= b;
      return *this;
    }
    __host__ __device__ __forceinline__ Complex &operator/=(double b)
    {
      x /= b;
      y /= b;
      return *this;
    }

    __host__ __device__ __forceinline__ friend Complex operator+(double a, const Complex &b)
    {
      return Complex(a + b.x, b.y);
    }
    __host__ __device__ __forceinline__ friend Complex operator-(double a, const Complex &b)
    {
      return Complex(a - b.x, -b.y);
    }
    __host__ __device__ __forceinline__ friend Complex operator*(double a, const Complex &b)
    {
      return Complex(a * b.x, a * b.y);
    }
    __host__ __device__ __forceinline__ friend Complex operator/(double a, const Complex &b)
    {
      const double denom = b.x * b.x + b.y * b.y;
      return Complex((a * b.x) / denom, (-a * b.y) / denom);
    }
  };

  template <> struct alignas(8) Complex<float> {
    float x, y;

    Complex() = default;
    __host__ __device__ __forceinline__ Complex(float r, float i = 0.0f) : x(r), y(i) { }
    __host__ __device__ __forceinline__ explicit Complex(const float2 &z) : x(z.x), y(z.y) { }
    __host__ __forceinline__ explicit Complex(const std::complex<float> &z) : x(z.real()), y(z.imag()) { }

    __host__ __device__ __forceinline__ operator float2() const { return make_float2(x, y); }
    __host__ __device__ __forceinline__ Complex &operator=(const float2 &z)
    {
      x = z.x;
      y = z.y;
      return *this;
    }
    __host__ __forceinline__ operator std::complex<float>() const { return std::complex<float>(x, y); }
    __host__ __forceinline__ Complex &operator=(const std::complex<float> &z)
    {
      x = z.real();
      y = z.imag();
      return *this;
    }

    __host__ __device__ __forceinline__ float real() const { return x; }
    // __host__ __device__ __forceinline__ float &real() { return x; }
    __host__ __device__ __forceinline__ float imag() const { return y; }
    // __host__ __device__ __forceinline__ float &imag() { return y; }

    __host__ __device__ __forceinline__ Complex operator+() const { return *this; }
    __host__ __device__ __forceinline__ Complex operator-() const { return Complex(-x, -y); }
    __host__ __device__ __forceinline__ Complex conj() const { return Complex(x, -y); }

    __host__ __device__ __forceinline__ Complex operator+(const Complex &b) const { return Complex(x + b.x, y + b.y); }
    __host__ __device__ __forceinline__ Complex operator-(const Complex &b) const { return Complex(x - b.x, y - b.y); }
    __host__ __device__ __forceinline__ Complex operator*(const Complex &b) const
    {
      return Complex(x * b.x - y * b.y, x * b.y + y * b.x);
    }
    __host__ __device__ __forceinline__ Complex operator/(const Complex &b) const
    {
      const float denom = b.x * b.x + b.y * b.y;
      return Complex((x * b.x + y * b.y) / denom, (y * b.x - x * b.y) / denom);
    }

    __host__ __device__ __forceinline__ Complex operator+(float b) const { return Complex(x + b, y); }
    __host__ __device__ __forceinline__ Complex operator-(float b) const { return Complex(x - b, y); }
    __host__ __device__ __forceinline__ Complex operator*(float b) const { return Complex(x * b, y * b); }
    __host__ __device__ __forceinline__ Complex operator/(float b) const { return Complex(x / b, y / b); }

    __host__ __device__ __forceinline__ Complex &operator+=(const Complex &b)
    {
      x += b.x;
      y += b.y;
      return *this;
    }
    __host__ __device__ __forceinline__ Complex &operator-=(const Complex &b)
    {
      x -= b.x;
      y -= b.y;
      return *this;
    }
    __host__ __device__ __forceinline__ Complex &operator*=(const Complex &b)
    {
      const float rx = x * b.x - y * b.y;
      const float ry = x * b.y + y * b.x;
      x = rx;
      y = ry;
      return *this;
    }
    __host__ __device__ __forceinline__ Complex &operator/=(const Complex &b)
    {
      const float denom = b.x * b.x + b.y * b.y;
      const float rx = (x * b.x + y * b.y) / denom;
      const float ry = (y * b.x - x * b.y) / denom;
      x = rx;
      y = ry;
      return *this;
    }
    __host__ __device__ __forceinline__ Complex &operator+=(float b)
    {
      x += b;
      return *this;
    }
    __host__ __device__ __forceinline__ Complex &operator-=(float b)
    {
      x -= b;
      return *this;
    }
    __host__ __device__ __forceinline__ Complex &operator*=(float b)
    {
      x *= b;
      y *= b;
      return *this;
    }
    __host__ __device__ __forceinline__ Complex &operator/=(float b)
    {
      x /= b;
      y /= b;
      return *this;
    }

    __host__ __device__ __forceinline__ friend Complex operator+(float a, const Complex &b)
    {
      return Complex(a + b.x, b.y);
    }
    __host__ __device__ __forceinline__ friend Complex operator-(float a, const Complex &b)
    {
      return Complex(a - b.x, -b.y);
    }
    __host__ __device__ __forceinline__ friend Complex operator*(float a, const Complex &b)
    {
      return Complex(a * b.x, a * b.y);
    }
    __host__ __device__ __forceinline__ friend Complex operator/(float a, const Complex &b)
    {
      const float denom = b.x * b.x + b.y * b.y;
      return Complex((a * b.x) / denom, (-a * b.y) / denom);
    }
  };

  typedef Complex<double> Complex128;
  typedef Complex<float> Complex64;

  template <typename T> __host__ __device__ __forceinline__ Complex<T> conj(const Complex<T> &z) { return z.conj(); }

} // namespace contract
