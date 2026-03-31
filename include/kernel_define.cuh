#pragma once

#if defined(GPU_TARGET_SYCL)
#ifndef __global__
#define __global__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __forceinline__
#define __forceinline__ inline __attribute__((always_inline))
#endif
#endif