// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <vkt/config.h>
#include <vkt/ExecutionPolicy.hpp>

#include "Logging.hpp"
#include "Timer.hpp"

#define VKT_CUDA_SAFE_CALL__(X) X /* TODO */

#ifdef __CUDACC__
#define VKT_FUNC __host__ __device__
#else
#define VKT_FUNC
#endif
