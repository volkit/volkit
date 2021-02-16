// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

// Include this STL header so we can evaluate version macros later on
#include <cstddef>

//-------------------------------------------------------------------------------------------------
// Determine compiler
//

#if defined(__clang__)
#define VKT_CXX_CLANG    (100 * __clang_major__ + __clang_minor__)
#elif defined(__INTEL_COMPILER) && defined(__GNUC__)
#define VKT_CXX_INTEL    (100 * __GNUC__ + __GNUC_MINOR__)
#elif defined(__GNUC__)
#define VKT_CXX_GCC      (100 * __GNUC__ + __GNUC_MINOR__)
#elif defined(_MSC_VER)
#define VKT_CXX_MSVC     (_MSC_VER)
#endif

#ifndef VKT_CXX_CLANG
#define VKT_CXX_CLANG 0
#endif
#ifndef VKT_CXX_INTEL
#define VKT_CXX_INTEL 0
#endif
#ifndef VKT_CXX_GCC
#define VKT_CXX_GCC 0
#endif
#ifndef VKT_CXX_MSVC
#define VKT_CXX_MSVC 0
#endif


//-------------------------------------------------------------------------------------------------
// Visibility
//

#if VKT_CXX_MSVC
#   define VKT_DLL_EXPORT __declspec(dllexport)
#   define VKT_DLL_IMPORT __declspec(dllimport)
#elif VKT_CXX_CLANG || VKT_CXX_GCC
#   define VKT_DLL_EXPORT __attribute__((visibility("default")))
#   define VKT_DLL_IMPORT __attribute__((visibility("default")))
#else
#   define VKT_DLL_EXPORT
#   define VKT_DLL_IMPORT
#endif

//-------------------------------------------------------------------------------------------------
// DLL export macro VKTAPI
//

#ifndef VKT_STATIC
#   ifdef volkit_EXPORTS
#       define VKTAPI VKT_DLL_EXPORT
#   else
#       define VKTAPI VKT_DLL_IMPORT
#   endif
#else
#   define VKTAPI
#endif
