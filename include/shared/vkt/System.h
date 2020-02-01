// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <sys/param.h>

//-------------------------------------------------------------------------------------------------
// Endianness
//

#if defined(__BYTE_ORDER)
# if (__BYTE_ORDER == __ORDER_LITTLE_ENDIAN__)
#  define VKT_LITTLE_ENDIAN
# elif (__BYTE_ORDER == __LITTLE_ENDIAN)
#  define VKT_BIG_ENDIAN
# endif
#elif defined(__BYTE_ORDER__)
# if (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#  define VKT_LITTLE_ENDIAN
# elif (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
#  define VKT_BIG_ENDIAN
# endif
#endif
