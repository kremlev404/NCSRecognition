#pragma once

// API macros allows export some symbols from the library
// It has different value for microsoft and gnu compiler
// Other compilers aren't supported
#ifdef _MSC_VER                 // Microsoft compiler
#elif __GNUC__                  // GCC compiler
#define DEFAULT_VISIBILITY __attribute__ ((visibility ("default")))
#else
#error UNKNOWN COMPILER TYPE (ONLY MICROSOFT AND GCC SUPPORTED) // error
#endif