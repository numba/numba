#ifndef DLLEXPORT

#ifdef _WIN32
    #define DLLEXPORT __declspec( dllexport )
#else
    #define DLLEXPORT
#endif

#endif
