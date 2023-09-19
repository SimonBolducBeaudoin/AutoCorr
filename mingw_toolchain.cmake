# Used to crosscompile gnu->windows 

# Set the C and C++ compilers to the MinGW versions
set(CMAKE_C_COMPILER "x86_64-w64-mingw32-gcc" CACHE PATH "C compiler" FORCE)
set(CMAKE_CXX_COMPILER "x86_64-w64-mingw32-g++" CACHE PATH "C++ compiler" FORCE)

set(CMAKE_CXX_FLAGS "-DMS_WIN64 -D_hypot=hypot" CACHE STRING "")
set(CMAKE_C_FLAGS "-DMS_WIN64 -D_hypot=hypot" CACHE STRING "")
