# OptiX7-Spectral-Path-Tracer
An spectral path tracer using optiX7

We use https://github.com/ingowald/optix7course as our code base and implement our renderer(.cu files) and add more features

If cmake cannot find OptiX, modify common/gdt/cmake/FindOptiX.cmake and change
```cpp
set(OptiX_INSTALL_DIR "D:/Codes/Optix/Optix7.4")
``` 
to your optiX path.
