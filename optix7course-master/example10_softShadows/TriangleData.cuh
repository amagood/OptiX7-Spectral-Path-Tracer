//
// Created by amagood on 2022/4/2.
//

#ifndef OPTIX7COURSE_TRIANGLEDATA_CUH
#define OPTIX7COURSE_TRIANGLEDATA_CUH

#include <optix_device.h>
#include <cuda_runtime.h>
#include "gdt/math/vec.h"
#include "LaunchParams.h"

struct TriangleData
{
    int primID;
    gdt::vec3i index;
    float u;
    float v;
    gdt::vec3f rayDir;
    gdt::vec3f A;
    gdt::vec3f B;
    gdt::vec3f C;
    gdt::vec3f Ng;
    gdt::vec3f Ns;
    gdt::vec3f rawNormal;
    gdt::vec3f diffuseColor;
    gdt::vec3f surfPos;

    __device__ TriangleData(const osc::TriangleMeshSBTData &sbtData)
    {
        using namespace gdt;

        // ------------------------------------------------------------------
        // gather some basic hit information
        // ------------------------------------------------------------------
        primID = optixGetPrimitiveIndex();
        index = sbtData.index[primID];
        u = optixGetTriangleBarycentrics().x;
        v = optixGetTriangleBarycentrics().y;

        // ------------------------------------------------------------------
        // compute normal, using either shading normal (if avail), or
        // geometry normal (fallback)
        // ------------------------------------------------------------------
        A = sbtData.vertex[index.x];
        B = sbtData.vertex[index.y];
        C = sbtData.vertex[index.z];

        Ng = normalize(-cross(B - A, C - A));
        Ns = (sbtData.normal)
                  ? ((1.f - u - v) * sbtData.normal[index.x]
                     + u * sbtData.normal[index.y]
                     + v * sbtData.normal[index.z])
                  : Ng;

        // ------------------------------------------------------------------
        // face-forward and normalize normals
        // ------------------------------------------------------------------
        rayDir = optixGetWorldRayDirection();
        rayDir = normalize(rayDir);

        rawNormal = Ns;

        if (dot(rayDir, Ng) > 0.f) Ng = -Ng;
        Ng = normalize(Ng);

        if (dot(Ng, Ns) < 0.f)
            Ns -= 2.f * dot(Ng, Ns) * Ng;

        rawNormal = normalize(rawNormal);
        Ns = normalize(Ns);

        // ------------------------------------------------------------------
        // compute diffuse material color, including diffuse texture, if
        // available
        // ------------------------------------------------------------------
        diffuseColor = sbtData.color;
        if (sbtData.hasTexture && sbtData.texcoord)
        {
            const vec2f tc
                    = (1.f - u - v) * sbtData.texcoord[index.x]
                      + u * sbtData.texcoord[index.y]
                      + v * sbtData.texcoord[index.z];

            vec4f fromTexture = tex2D<float4>(sbtData.texture, tc.x, tc.y);
            diffuseColor *= (vec3f) fromTexture;
        }

        surfPos = (1.f - u - v) * sbtData.vertex[index.x]
                  + u * sbtData.vertex[index.y]
                  + v * sbtData.vertex[index.z];

    }
};
/*
__device__ TriangleData getTriangleData(const osc::TriangleMeshSBTData &sbtData)
{
    using namespace gdt;
    TriangleData data;

    // ------------------------------------------------------------------
    // gather some basic hit information
    // ------------------------------------------------------------------
    primID = optixGetPrimitiveIndex();
    index = sbtData.index[primID];
    u = optixGetTriangleBarycentrics().x;
    v = optixGetTriangleBarycentrics().y;

    // ------------------------------------------------------------------
    // compute normal, using either shading normal (if avail), or
    // geometry normal (fallback)
    // ------------------------------------------------------------------
    A = sbtData.vertex[index.x];
    B = sbtData.vertex[index.y];
    C = sbtData.vertex[index.z];

    Ng = cross(B - A, C - A);
    Ns = (sbtData.normal)
         ? ((1.f - u - v) * sbtData.normal[index.x]
            + u * sbtData.normal[index.y]
            + v * sbtData.normal[index.z])
         : Ng;

    // ------------------------------------------------------------------
    // face-forward and normalize normals
    // ------------------------------------------------------------------
    rayDir = optixGetWorldRayDirection();

    if (dot(rayDir, Ng) > 0.f) Ng = -Ng;
    Ng = normalize(Ng);

    if (dot(Ng, Ns) < 0.f)
        Ns -= 2.f * dot(Ng, Ns) * Ng;
    Ns = normalize(Ns);

    // ------------------------------------------------------------------
    // compute diffuse material color, including diffuse texture, if
    // available
    // ------------------------------------------------------------------
    diffuseColor = sbtData.color;
    if (sbtData.hasTexture && sbtData.texcoord)
    {
        const vec2f tc
                = (1.f - u - v) * sbtData.texcoord[index.x]
                  + u * sbtData.texcoord[index.y]
                  + v * sbtData.texcoord[index.z];

        vec4f fromTexture = tex2D<float4>(sbtData.texture, tc.x, tc.y);
        diffuseColor *= (vec3f) fromTexture;
    }
}
*/
#endif //OPTIX7COURSE_TRIANGLEDATA_CUH
