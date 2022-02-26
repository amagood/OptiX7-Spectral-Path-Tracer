//
// Created by amagood on 2022/2/26.
//

#ifndef OPTIX7COURSE_MYSTRING_CUH
#define OPTIX7COURSE_MYSTRING_CUH


__device__ char * my_strcpy(char *dest, const char *src){
    int i = 0;
    do {
        dest[i] = src[i];}
    while (src[i++] != 0);
    return dest;
}

__device__ char * my_strcat(char *dest, const char *src){
    int i = 0;
    while (dest[i] != 0) i++;
    my_strcpy(dest+i, src);
    return dest;
}

#endif //OPTIX7COURSE_MYSTRING_CUH
