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

// returns true if `X` and `Y` are the same
__device__ int my_StringCompare(const char *X, const char *Y)
{
    while (*X && *Y)
    {
        if (*X != *Y) {
            return 0;
        }

        X++;
        Y++;
    }

    return (*Y == '\0');
}

// Function to implement `strstr()` function
__device__ const char* my_strstr(const char* X, const char* Y)
{
    while (*X != '\0')
    {
        if ((*X == *Y) && my_StringCompare(X, Y)) {
            return X;
        }
        X++;
    }

    return NULL;
}



#endif //OPTIX7COURSE_MYSTRING_CUH
