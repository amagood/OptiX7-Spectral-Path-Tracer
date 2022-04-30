//
// Created by amagood on 2022/4/9.
//
#include <stdio.h>

#ifndef OPTIX7COURSE_MYVECTOR_CUH
#define OPTIX7COURSE_MYVECTOR_CUH
/*
    void push(int data): This function takes one element and inserts it at the last. Amortized time complexity is O(1).
    void push(int data, int index): It inserts data at the specified index. Time complexity is O(1).
    int get(int index): It is used to get the element at the specified index. Time complexity is O(1).
    void pop(): It deletes the last element. Time complexity is O(1).
    int size(): It returns the size of the vector i.e, the number of elements in the vector. Time complexity is O(1).
    int getcapacity(): It returns the capacity of the vector. Time complexity is O(1).
    void print(): It is used to print array elements. Time complexity is O(N), where N is the size of the vector.
 */

template <typename T> class vectorClass
{

    // arr is the integer pointer
    // which stores the address of our vector
    T* arr;

    // capacity is the total storage
    // capacity of the vector
    int capacity;

    // current is the number of elements
    // currently present in the vector
    int current;

public:
    // Default constructor to initialise
    // an initial capacity of 1 element and
    // allocating storage using dynamic allocation
    __device__ vectorClass()
    {
        arr = new T[16];
        capacity = 16;
        current = 0;
    }

    __device__ vectorClass(int size)
    {
        arr = new T[size];
        capacity = size;
        current = 0;
    }

    __device__ ~vectorClass()
    {
        delete[] arr;
        capacity = 1;
        current = 0;
    }

    // Function to add an element at the last
    __device__ void push(T data)
    {

        //printf("cur = %d, cap = %d\n", current, capacity);
        // if the number of elements is equal to the
        // capacity, that means we don't have space to
        // accommodate more elements. We need to double the
        // capacity
        if (current == capacity) {
            printf("OSO\n");
            T* temp = new T[2 * capacity];

            // copying old array elements to new array
            for (int i = 0; i < capacity; i++) {
                temp[i] = arr[i];
            }

            // deleting previous array
            delete[] arr;
            capacity *= 2;
            arr = temp;
        }

        // Inserting data
        arr[current] = data;
        current++;
    }

    // function to add element at any index
    __device__ void push(T data, int index)
    {

        // if index is equal to capacity then this
        // function is same as push defined above
        if (index == capacity)
            push(data);
        else
            arr[index] = data;
    }

    // function to extract element at any index
    __device__ T get(int index)
    {

        // if index is within the range
        if (index < current)
            return arr[index];
    }

    // function to delete last element
    __device__ void pop() { current--; }

    // function to get size of the vector
    __device__ int size() { return current; }

    // function to get capacity of the vector
    __device__ int getcapacity() { return capacity; }

    __device__ void erase()
    {
        current = 0;
    }
};

template<typename T>
class LocalVector
{
private:
    T* m_begin;
    T* m_end;

    size_t capacity;
    size_t length;
    __device__ void expand() {
        capacity *= 2;
        size_t tempLength = (m_end - m_begin);
        T* tempBegin = new T[capacity];

        memcpy(tempBegin, m_begin, tempLength * sizeof(T));
        delete[] m_begin;
        m_begin = tempBegin;
        m_end = m_begin + tempLength;
        length = static_cast<size_t>(m_end - m_begin);
    }
public:
    __device__  explicit LocalVector() : length(0), capacity(16) {
        m_begin = new T[capacity];
        m_end = m_begin;
    }
    __device__ T& operator[] (unsigned int index) {
        return *(m_begin + index);//*(begin+index)
    }
    __device__ T* begin() {
        return m_begin;
    }
    __device__ T* end() {
        return m_end;
    }
    __device__ ~LocalVector()
    {
        delete[] m_begin;
        m_begin = nullptr;
    }

    __device__ void push(T t) {

        if ((m_end - m_begin) >= capacity) {
            expand();
        }

        new (m_end) T(t);
        m_end++;
        length++;
    }
    __device__ T pop() {
        T endElement = (*m_end);
        delete m_end;
        m_end--;
        return endElement;
    }

    __device__ size_t getSize() {
        return length;
    }

    __device__ void erase()
    {
        m_end = m_begin;
        length = 0;
    }
};

#endif //OPTIX7COURSE_MYVECTOR_CUH
