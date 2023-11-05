#pragma once

#include <iostream>

const double MAXM = 100;
const double MINM = -100;

template <class T>
class square_matrix
{
private:
    int size;
    T* arr;
public:
    square_matrix(int _size = 1000) : size(_size)
    {
        if (size <= 0) throw std::exception("size should be positive");

        arr = new T[size * size]();
    }
    square_matrix(square_matrix& _m)
    {
        size = _m.size;
        arr = new T[size * size]();
        for (int i = 0; i < size * size; i++)
            arr[i] = _m[i];
    }
    ~square_matrix()
    {
        delete[] arr;
        arr = nullptr;
        size = 0;
    }
    T& operator[](int i)
    {
        return arr[i];
    }
    const T& operator[](int i) const
    {
        return arr[i];
    }
    void randomfill()noexcept
    {
        for (int i = 0; i < size * size; i++)
            arr[i] = static_cast<T>((static_cast<double>(rand()) / static_cast<double>(RAND_MAX)) * (MAXM - MINM) + MINM);
    }
    int getsize() const noexcept
    {
        return size;
    }
    T* getarr() const noexcept
    {
        return arr;
    }
    square_matrix& operator=(const square_matrix& m)
    {
        if (*this == m) return *this;

        if (size != m.size)
        {
            T* tmp = new T[m.size]();
            delete[] arr;
            arr = tmp;
            size = m.size;
        }

        for (int i = 0; i < size * size; i++)
            arr[i] = m[i];

        return *this;
    }
    square_matrix& operator=(square_matrix&& m) noexcept
    {
        delete[] arr;

        arr = nullptr;

        size = 0;

        std::swap(arr, m.arr);
        std::swap(size, m.size);

        return *this;
    }
    bool operator == (const square_matrix & m) const noexcept
    {
        if (size != m.size) return false;

        for (int i = 0; i < size * size; i++)
            if (arr[i] != m[i]) return false;

        return true;
    }
    bool operator!= (const square_matrix & m) const
    {
        return !(*this == m);
    }
    square_matrix operator*(const square_matrix& m) 
    {
        if (size != m.size) throw std::out_of_range("matrices sizes should be equal!");

        square_matrix res(size);
        for (int i = 0; i < size; i++)
                for (int j = 0; j < size; j++)
                    for (int k = 0; k < size; k++)
                        res[i * size + j] += arr[i * size + k] * m[k * size + j];

        return res;
    }
    square_matrix operator-(const square_matrix& m)
    {
        if (size != m.size) throw std::out_of_range("matrices sizes should be equal!");

        square_matrix res(size);
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                res[i * size + j] = arr[i * size + j] - m[i * size + j];

        return res;
    }
    T abs() {
        T res=0;
        for (int i = 0; i < size * size; i++)
            res += arr[i] * arr[i];
        return res;
    }
    inline
    friend void mult(square_matrix& F, square_matrix& S, square_matrix& RES, int size)
    {
        //if ((F.size != S.size) || (F.size != RES.size) || (S.size != F.size)) throw std::exception("matrices sizes should be equal!");
        for (int i = 0; i < size; i++)
            for (int k = 0; k < size; k++)
                for (int j = 0; j < size; j++)
                    RES[i * size + j] += F[i * size + k] * S[k * size + j];
    }
    inline
    friend void block_mult(square_matrix& F, square_matrix& S, square_matrix& RES, int size)
    {
        //if ((F.size != S.size) || (F.size != RES.size) || (S.size != F.size)) throw std::exception("matrices sizes should be equal!");

        //L1 cache - 640 KB, L2 cache - 4 MB, L3 cache - 16 MB
        //block_size <= sqrt ( L2 / (3 * sizeof(T)) )
        // sqrt( L1 / (3 * sizeof(double)) = 165
        // T = double, block_size <= 418;

        int block_size = 400; 

        if (size % block_size == 0) 
        {
            for (int i1 = 0; i1 < size; i1 += block_size)
                for (int j1 = 0; j1 < size; j1 += block_size)
                    for (int k1 = 0; k1 < size; k1 += block_size)
                        for (int i2 = i1; i2 < i1 + block_size; i2++)
                            for (int k2 = k1; k2 < k1 + block_size; k2++)
                                for (int j2 = j1; j2 < j1 + block_size; j2++)
                                    RES[i2 * size + j2] += F[i2 * size + k2] * S[k2 * size + j2];  
        }
        else 
        {
            for (int i1 = 0; i1 < size; i1 += block_size)
                for (int j1 = 0; j1 < size; j1 += block_size)
                    for (int k1 = 0; k1 < size; k1 += block_size)
                        for (int i2 = i1; (i2 < i1 + block_size) && (i2 < size); i2++)
                            for (int k2 = k1; (k2 < k1 + block_size) && (k2 < size); k2++)
                                for (int j2 = j1; (j2 < j1 + block_size) && (j2 < size); j2++)
                                    RES[i2 * size + j2] += F[i2 * size + k2] * S[k2 * size + j2];
        }
    }
};

template <class T>
inline
void block_mult_pointers(T* F, T* S, T* RES, int size)
{

    //L1 cache - 640 KB, L2 cache - 4 MB, L3 cache - 16 MB
    // T = double, block_size = 418;

    int block_size = 400; //block_size:= sqrt ( L2 / (3 * sizeof(T)) )

    if (size % block_size == 0)
    {
        for (int i1 = 0; i1 < size; i1 += block_size)
            for (int j1 = 0; j1 < size; j1 += block_size)
                for (int k1 = 0; k1 < size; k1 += block_size)
                    for (int i2 = i1; i2 < i1 + block_size; i2++)
                        for (int k2 = k1; k2 < k1 + block_size; k2++)
                            for (int j2 = j1; j2 < j1 + block_size; j2++)
                                RES[i2 * size + j2] += F[i2 * size + k2] * S[k2 * size + j2];
    }
    else
    {
        for (int i1 = 0; i1 < size; i1 += block_size)
            for (int j1 = 0; j1 < size; j1 += block_size)
                for (int k1 = 0; k1 < size; k1 += block_size)
                    for (int i2 = i1; (i2 < i1 + block_size) && (i2 < size); i2++)
                        for (int k2 = k1; (k2 < k1 + block_size) && (k2 < size); k2++)
                            for (int j2 = j1; (j2 < j1 + block_size) && (j2 < size); j2++)
                                RES[i2 * size + j2] += F[i2 * size + k2] * S[k2 * size + j2];
    }
}

template <class T>
std::istream& operator>> (std::istream& in, square_matrix<T>& m)
{
    for (int i = 0; i < m.getsize(); i++)
        for (int j = 0; j < m.getsize(); j++)
            in >> m[i * m.getsize() + j];
    return in;
}

template <class T>
std::ostream& operator<< (std::ostream& out, square_matrix<T>& m)
{
    for (int i = 0; i < m.getsize(); i++)
    {
        for (int j = 0; j < m.getsize(); j++)
            out << m[i * m.getsize() + j] << " ";

        out << "\n";
    }
    return out;
}
