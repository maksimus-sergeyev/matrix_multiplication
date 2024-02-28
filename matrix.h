#pragma once
#include <iostream>

const double MAXM = 1000;
const double MINM = -1000;

// T - Fundamental type
template <typename T>
class matrix
{
private:
    int row;
    int col;
    T* data;
public:
    matrix(int _row = 1000, int _col = 1000) : row(_row), col(_col)
    {
        if ((row <= 0) || (col <= 0)) throw std::invalid_argument("size should be positive!");

        data = new T[row * col]();
    }
    matrix(const matrix& m) : row(m.row), col(m.col)
    {
        data = new T[row * col];

        std::memcpy(data, m.data, row * col * sizeof(T));
    }
    matrix(matrix&& m) noexcept: row(m.row), col(m.col), data(m.data)
    {
        m.row = m.col = 0;
        m.data = nullptr;
    }
    ~matrix()
    {
        delete[] data;
        data = nullptr;
        row = col = 0;
    }
    T& operator[](int i)
    {
        return data[i];
    }
    const T& operator[](int i) const
    {
        return data[i];
    }
    void randomfill() noexcept
    {
        for (int i = 0; i < row * col; i++)
            data[i] = static_cast<T>((static_cast<double>(rand()) / static_cast<double>(RAND_MAX)) * (MAXM - MINM) + MINM);
    }
    int getrow() const noexcept
    {
        return row;
    }
    int getcol() const noexcept
    {
        return col;
    }
    matrix& operator=(const matrix& m)
    {
        if (this == &m) return *this;

        if (row * col != m.row * m.col)
        {
            T* tmp = new T[m.row * m.col];
            delete[] data;
            data = tmp;
        }

        row = m.row;
        col = m.col;

        std::memcpy(data, m.data, row * col * sizeof(T) );

        return *this;
    }
    matrix& operator=(matrix&& m) noexcept
    {
        if (this == &m) return *this;

        std::swap(data, m.data);

        std::swap(row, m.row);

        std::swap(col, m.col);

        return *this;
    }
    bool operator == (const matrix& m) const noexcept
    {
        if ((row != m.row) || (col != m.col)) return false;

        for (int i = 0; i < row * col; i++)
            if (data[i] != m[i]) return false;

        return true;
    }
    bool operator!= (const matrix& m) const noexcept
    {
        return !(*this == m);
    }

    matrix& operator+=(const matrix& m)
    {
        if ((row != m.row) || (col != m.col)) throw std::invalid_argument("matrices sizes should match!");

#pragma omp parallel for
        for (int i = 0; i < row; i++)
#pragma omp simd
            for (int j = 0; j < col; j++)
                data[i * col + j] += m[i * m.col + j];

        return *this;
    }
    matrix operator*(const matrix& m) //standard function. do not edit! it used for correctness check!!
    {
        if (col != m.row) throw std::invalid_argument("matrices sizes should match!");

        matrix res(row, m.col);
        for (int i = 0; i < row; i++)
            for (int j = 0; j < m.col; j++)
                for (int k = 0; k < col; k++)
                    res[i * res.col + j] += data[i * col + k] * m[k * m.col + j];

        return res;
    }
    matrix operator-(const matrix& m)
    {
        if ((col != m.col) || (row != m.row)) throw std::invalid_argument("matrices sizes should match!");

        matrix res(*this);
#pragma omp parallel for
        for (int i = 0; i < row; i++)
#pragma omp simd
            for (int j = 0; j < col; j++)
                res[i * col + j] -= m[i * m.col + j];

        return res;
    }
    matrix operator+(const matrix& m)
    {
        if ((col != m.col) || (row != m.row)) throw std::invalid_argument("matrices sizes should match!");

        matrix res(*this);
#pragma omp parallel for
        for (int i = 0; i < row; i++)
#pragma omp simd
            for (int j = 0; j < col; j++)
                res[i * col + j] += m[i * m.col + j];

        return res;
    }
    T norm() const noexcept
    {
        T res = static_cast<T>(0);

        int sz = std::thread::hardware_concurrency();

        T tmp[sz];

        std::memset(tmp, 0, sz * sizeof(T));

#pragma omp parallel for
        for (int i = 0; i < row * col; i++)
            tmp[omp_get_thread_num()] += data[i] * data[i];

        for (int i = 0; i < sz; i++)
            res += tmp[i];

        return res;
    }
    inline
        friend void mult(matrix& F, matrix& S, matrix& RES)
    {
        if ((F.col != S.row) || (F.row != RES.row) || (S.col != RES.col)) throw std::invalid_argument("matrices sizes should match!");
        if ((&F == &RES) || (&S == &RES)) throw std::invalid_argument("RES cannot be used as argument F or S");

        std::memset(RES.data, 0, RES.row * RES.col * sizeof(T));

        for (int i = 0; i < F.row; i++)
            for (int k = 0; k < F.col; k++)
#pragma omp simd
                for (int j = 0; j < S.col; j++)
                    RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
    }
    inline
        friend void parallel_mult(matrix& F, matrix& S, matrix& RES)
    {
        if ((F.col != S.row) || (F.row != RES.row) || (S.col != RES.col)) throw std::invalid_argument("matrices sizes should match!");
        if ((&F == &RES) || (&S == &RES)) throw std::invalid_argument("RES cannot be used as argument F or S");

        //may return 0 when not able to detect
        const auto processor_count = std::thread::hardware_concurrency();// / 2;

        std::memset(RES.data, 0, RES.row * RES.col * sizeof(T));

#pragma omp parallel for num_threads(processor_count)
        for (int i = 0; i < F.row; i++)
            for (int k = 0; k < F.col; k++)
#pragma omp simd
                for (int j = 0; j < S.col; j++)
                    RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
    }
    inline
        friend void block_mult(matrix& F, matrix& S, matrix& RES)
    {
        if ((F.col != S.row) || (F.row != RES.row) || (S.col != RES.col)) throw std::invalid_argument("matrices sizes should match!");
        if ((&F == &RES) || (&S == &RES)) throw std::invalid_argument("RES cannot be used as argument F or S");
        
        const int block_size_row = 64, block_size_col = 64;

        std::memset(RES.data, 0, RES.row * RES.col * sizeof(T));

        int t = F.row - (F.row % block_size_row);// i
        int l = S.col - (S.col % block_size_row);// j
        int s = F.col - (F.col % block_size_col);// k

        
        for (int i1 = 0; i1 < t; i1 += block_size_row)
            for (int k1 = 0; k1 < s; k1 += block_size_col)
                for (int j1 = 0; j1 < l; j1 += block_size_row)
                    for (int i2 = i1; i2 < i1 + block_size_row; i2++)
                        for (int k2 = k1; k2 < k1 + block_size_col; k2++)
#pragma omp simd
                            for (int j2 = j1; j2 < j1 + block_size_row; j2++)
                                RES[i2 * RES.col + j2] += F[i2 * F.col + k2] * S[k2 * S.col + j2];

        if (S.col == l)
        {
            if ((F.row  != t) && (F.col != s))
            {
                //int s = F.col - (F.col % block_size_col);// k

                for (int i = 0; i < F.row; i++)
                    for (int k = s; k < F.col; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                //int t = F.row - (F.row % block_size_row);// i

                for (int i = t; i < F.row; i++)
                    for (int k = 0; k < s; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
            }
            else if ((F.row != t) && (F.col == s))
            {
                //int t = F.row - (F.row % block_size_row);// i

                for (int i = t; i < F.row; i++)
                    for (int k = 0; k < F.col; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

            }
            else if ((F.row == t) && (F.col != s))
            {
                //int s = F.col - (F.col % block_size_col);// k

                for (int i = 0; i < F.row; i++)
                    for (int k = s; k < F.col; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

            }
        }
        else if (S.col != l)
        {
            if ((F.row != t) && (F.col != s))
            {
                //int s = F.col - (F.col % block_size_col);// k

                for (int i = 0; i < F.row; i++)
                    for (int k = s; k < F.col; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                //int t = F.row - (F.row % block_size_row);// i

                for (int i = t; i < F.row; i++)
                    for (int k = 0; k < s; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                //int l = S.col - (S.col % block_size_row);// j

                for (int i = 0; i < t; i++)
                    for (int k = 0; k < s; k++)
#pragma omp simd
                        for (int j = l; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
            }
            else if ((F.row != t) && (F.col == s))
            {
                //int t = F.row - (F.row % block_size_row);// i

                for (int i = t; i < F.row; i++)
                    for (int k = 0; k < F.col; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                //int l = S.col - (S.col % block_size_row);// j

                for (int i = 0; i < t; i++)
                    for (int k = 0; k < F.col; k++)
#pragma omp simd
                        for (int j = l; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
            }
            else if ((F.row == t) && (F.col != s))
            {
                //int s = F.col - (F.col % block_size_col);// k

                for (int i = 0; i < F.row; i++)
                    for (int k = s; k < F.col; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                //int l = S.col - (S.col % block_size_row);// j

                for (int i = 0; i < F.row; i++)
                    for (int k = 0; k < s; k++)
#pragma omp simd
                        for (int j = l; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
            }
            else if ((F.row == t) && (F.col == s))
            {
                //int l = S.col - (S.col % block_size_row);// j

                for (int i = 0; i < F.row; i++)
                    for (int k = 0; k < F.col; k++)
#pragma omp simd
                        for (int j = l; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
            }
        }

    }
    inline
        friend void parallel_block_mult(matrix& F, matrix& S, matrix& RES)
    {
        if ((F.col != S.row) || (F.row != RES.row) || (S.col != RES.col)) throw std::invalid_argument("matrices sizes should match!");
        if ((&F == &RES) || (&S == &RES)) throw std::invalid_argument("RES cannot be used as argument F or S");
              
        const int block_size_row = 64, block_size_col = 64;

        const auto processor_count = std::thread::hardware_concurrency();// / 2;

        std::memset(RES.data, 0, RES.row * RES.col * sizeof(T) );

        int t = F.row - (F.row % block_size_row);// i
        int l = S.col - (S.col % block_size_row);// j
        int s = F.col - (F.col % block_size_col);// k

#pragma omp parallel for num_threads(processor_count)
        for (int i1 = 0; i1 < t; i1 += block_size_row)
            for (int k1 = 0; k1 < s; k1 += block_size_col)
                for (int j1 = 0; j1 < l; j1 += block_size_row)
                    for (int i2 = i1; i2 < i1 + block_size_row; i2++)
                        for (int k2 = k1; k2 < k1 + block_size_col; k2++)
#pragma omp simd
                            for (int j2 = j1; j2 < j1 + block_size_row; j2++)
                                RES[i2 * RES.col + j2] += F[i2 * F.col + k2] * S[k2 * S.col + j2];

        if (S.col == l)
        {
            if ((F.row != t) && (F.col != s))
            {
                //int s = F.col - (F.col % block_size_col);// k
#pragma omp parallel for
                for (int i = 0; i < F.row; i++)
                    for (int k = s; k < F.col; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                //int t = F.row - (F.row % block_size_row);// i
#pragma omp parallel for
                for (int i = t; i < F.row; i++)
                    for (int k = 0; k < s; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
            }
            else if ((F.row != t) && (F.col == s))
            {
                //int t = F.row - (F.row % block_size_row);// i
#pragma omp parallel for
                for (int i = t; i < F.row; i++)
                    for (int k = 0; k < F.col; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

            }
            else if ((F.row == t) && (F.col!= s))
            {
                //int s = F.col - (F.col % block_size_col);// k
#pragma omp parallel for
                for (int i = 0; i < F.row; i++)
                    for (int k = s; k < F.col; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

            }
        }
        else if (S.col != l)
        {
            if ((F.row != t) && (F.col != s))
            {
                //int s = F.col - (F.col % block_size_col);// k
#pragma omp parallel for
                for (int i = 0; i < F.row; i++)
                    for (int k = s; k < F.col; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                //int t = F.row - (F.row % block_size_row);// i
#pragma omp parallel for
                for (int i = t; i < F.row; i++)
                    for (int k = 0; k < s; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                //int l = S.col - (S.col % block_size_row);// j
#pragma omp parallel for
                for (int i = 0; i < t; i++)
                    for (int k = 0; k < s; k++)
#pragma omp simd
                        for (int j = l; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
            }
            else if ((F.row != t) && (F.col == s))
            {
                //int t = F.row - (F.row % block_size_row);// i
#pragma omp parallel for
                for (int i = t; i < F.row; i++)
                    for (int k = 0; k < F.col; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                //int l = S.col - (S.col % block_size_row);// j
#pragma omp parallel for
                for (int i = 0; i < t; i++)
                    for (int k = 0; k < F.col; k++)
#pragma omp simd
                        for (int j = l; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
            }
            else if ((F.row == t) && (F.col != s))
            {
                //int s = F.col - (F.col % block_size_col);// k
#pragma omp parallel for
                for (int i = 0; i < F.row; i++)
                    for (int k = s; k < F.col; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                //int l = S.col - (S.col % block_size_row);// j
#pragma omp parallel for
                for (int i = 0; i < F.row; i++)
                    for (int k = 0; k < s; k++)
#pragma omp simd
                        for (int j = l; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
            }
            else if ((F.row == t) && (F.col == s))
            {
                //int l = S.col - (S.col % block_size_row);// j
#pragma omp parallel for
                for (int i = 0; i < F.row; i++)
                    for (int k = 0; k < F.col; k++)
#pragma omp simd
                        for (int j = l; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
            }
        }
    }
    inline
        friend void parallel_block_mult2(matrix& F, matrix& S, matrix& RES)
    {
        if ((F.col != S.row) || (F.row != RES.row) || (S.col != RES.col)) throw std::invalid_argument("matrices sizes should match!");
        if ((&F == &RES) || (&S == &RES)) throw std::invalid_argument("RES cannot be used as argument F or S");

        const int block_size_row = 128, block_size_col = 256;

        const auto processor_count = std::thread::hardware_concurrency();// / 2;

        std::memset(RES.data, 0, RES.row * RES.col * sizeof(T));

        int t = F.row - (F.row % block_size_row);// i
        int l = S.col - (S.col % block_size_row);// j
        int s = F.col - (F.col % block_size_col);// k

            matrix<T> tmp(block_size_row, block_size_col);

            for (int j1 = 0; j1 < l; j1 += block_size_row)
                for (int k1 = 0; k1 < s; k1 += block_size_col)
                {
//???                    
#pragma omp parallel for num_threads(processor_count) collapse(2)
                    for (int k = k1; k < k1 + block_size_col; k++)
                        for (int j = j1; j < j1 + block_size_row; j++)
                            tmp[(j - j1) * tmp.col + (k - k1)] = S[k * S.col + j];
                        
#pragma omp parallel for num_threads(processor_count)
                    for (int i1 = 0; i1 < t; i1 += block_size_row)

//#pragma omp parallel for num_threads(processor_count)
                        for (int i2 = i1; i2 < i1 + block_size_row; i2++)
                            for (int j2 = j1; j2 < j1 + block_size_row; j2++)
#pragma omp simd
                                for (int k2 = k1; k2 < k1 + block_size_col; k2++)
                                    RES[i2 * RES.col + j2] += F[i2 * F.col + k2] * tmp[(j2-j1) * tmp.col + (k2 - k1)];
                }


        if (S.col == l)
        {
            if ((F.row != t) && (F.col != s))
            {
                //int s = F.col - (F.col % block_size_col);// k
#pragma omp parallel for num_threads(processor_count) 
                for (int i = 0; i < F.row; i++)
                    for (int k = s; k < F.col; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                //int t = F.row - (F.row % block_size_row);// i
#pragma omp parallel for num_threads(processor_count) 
                for (int i = t; i < F.row; i++)
                    for (int k = 0; k < s; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
            }
            else if ((F.row != t) && (F.col == s))
            {
                //int t = F.row - (F.row % block_size_row);// i
#pragma omp parallel for num_threads(processor_count) 
                for (int i = t; i < F.row; i++)
                    for (int k = 0; k < F.col; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

            }
            else if ((F.row == t) && (F.col != s))
            {
                //int s = F.col - (F.col % block_size_col);// k
#pragma omp parallel for num_threads(processor_count) 
                for (int i = 0; i < F.row; i++)
                    for (int k = s; k < F.col; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

            }
        }
        else if (S.col != l)
        {
            if ((F.row != t) && (F.col != s))
            {
                //int s = F.col - (F.col % block_size_col);// k
#pragma omp parallel for num_threads(processor_count) 
                for (int i = 0; i < F.row; i++)
                    for (int k = s; k < F.col; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                //int t = F.row - (F.row % block_size_row);// i
#pragma omp parallel for num_threads(processor_count) 
                for (int i = t; i < F.row; i++)
                    for (int k = 0; k < s; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                //int l = S.col - (S.col % block_size_row);// j
#pragma omp parallel for num_threads(processor_count) 
                for (int i = 0; i < t; i++)
                    for (int k = 0; k < s; k++)
#pragma omp simd
                        for (int j = l; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
            }
            else if ((F.row != t) && (F.col == s))
            {
                //int t = F.row - (F.row % block_size_row);// i
#pragma omp parallel for num_threads(processor_count) 
                for (int i = t; i < F.row; i++)
                    for (int k = 0; k < F.col; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                //int l = S.col - (S.col % block_size_row);// j
#pragma omp parallel for num_threads(processor_count) 
                for (int i = 0; i < t; i++)
                    for (int k = 0; k < F.col; k++)
#pragma omp simd
                        for (int j = l; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
            }
            else if ((F.row == t) && (F.col != s))
            {
                //int s = F.col - (F.col % block_size_col);// k
#pragma omp parallel for num_threads(processor_count) 
                for (int i = 0; i < F.row; i++)
                    for (int k = s; k < F.col; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                //int l = S.col - (S.col % block_size_row);// j
#pragma omp parallel for num_threads(processor_count) 
                for (int i = 0; i < F.row; i++)
                    for (int k = 0; k < s; k++)
#pragma omp parallel for num_threads(processor_count) 
                        for (int j = l; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
            }
            else if ((F.row == t) && (F.col == s))
            {
                //int l = S.col - (S.col % block_size_row);// j
#pragma omp parallel for num_threads(processor_count) 
                for (int i = 0; i < F.row; i++)
                    for (int k = 0; k < F.col; k++)
#pragma omp simd
                        for (int j = l; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
            }
        }
    }
    inline // let block_size_row % sub_block_size == 0 && block_size_col % sub_block_size == 0
        friend void parallel_block_mult3(matrix& F, matrix& S, matrix& RES)
    {
        if ((F.col != S.row) || (F.row != RES.row) || (S.col != RES.col)) throw std::invalid_argument("matrices sizes should match!");
        if ((&F == &RES) || (&S == &RES)) throw std::invalid_argument("RES cannot be used as argument F or S");
        
        const int block_size_row = 128, block_size_col = 256, sub_block_size = 64;

        const auto processor_count = std::thread::hardware_concurrency();// / 2;

        std::memset(RES.data, 0, RES.row * RES.col * sizeof(T));

        int t = F.row - (F.row % block_size_row);// i
        int l = S.col - (S.col % block_size_row);// j
        int s = F.col - (F.col % block_size_col);// k

        // let block_size_row % sub_block_size == 0 && block_size_col % sub_block_size == 0

#pragma omp parallel for num_threads(processor_count)
        for (int i1 = 0; i1 < t; i1 += block_size_row)
            for (int k1 = 0; k1 < s; k1 += block_size_col)
                for (int j1 = 0; j1 < l; j1 += block_size_row)
                    
//#pragma omp parallel for num_threads(processor_count)              
                    for (int i2 = i1; i2 < i1 + block_size_row; i2 += sub_block_size)
                        for (int k2 = k1; k2 < k1 + block_size_col; k2 += sub_block_size)
                            for (int j2 = j1; j2 < j1 + block_size_row; j2 += sub_block_size)
                                
//#pragma omp parallel for num_threads(processor_count)      
                                for (int i3 = i2; i3 < i2 + sub_block_size; i3++)
                                    for (int k3 = k2; k3 < k2 + sub_block_size; k3++)
#pragma omp simd
                                        for (int j3 = j2; j3 < j2 + sub_block_size; j3++)
                                            RES[i3 * RES.col + j3] += F[i3 * F.col + k3] * S[k3 * S.col + j3];

        if (S.col == l)
        {
            if ((F.row != t) && (F.col != s))
            {
                //int s = F.col - (F.col % block_size_col);// k
#pragma omp parallel for num_threads(processor_count)
                for (int i = 0; i < F.row; i++)
                    for (int k = s; k < F.col; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                //int t = F.row - (F.row % block_size_row);// i
#pragma omp parallel for num_threads(processor_count)
                for (int i = t; i < F.row; i++)
                    for (int k = 0; k < s; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
            }
            else if ((F.row != t) && (F.col == s))
            {
                //int t = F.row - (F.row % block_size_row);// i
#pragma omp parallel for num_threads(processor_count)
                for (int i = t; i < F.row; i++)
                    for (int k = 0; k < F.col; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

            }
            else if ((F.row == t) && (F.col != s))
            {
                //int s = F.col - (F.col % block_size_col);// k
#pragma omp parallel for num_threads(processor_count)
                for (int i = 0; i < F.row; i++)
                    for (int k = s; k < F.col; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

            }
        }
        else if (S.col != l)
        {
            if ((F.row != t) && (F.col != s))
            {
                //int s = F.col - (F.col % block_size_col);// k
#pragma omp parallel for num_threads(processor_count)
                for (int i = 0; i < F.row; i++)
                    for (int k = s; k < F.col; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                //int t = F.row - (F.row % block_size_row);// i
#pragma omp parallel for num_threads(processor_count)
                for (int i = t; i < F.row; i++)
                    for (int k = 0; k < s; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                //int l = S.col - (S.col % block_size_row);// j
#pragma omp parallel for num_threads(processor_count)
                for (int i = 0; i < t; i++)
                    for (int k = 0; k < s; k++)
#pragma omp simd
                        for (int j = l; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
            }
            else if ((F.row != t) && (F.col == s))
            {
                //int t = F.row - (F.row % block_size_row);// i
#pragma omp parallel for num_threads(processor_count)
                for (int i = t; i < F.row; i++)
                    for (int k = 0; k < F.col; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                //int l = S.col - (S.col % block_size_row);// j
#pragma omp parallel for num_threads(processor_count)
                for (int i = 0; i < t; i++)
                    for (int k = 0; k < F.col; k++)
#pragma omp simd
                        for (int j = l; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
            }
            else if ((F.row == t) && (F.col != s))
            {
                //int s = F.col - (F.col % block_size_col);// k
#pragma omp parallel for num_threads(processor_count)
                for (int i = 0; i < F.row; i++)
                    for (int k = s; k < F.col; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                //int l = S.col - (S.col % block_size_row);// j
#pragma omp parallel for num_threads(processor_count)
                for (int i = 0; i < F.row; i++)
                    for (int k = 0; k < s; k++)
#pragma omp simd
                        for (int j = l; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
            }
            else if ((F.row == t) && (F.col == s))
            {
                //int l = S.col - (S.col % block_size_row);// j
#pragma omp parallel for num_threads(processor_count)
                for (int i = 0; i < F.row; i++)
                    for (int k = 0; k < F.col; k++)
#pragma omp simd
                        for (int j = l; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
            }
        }
    }
    inline
        friend void parallel_block_mult4(matrix& F, matrix& S, matrix& RES)
    {
        if ((F.col != S.row) || (F.row != RES.row) || (S.col != RES.col)) throw std::invalid_argument("matrices sizes should match!");
        if ((&F == &RES) || (&S == &RES)) throw std::invalid_argument("RES cannot be used as argument F or S");
        
        const int block_size_row = 128, block_size_col = 256, sub_block_size = 64;

        const auto processor_count = std::thread::hardware_concurrency();// / 2;

        std::memset(RES.data, 0, RES.row * RES.col * sizeof(T));

        int t = F.row - (F.row % block_size_row);// i
        int l = S.col - (S.col % block_size_row);// j
        int s = F.col - (F.col % block_size_col);// k

        // let block_size_row % sub_block_size == 0 && block_size_col % sub_block_size == 0

        matrix<T> tmp(block_size_row, block_size_col);

        for (int j1 = 0; j1 < l; j1 += block_size_row)
            for (int k1 = 0; k1 < s; k1 += block_size_col)
            {

//#pragma omp parallel for num_threads(processor_count) collapse(2)
                for (int k = k1; k < k1 + block_size_col; k++)
                    for (int j = j1; j < j1 + block_size_row; j++)
                        tmp[(j - j1) * tmp.col + (k - k1)] = S[k * S.col + j];

#pragma omp parallel for num_threads(processor_count)
                for (int i1 = 0; i1 < t; i1 += block_size_row)

//#pragma omp parallel for num_threads(processor_count)
                    for (int i2 = i1; i2 < i1 + block_size_row; i2 += sub_block_size)
                    for (int j2 = j1; j2 < j1 + block_size_row; j2 += sub_block_size)
                    for (int k2 = k1; k2 < k1 + block_size_col; k2 += sub_block_size)

#pragma omp parallel for num_threads(processor_count)
                    for (int i3 = i2; i3 < i2 + sub_block_size; i3++)
                        for (int j3 = j2; j3 < j2 + sub_block_size; j3++)
#pragma omp simd
                            for (int k3 = k2; k3 < k2 + sub_block_size; k3++)
                                RES[i3 * RES.col + j3] += F[i3 * F.col + k3] * tmp[(j3 - j1) * tmp.col + (k3 - k1)];
            }


        if (S.col == l)
        {
            if ((F.row != t) && (F.col != s))
            {
                //int s = F.col - (F.col % block_size_col);// k
#pragma omp parallel for num_threads(processor_count) 
                for (int i = 0; i < F.row; i++)
                    for (int k = s; k < F.col; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                //int t = F.row - (F.row % block_size_row);// i
#pragma omp parallel for num_threads(processor_count) 
                for (int i = t; i < F.row; i++)
                    for (int k = 0; k < s; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
            }
            else if ((F.row != t) && (F.col == s))
            {
                //int t = F.row - (F.row % block_size_row);// i
#pragma omp parallel for num_threads(processor_count) 
                for (int i = t; i < F.row; i++)
                    for (int k = 0; k < F.col; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

            }
            else if ((F.row == t) && (F.col != s))
            {
                //int s = F.col - (F.col % block_size_col);// k
#pragma omp parallel for num_threads(processor_count) 
                for (int i = 0; i < F.row; i++)
                    for (int k = s; k < F.col; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

            }
        }
        else if (S.col != l)
        {
            if ((F.row != t) && (F.col != s))
            {
                //int s = F.col - (F.col % block_size_col);// k
#pragma omp parallel for num_threads(processor_count) 
                for (int i = 0; i < F.row; i++)
                    for (int k = s; k < F.col; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                //int t = F.row - (F.row % block_size_row);// i
#pragma omp parallel for num_threads(processor_count) 
                for (int i = t; i < F.row; i++)
                    for (int k = 0; k < s; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                //int l = S.col - (S.col % block_size_row);// j
#pragma omp parallel for num_threads(processor_count) 
                for (int i = 0; i < t; i++)
                    for (int k = 0; k < s; k++)
#pragma omp simd
                        for (int j = l; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
            }
            else if ((F.row != t) && (F.col == s))
            {
                //int t = F.row - (F.row % block_size_row);// i
#pragma omp parallel for num_threads(processor_count) 
                for (int i = t; i < F.row; i++)
                    for (int k = 0; k < F.col; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                //int l = S.col - (S.col % block_size_row);// j
#pragma omp parallel for num_threads(processor_count) 
                for (int i = 0; i < t; i++)
                    for (int k = 0; k < F.col; k++)
#pragma omp simd
                        for (int j = l; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
            }
            else if ((F.row == t) && (F.col != s))
            {
                //int s = F.col - (F.col % block_size_col);// k
#pragma omp parallel for num_threads(processor_count) 
                for (int i = 0; i < F.row; i++)
                    for (int k = s; k < F.col; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                //int l = S.col - (S.col % block_size_row);// j
#pragma omp parallel for num_threads(processor_count) 
                for (int i = 0; i < F.row; i++)
                    for (int k = 0; k < s; k++)
#pragma omp parallel for num_threads(processor_count) 
                        for (int j = l; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
            }
            else if ((F.row == t) && (F.col == s))
            {
                //int l = S.col - (S.col % block_size_row);// j
#pragma omp parallel for num_threads(processor_count) 
                for (int i = 0; i < F.row; i++)
                    for (int k = 0; k < F.col; k++)
#pragma omp simd
                        for (int j = l; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
            }
        }
    }

    inline // let block_size_row % sub_block_size == 0 && block_size_col % sub_block_size == 0
        friend void parallel_block_mult5(matrix& F, matrix& S, matrix& RES)
    {
        if ((F.col != S.row) || (F.row != RES.row) || (S.col != RES.col)) throw std::invalid_argument("matrices sizes should match!");
        if ((&F == &RES) || (&S == &RES)) throw std::invalid_argument("RES cannot be used as argument F or S");

        const int block_size_row = 128, block_size_col = 256, sub_block_size = 64;
        
        const int sub_sub_block_size = 8;

        /*const int sub_sub_block_size = 4;*/

        const auto processor_count = std::thread::hardware_concurrency();// / 2;

        std::memset(RES.data, 0, RES.row * RES.col * sizeof(T));

        int t = F.row - (F.row % block_size_row);// i
        int l = S.col - (S.col % block_size_row);// j
        int s = F.col - (F.col % block_size_col);// k

        // let block_size_row % sub_block_size == 0 && block_size_col % sub_block_size == 0

#pragma omp parallel for collapse (1) num_threads(processor_count)
        for (int i1 = 0; i1 < t; i1 += block_size_row)
            for (int k1 = 0; k1 < s; k1 += block_size_col)
                for (int j1 = 0; j1 < l; j1 += block_size_row)
                    

//#pragma omp parallel for num_threads(processor_count)              
                    for (int i2 = i1; i2 < i1 + block_size_row; i2 += sub_block_size)
                        for (int k2 = k1; k2 < k1 + block_size_col; k2 += sub_block_size)
                            for (int j2 = j1; j2 < j1 + block_size_row; j2 += sub_block_size)

//#pragma omp parallel for num_threads(processor_count)      
                                for (int i3 = i2; i3 < i2 + sub_block_size; i3+=sub_sub_block_size)
                                    for (int k3 = k2; k3 < k2 + sub_block_size; k3+=sub_sub_block_size)
//#pragma omp simd
                                        for (int j3 = j2; j3 < j2 + sub_block_size; j3+= sub_sub_block_size)
                                        {
                                            
                                                //#1
                                                    
                                            //  /*
                                                
                                                // 8 x 8
                                                // vector loads     loads       vector stores       FMA
                                                // 16               64          16                  64

                                                // ? count of registers per cor
                                                // parallel

                                                __m512 c[sub_sub_block_size]{};
                                                __m512 a, b[sub_sub_block_size]{};

                                                for (int k4 = 0; k4 < sub_sub_block_size; k4++)
                                                    b[k4] = _mm512_loadu_pd(&S[(k4+k3) * S.col + j3]);
                                                //std::cout << "!";
                                                for (int i4 = 0; i4 < sub_sub_block_size; i4++)
                                                    c[i4] = _mm512_loadu_pd(&RES[(i4 + i3) * RES.col + j3]); //  _mm512_load_pd
                                                    //c[i4] = _mm512_load_pd(&RES.data + ((i4 + i3) * RES.col + j3) * sizeof(T) );
                                                //std::cout << "?";
                                                for (int i5 = 0; i5 < sub_sub_block_size; i5++)
                                                    for(int k5 = 0; k5 < sub_sub_block_size; k5++)
                                                    {
                                                        //RES[i3 * RES.col + j3] += F[i3 * F.col + k3] * S[k3 * S.col + j3];
                                                        a = _mm512_set1_pd(F[ (i3+i5) * F.col + (k3 + k5)]);

                                                        c[i5] = _mm512_fmadd_pd(a, b[k5],c[i5]);

                                                    }

                                                for (int i6 = 0; i6 < sub_sub_block_size; i6++)
                                                    _mm512_storeu_pd(&RES[(i6+i3)*RES.col + j3], c[i6]);

                                            //  */

                                                //#2  
                                            /*
                                                                                           
                                            
                                                __m512 a, b, c;

                                                for (int i5 = 0; i5 < sub_sub_block_size; i5++)
                                                {
                                                    c = _mm512_loadu_pd(&RES[(i5 + i3) * RES.col + j3]);
                                                    for (int k5 = 0; k5 < sub_sub_block_size; k5++)
                                                    {
                                                        //RES[i3 * RES.col + j3] += F[i3 * F.col + k3] * S[k3 * S.col + j3];
                                                        a = _mm512_set1_pd(F[(i3 + i5) * F.col + (k3 + k5)]);

                                                        b = _mm512_loadu_pd(&S[(k5 + k3) * S.col + j3]);

                                                        c = _mm512_fmadd_pd(a, b, c);

                                                    }
                                                    _mm512_storeu_pd(&RES[(i5 + i3) * RES.col + j3], c);
                                                }

                                            */

                                                //#3
                                            /*
                                                 
                                             
                                                __m256 c[sub_sub_block_size]{};
                                                __m256 a, b[sub_sub_block_size]{};

                                                // std::cout << "0";

                                                for (int k4 = 0; k4 < sub_sub_block_size; k4++)
                                                    b[k4] = _mm256_loadu_pd(&S[(k4 + k3) * S.col + j3]);
                                                //std::cout << "!";
                                                for (int i4 = 0; i4 < sub_sub_block_size; i4++)
                                                    c[i4] = _mm256_loadu_pd(&RES[(i4 + i3) * RES.col + j3]); //  _mm512_load_pd
                                                //c[i4] = _mm512_load_pd(&RES.data + ((i4 + i3) * RES.col + j3) * sizeof(T) );
                                            //std::cout << "?";
                                                for (int i5 = 0; i5 < sub_sub_block_size; i5++)
                                                    for (int k5 = 0; k5 < sub_sub_block_size; k5++)
                                                    {
                                                        //RES[i3 * RES.col + j3] += F[i3 * F.col + k3] * S[k3 * S.col + j3];
                                                        a = _mm256_set1_pd(F[(i3 + i5) * F.col + (k3 + k5)]);

                                                        c[i5] = _mm256_fmadd_pd(a, b[k5], c[i5]);

                                                    }

                                                for (int i6 = 0; i6 < sub_sub_block_size; i6++)
                                                    _mm256_storeu_pd(&RES[(i6 + i3) * RES.col + j3], c[i6]);
                                            */

                                                //#4
                                            /*
                                                

                                                __m256 a, b, c;

                                                for (int i5 = 0; i5 < sub_sub_block_size; i5++)
                                                {
                                                    c = _mm256_loadu_pd(&RES[(i5 + i3) * RES.col + j3]);
                                                    for (int k5 = 0; k5 < sub_sub_block_size; k5++)
                                                    {
                                                        //RES[i3 * RES.col + j3] += F[i3 * F.col + k3] * S[k3 * S.col + j3];
                                                        a = _mm256_set1_pd(F[(i3 + i5) * F.col + (k3 + k5)]);

                                                        b = _mm256_loadu_pd(&S[(k5 + k3) * S.col + j3]);

                                                        c = _mm256_fmadd_pd(a, b, c);

                                                    }
                                                    _mm256_storeu_pd(&RES[(i5 + i3) * RES.col + j3], c);
                                                }

                                            */
                                        }
                                            

        if (S.col == l)
        {
            if ((F.row != t) && (F.col != s))
            {
                //int s = F.col - (F.col % block_size_col);// k
#pragma omp parallel for num_threads(processor_count)
                for (int i = 0; i < F.row; i++)
                    for (int k = s; k < F.col; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                //int t = F.row - (F.row % block_size_row);// i
#pragma omp parallel for num_threads(processor_count)
                for (int i = t; i < F.row; i++)
                    for (int k = 0; k < s; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
            }
            else if ((F.row != t) && (F.col == s))
            {
                //int t = F.row - (F.row % block_size_row);// i
#pragma omp parallel for num_threads(processor_count)
                for (int i = t; i < F.row; i++)
                    for (int k = 0; k < F.col; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

            }
            else if ((F.row == t) && (F.col != s))
            {
                //int s = F.col - (F.col % block_size_col);// k
#pragma omp parallel for num_threads(processor_count)
                for (int i = 0; i < F.row; i++)
                    for (int k = s; k < F.col; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

            }
        }
        else if (S.col != l)
        {
            if ((F.row != t) && (F.col != s))
            {
                //int s = F.col - (F.col % block_size_col);// k
#pragma omp parallel for num_threads(processor_count)
                for (int i = 0; i < F.row; i++)
                    for (int k = s; k < F.col; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                //int t = F.row - (F.row % block_size_row);// i
#pragma omp parallel for num_threads(processor_count)
                for (int i = t; i < F.row; i++)
                    for (int k = 0; k < s; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                //int l = S.col - (S.col % block_size_row);// j
#pragma omp parallel for num_threads(processor_count)
                for (int i = 0; i < t; i++)
                    for (int k = 0; k < s; k++)
#pragma omp simd
                        for (int j = l; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
            }
            else if ((F.row != t) && (F.col == s))
            {
                //int t = F.row - (F.row % block_size_row);// i
#pragma omp parallel for num_threads(processor_count)
                for (int i = t; i < F.row; i++)
                    for (int k = 0; k < F.col; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                //int l = S.col - (S.col % block_size_row);// j
#pragma omp parallel for num_threads(processor_count)
                for (int i = 0; i < t; i++)
                    for (int k = 0; k < F.col; k++)
#pragma omp simd
                        for (int j = l; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
            }
            else if ((F.row == t) && (F.col != s))
            {
                //int s = F.col - (F.col % block_size_col);// k
#pragma omp parallel for num_threads(processor_count)
                for (int i = 0; i < F.row; i++)
                    for (int k = s; k < F.col; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                //int l = S.col - (S.col % block_size_row);// j
#pragma omp parallel for num_threads(processor_count)
                for (int i = 0; i < F.row; i++)
                    for (int k = 0; k < s; k++)
#pragma omp simd
                        for (int j = l; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
            }
            else if ((F.row == t) && (F.col == s))
            {
                //int l = S.col - (S.col % block_size_row);// j
#pragma omp parallel for num_threads(processor_count)
                for (int i = 0; i < F.row; i++)
                    for (int k = 0; k < F.col; k++)
#pragma omp simd
                        for (int j = l; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
            }
        }
    }

    inline // let block_size_row % sub_block_size == 0 && block_size_col % sub_block_size == 0
        friend void parallel_block_mult6(matrix& F, matrix& S, matrix& RES)
    {
        if ((F.col != S.row) || (F.row != RES.row) || (S.col != RES.col)) throw std::invalid_argument("matrices sizes should match!");
        if ((&F == &RES) || (&S == &RES)) throw std::invalid_argument("RES cannot be used as argument F or S");

        const int block_size_row = 128, block_size_col = 256, sub_block_size = 64;

        const int sub_sub_block_size = 8, sub_sub_block_size2 = 16;

        /*const int sub_sub_block_size = 4;*/

        const auto processor_count = std::thread::hardware_concurrency();// / 2;

        std::memset(RES.data, 0, RES.row * RES.col * sizeof(T));

        int t = F.row - (F.row % block_size_row);// i
        int l = S.col - (S.col % block_size_row);// j
        int s = F.col - (F.col % block_size_col);// k

        // let block_size_row % sub_block_size == 0 && block_size_col % sub_block_size == 0

#pragma omp parallel for collapse (1) num_threads(processor_count)
        for (int i1 = 0; i1 < t; i1 += block_size_row)
            for (int k1 = 0; k1 < s; k1 += block_size_col)
                for (int j1 = 0; j1 < l; j1 += block_size_row)


                    //#pragma omp parallel for num_threads(processor_count)              
                    for (int i2 = i1; i2 < i1 + block_size_row; i2 += sub_block_size)
                        for (int k2 = k1; k2 < k1 + block_size_col; k2 += sub_block_size)
                            for (int j2 = j1; j2 < j1 + block_size_row; j2 += sub_block_size)

                                //#pragma omp parallel for num_threads(processor_count)      
                                for (int i3 = i2; i3 < i2 + sub_block_size; i3 += sub_sub_block_size)
                                    for (int k3 = k2; k3 < k2 + sub_block_size; k3 += sub_sub_block_size2)
                                        //#pragma omp simd
                                        for (int j3 = j2; j3 < j2 + sub_block_size; j3 += sub_sub_block_size)
                                        {

                                            //#1

                                        //  /*

                                            // (8 x 16)  (16 x 8)  -> (8 x 8)
                                            // vector loads     loads       vector stores       FMA
                                            // 24               128         8                   128

                                            __m512 c[sub_sub_block_size]{};
                                            __m512 a, b[sub_sub_block_size2]{};

                                            for (int k4 = 0; k4 < sub_sub_block_size2; k4++)
                                                b[k4] = _mm512_loadu_pd(&S[(k4 + k3) * S.col + j3]);
                                            //std::cout << "!";
                                            for (int i4 = 0; i4 < sub_sub_block_size; i4++)
                                                c[i4] = _mm512_loadu_pd(&RES[(i4 + i3) * RES.col + j3]); //  _mm512_load_pd
                                            //c[i4] = _mm512_load_pd(&RES.data + ((i4 + i3) * RES.col + j3) * sizeof(T) );
                                        //std::cout << "?";
                                            for (int i5 = 0; i5 < sub_sub_block_size; i5++)
                                                for (int k5 = 0; k5 < sub_sub_block_size2; k5++)
                                                {
                                                    //RES[i3 * RES.col + j3] += F[i3 * F.col + k3] * S[k3 * S.col + j3];
                                                    a = _mm512_set1_pd(F[(i3 + i5) * F.col + (k3 + k5)]);

                                                    c[i5] = _mm512_fmadd_pd(a, b[k5], c[i5]);

                                                }

                                            for (int i6 = 0; i6 < sub_sub_block_size; i6++)
                                                _mm512_storeu_pd(&RES[(i6 + i3) * RES.col + j3], c[i6]);

                                            //  */

                                                //#2  
                                            /*


                                                __m512 a, b, c;

                                                for (int i5 = 0; i5 < sub_sub_block_size; i5++)
                                                {
                                                    c = _mm512_loadu_pd(&RES[(i5 + i3) * RES.col + j3]);
                                                    for (int k5 = 0; k5 < sub_sub_block_size; k5++)
                                                    {
                                                        //RES[i3 * RES.col + j3] += F[i3 * F.col + k3] * S[k3 * S.col + j3];
                                                        a = _mm512_set1_pd(F[(i3 + i5) * F.col + (k3 + k5)]);

                                                        b = _mm512_loadu_pd(&S[(k5 + k3) * S.col + j3]);

                                                        c = _mm512_fmadd_pd(a, b, c);

                                                    }
                                                    _mm512_storeu_pd(&RES[(i5 + i3) * RES.col + j3], c);
                                                }

                                            */

                                            //#3
                                        /*


                                            __m256 c[sub_sub_block_size]{};
                                            __m256 a, b[sub_sub_block_size]{};

                                            // std::cout << "0";

                                            for (int k4 = 0; k4 < sub_sub_block_size; k4++)
                                                b[k4] = _mm256_loadu_pd(&S[(k4 + k3) * S.col + j3]);
                                            //std::cout << "!";
                                            for (int i4 = 0; i4 < sub_sub_block_size; i4++)
                                                c[i4] = _mm256_loadu_pd(&RES[(i4 + i3) * RES.col + j3]); //  _mm512_load_pd
                                            //c[i4] = _mm512_load_pd(&RES.data + ((i4 + i3) * RES.col + j3) * sizeof(T) );
                                        //std::cout << "?";
                                            for (int i5 = 0; i5 < sub_sub_block_size; i5++)
                                                for (int k5 = 0; k5 < sub_sub_block_size; k5++)
                                                {
                                                    //RES[i3 * RES.col + j3] += F[i3 * F.col + k3] * S[k3 * S.col + j3];
                                                    a = _mm256_set1_pd(F[(i3 + i5) * F.col + (k3 + k5)]);

                                                    c[i5] = _mm256_fmadd_pd(a, b[k5], c[i5]);

                                                }

                                            for (int i6 = 0; i6 < sub_sub_block_size; i6++)
                                                _mm256_storeu_pd(&RES[(i6 + i3) * RES.col + j3], c[i6]);
                                        */

                                        //#4
                                    /*


                                        __m256 a, b, c;

                                        for (int i5 = 0; i5 < sub_sub_block_size; i5++)
                                        {
                                            c = _mm256_loadu_pd(&RES[(i5 + i3) * RES.col + j3]);
                                            for (int k5 = 0; k5 < sub_sub_block_size; k5++)
                                            {
                                                //RES[i3 * RES.col + j3] += F[i3 * F.col + k3] * S[k3 * S.col + j3];
                                                a = _mm256_set1_pd(F[(i3 + i5) * F.col + (k3 + k5)]);

                                                b = _mm256_loadu_pd(&S[(k5 + k3) * S.col + j3]);

                                                c = _mm256_fmadd_pd(a, b, c);

                                            }
                                            _mm256_storeu_pd(&RES[(i5 + i3) * RES.col + j3], c);
                                        }

                                    */
                                        }


        if (S.col == l)
        {
            if ((F.row != t) && (F.col != s))
            {
                //int s = F.col - (F.col % block_size_col);// k
#pragma omp parallel for num_threads(processor_count)
                for (int i = 0; i < F.row; i++)
                    for (int k = s; k < F.col; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                //int t = F.row - (F.row % block_size_row);// i
#pragma omp parallel for num_threads(processor_count)
                for (int i = t; i < F.row; i++)
                    for (int k = 0; k < s; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
            }
            else if ((F.row != t) && (F.col == s))
            {
                //int t = F.row - (F.row % block_size_row);// i
#pragma omp parallel for num_threads(processor_count)
                for (int i = t; i < F.row; i++)
                    for (int k = 0; k < F.col; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

            }
            else if ((F.row == t) && (F.col != s))
            {
                //int s = F.col - (F.col % block_size_col);// k
#pragma omp parallel for num_threads(processor_count)
                for (int i = 0; i < F.row; i++)
                    for (int k = s; k < F.col; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

            }
        }
        else if (S.col != l)
        {
            if ((F.row != t) && (F.col != s))
            {
                //int s = F.col - (F.col % block_size_col);// k
#pragma omp parallel for num_threads(processor_count)
                for (int i = 0; i < F.row; i++)
                    for (int k = s; k < F.col; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                //int t = F.row - (F.row % block_size_row);// i
#pragma omp parallel for num_threads(processor_count)
                for (int i = t; i < F.row; i++)
                    for (int k = 0; k < s; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                //int l = S.col - (S.col % block_size_row);// j
#pragma omp parallel for num_threads(processor_count)
                for (int i = 0; i < t; i++)
                    for (int k = 0; k < s; k++)
#pragma omp simd
                        for (int j = l; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
            }
            else if ((F.row != t) && (F.col == s))
            {
                //int t = F.row - (F.row % block_size_row);// i
#pragma omp parallel for num_threads(processor_count)
                for (int i = t; i < F.row; i++)
                    for (int k = 0; k < F.col; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                //int l = S.col - (S.col % block_size_row);// j
#pragma omp parallel for num_threads(processor_count)
                for (int i = 0; i < t; i++)
                    for (int k = 0; k < F.col; k++)
#pragma omp simd
                        for (int j = l; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
            }
            else if ((F.row == t) && (F.col != s))
            {
                //int s = F.col - (F.col % block_size_col);// k
#pragma omp parallel for num_threads(processor_count)
                for (int i = 0; i < F.row; i++)
                    for (int k = s; k < F.col; k++)
#pragma omp simd
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                //int l = S.col - (S.col % block_size_row);// j
#pragma omp parallel for num_threads(processor_count)
                for (int i = 0; i < F.row; i++)
                    for (int k = 0; k < s; k++)
#pragma omp simd
                        for (int j = l; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
            }
            else if ((F.row == t) && (F.col == s))
            {
                //int l = S.col - (S.col % block_size_row);// j
#pragma omp parallel for num_threads(processor_count)
                for (int i = 0; i < F.row; i++)
                    for (int k = 0; k < F.col; k++)
#pragma omp simd
                        for (int j = l; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
            }
        }
    }
};

template <class T>
std::istream& operator>> (std::istream& in, matrix<T>& m)
{
    for (int i = 0; i < m.getrow(); i++)
        for (int j = 0; j < m.getcol(); j++)
            in >> m[i * m.getcol() + j];
    return in;
}

template <class T>
std::ostream& operator<< (std::ostream& out, matrix<T>& m)
{
    for (int i = 0; i < m.getrow(); i++)
    {
        for (int j = 0; j < m.getcol(); j++)
            out << m[i * m.getcol() + j] << " ";

        out << "\n";
    }
    return out;
}