#pragma once
#include <omp.h>
#include <iostream>

const double MAXM = 1000;
const double MINM = -1000;

template <class T>
class matrix
{
private:
    int row;
    int col;
    T* data;
public:
    matrix(int _row = 1000, int _col = 1000) : row(_row), col(_col)
    {
        if ((row <= 0)||(col <= 0)) throw std::exception("size should be positive!");

        data = new T[row * col]();
    }
    matrix(matrix& m): row(m.row), col(m.col)
    {
        data = new T[row * col];

        for (int i = 0; i < row * col; i++)
            data[i] = m[i];
    }
    matrix(matrix&& m): row(m.row), col(m.col), data(m.data)
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
    void randomfill()noexcept
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

        if (row*col != m.row * m.col)
        {
            row = m.row;
            col = m.col;
            T* tmp = new T[row * col];
            delete[] data;
            data = tmp;
        }

        for (int i = 0; i < row * col; i++)
            data[i] = m[i];

        return *this;
    }
    matrix& operator=(matrix&& m) noexcept
    {
        delete[] data;

        data = m.data;

        row = m.row;

        col = m.col;

        m.data = nullptr;

        m.row = m.col = 0;

        return *this;
    }
    bool operator == (const matrix & m) const noexcept
    {
        if ((row != m.row)||(col != m.col)) return false;

        for (int i = 0; i < row * col; i++)
            if (data[i] != m[i]) return false;

        return true;
    }
    bool operator!= (const matrix & m) const
    {
        return !(*this == m);
    }
    matrix operator*(const matrix& m) 
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
        if ((col != m.col)||(row != m.row)) throw std::invalid_argument("matrices sizes should match!");

        matrix res(*this);
        for (int i = 0; i < row * col; i++)
                res[i] -= m[i];

        return res;
    }
    T norm() 
    {
        T res = static_cast<T>(0);

        for (int i = 0; i < row * col; i++)
            res += data[i] * data[i];

        return res;
    }
    inline
    friend void mult(matrix& F, matrix& S, matrix& RES)
    {
        if ((F.col != S.row)|| (F.row != RES.row) || (S.col != RES.col)) throw std::invalid_argument("matrices sizes should match!");

        for (int i = 0; i < F.row; i++)
            for (int k = 0; k < F.col; k++)
                for (int j = 0; j < S.col; j++)
                    RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
    }
    inline
    friend void block_mult(matrix& F, matrix& S, matrix& RES, int block_size = 40)
    {
        if ((F.col != S.row) || (F.row != RES.row) || (S.col != RES.col)) throw std::invalid_argument("matrices sizes should match!");

        int t = F.row - (F.row % block_size);
        int l = S.col - (S.col % block_size);
        int s = F.col - (F.col % block_size);

        for (int i1 = 0; i1 < t; i1 += block_size)
            for (int j1 = 0; j1 < l; j1 += block_size)
                for (int k1 = 0; k1 < s; k1 += block_size)
                    for (int i2 = i1; i2 < i1 + block_size; i2++)
                        for (int k2 = k1; k2 < k1 + block_size; k2++)
                            for (int j2 = j1; j2 < j1 + block_size; j2++)
                                RES[i2 * RES.col + j2] += F[i2 * F.col + k2] * S[k2 * S.col + j2];

        if (S.col % block_size == 0)
        {
            if (((F.row % block_size) != 0) && ((F.col % block_size) != 0))
            {
                int s = F.col - (F.col % block_size);

                for (int i = 0; i < F.row; i++)
                    for (int k = s; k < F.col; k++)
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                int t = F.row - (F.row % block_size);

                for (int i = t; i < F.row; i++)
                    for (int k = 0; k < s; k++)
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
            }
            else if (((F.row % block_size) != 0) && ((F.col % block_size) == 0))
            {
                int t = F.row - (F.row % block_size);

                for (int i = t; i < F.row; i++)
                    for (int k = 0; k < F.col; k++)
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
                
            }
            else if (((F.row % block_size) == 0) && ((F.col % block_size) != 0))
            {
                int s = F.col - (F.col % block_size);

                for (int i = 0; i < F.row; i++)
                    for (int k = s; k < F.col; k++)
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

            }
        }
        else if (S.col % block_size != 0) 
        {
            if (((F.row % block_size) != 0) && ((F.col % block_size) != 0))
            {
                int s = F.col - (F.col % block_size);

                for (int i = 0; i < F.row; i++)
                    for (int k = s; k < F.col; k++)
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                int t = F.row - (F.row % block_size);

                for (int i = t; i < F.row; i++)
                    for (int k = 0; k < s; k++)
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                int l = S.col - (S.col % block_size);

                for (int i = 0; i < t; i++)
                    for (int k = 0; k < s; k++)
                        for (int j = l; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
            }
            else if (((F.row % block_size) != 0) && ((F.col % block_size) == 0))
            {
                int t = F.row - (F.row % block_size);

                for (int i = t; i < F.row; i++)
                    for (int k = 0; k < F.col; k++)
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                int l = S.col - (S.col % block_size);

                for (int i = 0; i < t; i++)
                    for (int k = 0; k < F.col; k++)
                        for (int j = l; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
            }
            else if (((F.row% block_size) == 0) && ((F.col% block_size) != 0))
            {
                int s = F.col - (F.col % block_size);

                for (int i = 0; i < F.row; i++)
                    for (int k = s; k < F.col; k++)
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                int l = S.col - (S.col % block_size);

                for (int i = 0; i < F.row; i++)
                    for (int k = 0; k < s; k++)
                        for (int j = l; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
            }
            else if (((F.row % block_size) == 0) && ((F.col % block_size) == 0))
            {
                int l = S.col - (S.col % block_size);

                for (int i = 0; i < F.row; i++)
                    for (int k = 0; k < F.col; k++)
                        for (int j = l; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
            }
        }

    }
    inline
    friend void parallel_block_mult(matrix& F, matrix& S, matrix& RES, int block_size = 40)
    {
        if ((F.col != S.row) || (F.row != RES.row) || (S.col != RES.col)) throw std::invalid_argument("matrices sizes should match!");

        int s = F.col - (F.col % block_size);
        int t = F.row - (F.row % block_size);
        int l = S.col - (S.col % block_size);

#pragma omp parallel for
        for (int i1 = 0; i1 < t; i1 += block_size)
            for (int k1 = 0; k1 < s; k1 += block_size)
                for (int j1 = 0; j1 < l; j1 += block_size)
                    for (int i2 = i1; i2 < i1 + block_size; i2++)
                        for (int k2 = k1; k2 < k1 + block_size; k2++)
#pragma omp simd
                            for (int j2 = j1; j2 < j1 + block_size; j2++)
                                RES[i2 * RES.col + j2] += F[i2 * F.col + k2] * S[k2 * S.col + j2];

        if (S.col % block_size == 0)
        {
            if (((F.row % block_size) != 0) && ((F.col % block_size) != 0))
            {
                int s = F.col - (F.col % block_size);

                for (int i = 0; i < F.row; i++)
                    for (int k = s; k < F.col; k++)
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                int t = F.row - (F.row % block_size);

                for (int i = t; i < F.row; i++)
                    for (int k = 0; k < s; k++)
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
            }
            else if (((F.row % block_size) != 0) && ((F.col % block_size) == 0))
            {
                int t = F.row - (F.row % block_size);

                for (int i = t; i < F.row; i++)
                    for (int k = 0; k < F.col; k++)
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

            }
            else if (((F.row % block_size) == 0) && ((F.col % block_size) != 0))
            {
                int s = F.col - (F.col % block_size);

                for (int i = 0; i < F.row; i++)
                    for (int k = s; k < F.col; k++)
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

            }
        }
        else if (S.col % block_size != 0)
        {
            if (((F.row % block_size) != 0) && ((F.col % block_size) != 0))
            {
                int s = F.col - (F.col % block_size);

                for (int i = 0; i < F.row; i++)
                    for (int k = s; k < F.col; k++)
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                int t = F.row - (F.row % block_size);

                for (int i = t; i < F.row; i++)
                    for (int k = 0; k < s; k++)
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                int l = S.col - (S.col % block_size);

                for (int i = 0; i < t; i++)
                    for (int k = 0; k < s; k++)
                        for (int j = l; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
            }
            else if (((F.row % block_size) != 0) && ((F.col % block_size) == 0))
            {
                int t = F.row - (F.row % block_size);

                for (int i = t; i < F.row; i++)
                    for (int k = 0; k < F.col; k++)
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                int l = S.col - (S.col % block_size);

                for (int i = 0; i < t; i++)
                    for (int k = 0; k < F.col; k++)
                        for (int j = l; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
            }
            else if (((F.row % block_size) == 0) && ((F.col % block_size) != 0))
            {
                int s = F.col - (F.col % block_size);

                for (int i = 0; i < F.row; i++)
                    for (int k = s; k < F.col; k++)
                        for (int j = 0; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];

                int l = S.col - (S.col % block_size);

                for (int i = 0; i < F.row; i++)
                    for (int k = 0; k < s; k++)
                        for (int j = l; j < S.col; j++)
                            RES[i * RES.col + j] += F[i * F.col + k] * S[k * S.col + j];
            }
            else if (((F.row % block_size) == 0) && ((F.col % block_size) == 0))
            {
                int l = S.col - (S.col % block_size);

                for (int i = 0; i < F.row; i++)
                    for (int k = 0; k < F.col; k++)
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
