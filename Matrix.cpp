#include "Matrix.h"

Matrix::Matrix()
{
   // br++;
   // cout << br << endl;
    n = 0;
    m = 0;
    a = new double *[0];
}

Matrix::Matrix(int _n, int _m)
{
    //br++;
    //cout << br << endl;
    n = _n;
    m = _m;
    a = new double* [n];
    for(int i = 0; i < n; i++)
    {
        a[i] = new double[m];
        for(int j = 0; j < m; j++)
            a[i][j] = 0.0;
    }
}

Matrix::~Matrix()
{
    //br--;
    //cout << br << endl;
    for(int i = 0; i < n; i++)
        delete[] a[i];
    delete[] a;
}

Matrix::Matrix(const Matrix &matrixCopy)
{
    //br++;
    //cout << br << endl;
    n = matrixCopy.n;
    m = matrixCopy.m;
    a = new double* [n];
    for(int i = 0; i < n; i++)
    {
        a[i] = new double[m];
        for(int j = 0; j < m; j++)
        {
            a[i][j] = matrixCopy.a[i][j];
        }
    }
}

void Matrix::randomize()
{
    for(int i = 0; i < n; i++)
        for(int j = 0; j < m; j++){
            a[i][j] = rng.rand_01();
            if(rng.next()&1) a[i][j] = -a[i][j];
        }
}

void Matrix::operator = (const Matrix &matrixCopy)
{
    if(this == &matrixCopy)
    {
        return;
    }
    for(int i = 0; i < n; i++)
        delete[] a[i];
    delete[] a;
    n = matrixCopy.n;
    m = matrixCopy.m;
    a = new double* [n];
    for(int i = 0; i < n; i++)
    {
        a[i] = new double[m];
        for(int j = 0; j < m; j++)
        {
            a[i][j] = matrixCopy.a[i][j];
        }
    }
}

void Matrix::operator += (const Matrix &matrixCopy)
{
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < m; j++)
        {
            a[i][j] += matrixCopy.a[i][j];
        }
    }
}

void Matrix::operator -= (const Matrix &matrixCopy)
{
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < m; j++)
        {
            a[i][j] -= matrixCopy.a[i][j];
        }
    }
}

void Matrix::operator *= (const double c)
{
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < m; j++)
        {
            a[i][j] *= c;
        }
    }
}

Matrix Matrix::operator + (const Matrix &toAdd) const
{
    assert(n == toAdd.n);
    assert(m == toAdd.m);
    Matrix sum = *this;
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < m; j++)
        {
            sum.a[i][j] += toAdd.a[i][j];
        }
    }
    return sum;
}


Matrix Matrix::operator - (const Matrix &toSub) const
{
    assert(n == toSub.n);
    assert(m == toSub   .m);
    Matrix sum = *this;
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < m; j++)
        {
            sum.a[i][j] -= toSub.a[i][j];
        }
    }
    return sum;
}

Matrix Matrix::operator * (const Matrix &toMulti) const
{
    assert(m == toMulti.n);
    Matrix mult(n, toMulti.m);
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < toMulti.m; j++)
        {
            for(int k = 0; k < m; k++)
            {
                mult.a[i][j] += a[i][k] * toMulti.a[k][j];
            }
        }
    }
    return mult;
}

Matrix Matrix::operator * (const double num) const
{
    Matrix mult = *this;
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < m; j++)
        {
            mult.a[i][j] *= num;
        }
    }
    return mult;
}

///Hadamard product
Matrix Matrix::operator % (const Matrix &toMult) const
{
    assert(n == toMult.n);
    assert(m == toMult.m);
    Matrix mult(n,m);
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < m; j++)
        {
            mult.a[i][j] = a[i][j] * toMult.a[i][j];
        }
    }
    return mult;
}

Matrix Matrix::transpose() const
{
    Matrix tr(m,n);
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < m; j++)
        {
            tr.a[j][i] = a[i][j];
        }
    }
    return tr;
}

Matrix Matrix::print()
{
    cout << n << " " << m << endl;
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < m; j++)
        {
            cout << a[i][j] << " ";
        }
        cout << endl;
    }
}

Matrix Matrix::sigmoid() const
{
    Matrix sig(n,m);
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < m; j++)
        {
            sig.a[i][j] = sigmoid_double(a[i][j]);
        }
    }
    return sig;
}

Matrix Matrix::sigmoid_derivative() const
{
    Matrix sig(n,m);
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < m; j++)
        {
            sig.a[i][j] = sigmoid_der_double(a[i][j]);
        }
    }
    return sig;
}

void Matrix::zero()
{
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < m; j++)
        {
            a[i][j] = 0.0;
        }
    }
}

double* Matrix::operator[] (int x)
{
    assert(x>=0 && x < n);
    return a[x];
}

double sigmoid_double(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_der_double(double x)
{
    return sigmoid_double(x) * (1.0 - sigmoid_double(x));
}
