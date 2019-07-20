#ifndef MATRIX_H_INCLUDED
#define MATRIX_H_INCLUDED

#include <iostream>
#include <cmath>
#include <cassert>
#include "Xorshift.h"

using namespace std;

class Matrix
{

public:

    Xorshift rng;

    double **a;

    int n,m;

    Matrix();

    Matrix(int _n, int _m);

    ~Matrix();

    Matrix(const Matrix &matrixCopy);

    void randomize();

    void operator = (const Matrix &matrixCopy);

    void operator += (const Matrix &matrixCopy);

    void operator -= (const Matrix &matrixCopy);

    void operator *= (const double c);

    Matrix operator + (const Matrix &toAdd) const;

    Matrix operator - (const Matrix &toSub) const;

    Matrix operator * (const Matrix &toMulti) const;

    Matrix operator * (const double num) const;

    ///Hadamard product
    Matrix operator % (const Matrix &toMult) const;

    Matrix transpose() const;

    Matrix print();

    double* operator[] (int);

    Matrix sigmoid() const;

    Matrix sigmoid_derivative() const;

    void zero();
};

double sigmoid_double(double x);

double sigmoid_der_double(double x);


#endif // MATRIX_H_INCLUDED
