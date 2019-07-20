#ifndef QUADRATICCOSTFUNCTION_H_INCLUDED
#define QUADRATICCOSTFUNCTION_H_INCLUDED

#include "CostFunction.h"

using namespace std;

class QuadraticCostFunction : public CostFunction
{

public:

    QuadraticCostFunction();

    virtual Matrix calculate(const Matrix& outputLayer, const Matrix& desiredOutput);

    virtual Matrix derivative(const Matrix& outputLayer, const Matrix& desiredOutput, const Matrix& z);

    virtual ~QuadraticCostFunction();
};

#endif // QUADRATICCOSTFUNCTION_H_INCLUDED
