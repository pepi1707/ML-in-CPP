#ifndef COSTFUNCTION_H_INCLUDED
#define COSTFUNCTION_H_INCLUDED


#include "Matrix.h"

using namespace std;

class CostFunction
{
public:

    virtual Matrix calculate(const Matrix& outputLayer, const Matrix& desiredOutput) = 0;

    virtual Matrix derivative(const Matrix& outputLayer, const Matrix& desiredOutput, const Matrix& z) = 0;

    virtual ~CostFunction() {};
};



#endif // COSTFUNCTION_H_INCLUDED
