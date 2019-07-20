#ifndef CROSSENTROPYCOSTFUNCTION_H_INCLUDED
#define CROSSENTROPYCOSTFUNCTION_H_INCLUDED

#include "CostFunction.h"

using namespace std;

class CrossentropyCostFunction : public CostFunction
{

public:

    CrossentropyCostFunction();

    virtual Matrix calculate(const Matrix& outputLayer, const Matrix& desiredOutput);

    virtual Matrix derivative(const Matrix& outputLayer, const Matrix& desiredOutput, const Matrix& z);

    virtual ~CrossentropyCostFunction();
};

#endif // CROSSENTROPYCOSTFUNCTION_H_INCLUDED
