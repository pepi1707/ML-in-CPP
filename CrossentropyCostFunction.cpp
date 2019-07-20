#include "CrossentropyCostFunction.h"


CrossentropyCostFunction::CrossentropyCostFunction() : CostFunction()
{

}

Matrix CrossentropyCostFunction::calculate(const Matrix& outputLayer, const Matrix& desiredOutput)
{
    const double eps = 1e-9;

    double sum = 0;
    Matrix ret(1, outputLayer.m);
    for(int i = 0; i < outputLayer.m; i++)
    {
        if(fabs(outputLayer.a[0][i]) < eps  || fabs(outputLayer.a[0][i] - 1.0) < eps)
        {
            for(int j = 0; j < outputLayer.m; j++)
            {
                ret.a[0][j] = -1e9;
            }
            return ret;
        }

        ret.a[0][i] = - log(outputLayer.a[0][i]) * desiredOutput.a[0][i] - log(1.0 - outputLayer.a[0][i]) * (1.0 - desiredOutput.a[0][i]);
    }

    return ret;
}

Matrix CrossentropyCostFunction::derivative(const Matrix& outputLayer, const Matrix& desiredOutput, const Matrix& z)
{
    return (outputLayer - desiredOutput);
}

CrossentropyCostFunction::~CrossentropyCostFunction()
{

}
