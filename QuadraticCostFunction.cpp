#include "QuadraticCostFunction.h"

QuadraticCostFunction::QuadraticCostFunction() : CostFunction()
{

}

Matrix QuadraticCostFunction::calculate(const Matrix& outputLayer, const Matrix& desiredOutput)
{
    return (outputLayer - desiredOutput) * (outputLayer - desiredOutput) * 0.5;
}

Matrix QuadraticCostFunction::derivative(const Matrix& outputLayer, const Matrix& desiredOutput, const Matrix& z)
{
    return (outputLayer - desiredOutput) * z.sigmoid_derivative();
}

QuadraticCostFunction::~QuadraticCostFunction()
{

}
