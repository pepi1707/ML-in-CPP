#ifndef QUADRATICCOSTFUNCTION_H_INCLUDED
#define QUADRATICCOSTFUNCTION_H_INCLUDED

#include "CostFunction.h"

using namespace std;

class QuadraticCostFunction : public CostFunction
{

public:

    QuadraticCostFunction() : CostFunction()
    {

    }

    virtual double calculate(const vector<double>& outputLayer, const vector<double>& desiredOutput)
    {
        assert(outputLayer.size() == desiredOutput.size());
        double sum = 0;
        for(int i = 0; i < outputLayer.size(); i++)
        {
            sum +=  (outputLayer[i] - desiredOutput[i]) * (outputLayer[i] - desiredOutput[i]) * 0.5;
        }
        return sum;
    }

    virtual void derivative(vector<double>& derivative, const vector<double>& outputLayer, const vector<double>& desiredOutput, const vector<double>& z, const string& activation)
    {
        assert(outputLayer.size() == desiredOutput.size() && outputLayer.size() == z.size());
        if(activation == "sigmoid")
        {
            for(int i = 0; i < outputLayer.size(); i++)
            {
                derivative[i] = (outputLayer[i] - desiredOutput[i]) * sigmoid_der_double(z[i]);
            }
        }
    }

    virtual ~QuadraticCostFunction()
    {

    }

};

#endif // QUADRATICCOSTFUNCTION_H_INCLUDED
