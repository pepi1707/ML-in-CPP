#ifndef CROSSENTROPYCOSTFUNCTION_H_INCLUDED
#define CROSSENTROPYCOSTFUNCTION_H_INCLUDED

#include "CostFunction.h"

using namespace std;

class CrossentropyCostFunction : public CostFunction
{

public:

    CrossentropyCostFunction() : CostFunction()
    {

    }

    virtual double calculate(const vector<double>& outputLayer, const vector<double>& desiredOutput)
    {
        assert(outputLayer.size() == desiredOutput.size());
        const double eps = 1e-9;

        double sum = 0;
        for(int i = 0; i < outputLayer.size(); i++)
        {
            if(fabs(outputLayer[i]) < eps  || fabs(outputLayer[i] - 1.0) < eps)
            {
                return 0;
            }

            sum += - log(outputLayer[i]) * desiredOutput[i] - log(1.0 - outputLayer[i]) * (1.0 - desiredOutput[i]);
        }

        return sum;
    }

    virtual void derivative(vector<double>& derivative, const vector<double>& outputLayer, const vector<double>& desiredOutput, const vector<double>& z, const string& activation)
    {
        assert(outputLayer.size() == desiredOutput.size() && outputLayer.size() == z.size());
        if(activation == "sigmoid" || activation == "softmax")
        {
            for(int i = 0; i < derivative.size(); i++)
            {
                derivative[i] = outputLayer[i] - desiredOutput[i];
            }
        }
    }

    virtual ~CrossentropyCostFunction()
    {

    }
};

#endif // CROSSENTROPYCOSTFUNCTION_H_INCLUDED
