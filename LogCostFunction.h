#ifndef LOGCOSTFUNCTION_H_INCLUDED
#define LOGCOSTFUNCTION_H_INCLUDED

#include "CostFunction.h"

using namespace std;

class LogCostFunction : public CostFunction
{

public:

    LogCostFunction() : CostFunction()
    {

    }

    virtual double calculate(const vector<double>& outputLayer, const vector<double>& desiredOutput)
    {
        assert(outputLayer.size() == desiredOutput.size());
        const double eps = 1e-9;

        double sum = 0;
        for(int i = 0; i < outputLayer.size(); i++)
        {
            if(fabs(desiredOutput[i]) > eps)
            {
                return -log(outputLayer[i]);
            }
        }
    }

    virtual void derivative(vector<double>& derivative, const vector<double>& outputLayer, const vector<double>& desiredOutput, const vector<double>& z, const string& activation)
    {
        assert(outputLayer.size() == desiredOutput.size() && outputLayer.size() == z.size());
        if(activation == "softmax")
        {
            for(int i = 0; i < derivative.size(); i++)
            {
                derivative[i] = outputLayer[i] - desiredOutput[i];
            }
        }
    }

    virtual ~LogCostFunction()
    {

    }
};


#endif // LOGCOSTFUNCTION_H_INCLUDED
