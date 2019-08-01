#ifndef COSTFUNCTION_H_INCLUDED
#define COSTFUNCTION_H_INCLUDED


#include <vector>
#include "Matrix.h"

using namespace std;

class CostFunction
{
public:

    virtual double calculate(const vector<double>& outputLayer, const vector<double>& desiredOutput) = 0;

    virtual void derivative(vector<double>& derivative, const vector<double>& outputLayer, const vector<double>& desiredOutput, const vector<double>& z, const string&) = 0;

    virtual ~CostFunction() {};
};



#endif // COSTFUNCTION_H_INCLUDED
