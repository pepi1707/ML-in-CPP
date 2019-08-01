#ifndef LAYER_H_INCLUDED
#define LAYER_H_INCLUDED

#include <vector>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <random>

class Layer
{

public:

    Layer()
    {

    }

    virtual ~Layer()
    {

    }

    virtual string type() = 0;

    virtual void feed_input(const vector<double>& actLast) = 0;

    virtual void init()
    {

    }

    virtual void setAct(const vector<double>&)
    {

    }

    virtual void backprop(const vector<double>&, const vector<vector<double>>&, const vector<double>&, const string&) = 0;

    virtual void backpropLastLayer(const vector<double>&)
    {

    }

    virtual void noDropout()
    {

    }

    virtual vector<vector<double>>& getWeight() = 0;
    virtual vector<double>& getAct() = 0;
    virtual vector<double>& getZ() = 0;
    virtual string& getActivation() = 0;
    virtual vector<double>& getError() = 0;
    virtual vector<double>& getBias() = 0;

    virtual void correctVars(const double&, const int&, const double&, const int&) = 0;

    virtual void load(ifstream&)
    {

    }

    virtual void save(ofstream&)
    {

    }
};


#endif // LAYER_H_INCLUDED
