#ifndef POOLLAYER_H_INCLUDED
#define POOLLAYER_H_INCLUDED

#include "Layer.h"

class PoolLayer : public Layer
{

public:

    string poolType;

    int n_in_d, n_in_w, n_in_h;

    vector< vector<double> > weight;
    vector<double> act;
    vector<double> error;

    int pool;

    PoolLayer(int _n_in_d, int _n_in_w, int _n_in_h, int _pool, string _poolType) : Layer(), n_in_d(_n_in_d), n_in_w(_n_in_w), n_in_h(_n_in_h), pool(_pool), poolType(_poolType)
    {
        weight.resize(1);
        weight[0].resize(n_in_d * n_in_h * n_in_w);
        act.resize(n_in_d * (n_in_w / pool) * (n_in_h / pool));
        error.resize(n_in_d * (n_in_w / pool) * (n_in_h / pool));
    }

    virtual string type()
    {
        return "pool";
    }

    virtual void feed_input(const vector<double>& actLast)
    {
        for(int k = 0; k < n_in_d; k++)
        {
            for(int i = 0; i < n_in_h; i+=pool)
            {
                for(int j = 0; j < n_in_w; j+=pool)
                {
                    int idx = k * n_in_h * n_in_w + i * n_in_w + j;
                    for(int h = 0; h < pool; h++)
                    {
                        for(int w = 0; w < pool; w++)
                        {
                            int cand = k * n_in_h * n_in_w + (i + h) * n_in_w + (j + w);
                            if(actLast[cand] > actLast[idx])
                            {
                                idx = cand;
                            }
                            weight[0][cand] = 0.0;
                        }
                    }
                    weight[0][idx] = 1.0;
                    act[k * (n_in_w / pool) * (n_in_h / pool) + (i/pool) * (n_in_w/pool) + (j/pool)] = actLast[idx];
                }
            }
        }
    }

    virtual void init()
    {

    }

    virtual void backprop(const vector<double>& actPrev, const vector<vector<double> >& weightLast, const vector<double>& errorLast, const string& typeLast)
    {
        if(typeLast == "pool")
        {
            int idx;
            for(int k = 0; k < n_in_d; k++)
            {
                for(int i = 0; i < n_in_h / pool; i++)
                {
                    for(int j = 0; j < n_in_w / pool; j++)
                    {
                        idx = k * (n_in_w / pool) * (n_in_h / pool) + i * (n_in_w / pool) + j;
                        error[idx] = weightLast[0][idx];
                    }
                }
            }
        }
        else if(typeLast == "dense")
        {
            int idx;
            for(int k = 0; k < n_in_d; k++)
            {
                for(int i = 0; i < n_in_h / pool; i++)
                {
                    for(int j = 0; j < n_in_w / pool; j++)
                    {
                        idx = k * (n_in_w / pool) * (n_in_h / pool) + i * (n_in_w / pool) + j;
                        error[idx] = 0;
                        for(int err = 0; err < errorLast.size(); err++)
                        {
                            error[idx] += errorLast[err] * weightLast[err][idx];
                        }
                    }
                }
            }
        }
        else if(typeLast == "conv")
        {
            int idx;
            for(int k = 0; k < n_in_d; k++)
            {
                for(int i = 0; i < n_in_h / pool; i++)
                {
                    for(int j = 0; j < n_in_w / pool; j++)
                    {
                        idx = k * (n_in_w / pool) * (n_in_h / pool) + i * (n_in_w / pool) + j;
                        error[idx] = 0;
                        int ker_sz = sqrt(weightLast[0].size() / n_in_d);
                        for(int fil = 0; fil < weightLast.size(); fil++)
                        {
                            for(int h = 0; h < ker_sz; h++)
                            {
                                for(int w = 0; w < ker_sz; w++)
                                {
                                    if(i + ker_sz - h - 1 >= n_in_h / pool ||
                                       j + ker_sz - w - 1 >= n_in_w / pool)
                                        continue;
                                    if(i - h < 0 || j - w < 0)
                                        break;
                                    error[idx] += weightLast[fil][k * ker_sz * ker_sz + h * ker_sz + w] * errorLast[fil * (errorLast.size() / weightLast.size()) + (i - h) * (n_in_w / pool + 1 - ker_sz) + (j - w)];
                                }
                            }
                        }
                    }
                }
            }
        }

        for(int k = 0; k < n_in_d; k++)
        {
            for(int i = 0; i < n_in_h; i+=pool)
            {
                for(int j = 0; j < n_in_w; j+=pool)
                {
                    int idx = k * (n_in_w / pool) * (n_in_h / pool) + (i / pool) * (n_in_w / pool) + (j / pool);
                    for(int h = 0; h < pool; h++)
                    {
                        for(int w = 0; w < pool; w++)
                        {
                            int cand = k * n_in_h * n_in_w + (i + h) * n_in_w + (j + w);
                            weight[0][cand] *= error[idx];
                        }
                    }
                }
            }
        }
    }

    virtual vector<vector<double> >& getWeight()
    {
        return weight;
    }

    virtual vector<double>& getAct()
    {
        return act;
    }

    virtual vector<double>& getZ()
    {

    }

    virtual string& getActivation()
    {

    }

    virtual vector<double>& getError()
    {
        return error;
    }

    virtual vector<double>& getBias()
    {

    }


    virtual void correctVars(const double& eta, const int& batch_sz, const double& weight_lambda, const int& input_sz)
    {

    }
};

#endif // POOLLAYER_H_INCLUDED
