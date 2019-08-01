#ifndef CONVLAYER_H_INCLUDED
#define CONVLAYER_H_INCLUDED


#include "Layer.h"

class ConvLayer : public Layer
{

public:

    string activation;

    int layerSize;
    int n_in_d, n_in_h, n_in_w;
    int kernel;

    vector<vector<double>> weight;
    vector<double> bias;
    vector<double> z;
    vector<double> act;
    vector<double> error;
    vector<double> delta_b;
    vector<vector<double>> delta_w;

    ConvLayer(int _n_in_d, int _n_in_h, int _n_in_w, int _layerSize, int _kernel, string _activation) : n_in_d(_n_in_d), n_in_h(_n_in_h), n_in_w(_n_in_w), layerSize(_layerSize), kernel(_kernel), activation(_activation)
    {
        bias.resize(layerSize);
        delta_b.resize(layerSize);
        weight.resize(layerSize);
        delta_w.resize(layerSize);
        for(int i = 0; i < layerSize; i++)
        {
            weight[i].resize(kernel * kernel * n_in_d);
            delta_w[i].resize(kernel * kernel * n_in_d);
        }

        z.resize((n_in_w - kernel + 1) * (n_in_h - kernel + 1) * layerSize );
        act.resize((n_in_w - kernel + 1) * (n_in_h - kernel + 1) * layerSize );
        error.resize((n_in_w - kernel + 1) * (n_in_h - kernel + 1) * layerSize );

        ///random initialization of weights and biases
        unsigned seed = chrono::system_clock::now().time_since_epoch().count();
        default_random_engine generator (seed);


        normal_distribution<double> distribution (0.0, sqrt(1.0 / layerSize));
        fill(delta_b.begin(), delta_b.end(), 0);
        for(int i = 0; i < layerSize; i++)
        {
            bias[i] = distribution(generator);
            for(int j = 0; j < weight[i].size(); j++)
            {
                weight[i][j] = distribution(generator);
            }
            fill(delta_w[i].begin(), delta_w[i].end(), 0);
        }



    }

    virtual string type()
    {
        return "conv";
    }

    virtual void init()
    {

    }

    virtual void feed_input(const vector<double>& actLast)
    {
        int idx;
        if(activation == "sigmoid")
        {
            for(int fil = 0; fil < layerSize; fil++)
            {
                for(int h = 0; h < n_in_h - kernel + 1; h ++)
                {
                    for(int w = 0; w < n_in_w - kernel + 1; w ++)
                    {
                        idx = fil * (n_in_h - kernel + 1) * (n_in_w - kernel + 1) + h * (n_in_w - kernel + 1) + w;
                        z[idx] = 0;
                        for(int k = 0; k < n_in_d; k++)
                        {
                            for(int i = 0; i < kernel; i++)
                            {
                                for(int j = 0; j < kernel; j++)
                                {
                                    z[idx] += actLast[k * n_in_h * n_in_w + (i + h) * n_in_w + (j + w)] * weight[fil][k * kernel * kernel + i * kernel + j];
                                }
                            }
                        }
                        z[idx] += bias[fil];

                        act[idx] = sigmoid_double(z[idx]);
                    }
                }
            }
        }
        else if(activation == "relu")
        {
            for(int fil = 0; fil < layerSize; fil++)
            {
                for(int h = 0; h < n_in_h - kernel + 1; h ++)
                {
                    for(int w = 0; w < n_in_w - kernel + 1; w ++)
                    {
                        idx = fil * (n_in_h - kernel + 1) * (n_in_w - kernel + 1) + h * (n_in_w - kernel + 1) + w;
                        z[idx] = 0;
                        for(int k = 0; k < n_in_d; k++)
                        {
                            for(int i = 0; i < kernel; i++)
                            {
                                for(int j = 0; j < kernel; j++)
                                {
                                    z[idx] += actLast[k * n_in_h * n_in_w + (i + h) * n_in_w + (j + w)] * weight[fil][k * kernel * kernel + i * kernel + j];
                                }
                            }
                        }
                        z[idx] += bias[fil];

                        act[idx] = max(0.0, z[idx]);
                    }
                }
            }
        }
    }

    virtual void backprop(const vector<double>& actPrev, const vector<vector<double> >& weightLast, const vector<double>& errorLast, const string& typeLast)
    {
        ///calculate error
        if(typeLast == "pool")
        {
            int idx;
            for(int k = 0; k < layerSize; k++)
            {
                for(int i = 0; i < n_in_h - kernel + 1; i++)
                {
                    for(int j = 0; j < n_in_w - kernel + 1; j++)
                    {
                        idx = k * (n_in_w - kernel + 1) * (n_in_h - kernel + 1) + i * (n_in_w - kernel + 1) + j;
                        error[idx] = weightLast[0][idx];
                    }
                }
            }
        }
        else if(typeLast == "dense")
        {
            int idx;
            for(int k = 0; k < layerSize; k++)
            {
                for(int i = 0; i < n_in_h - kernel + 1; i++)
                {
                    for(int j = 0; j < n_in_w - kernel + 1; j++)
                    {
                        idx = k * (n_in_w - kernel + 1) * (n_in_h - kernel + 1) + i * (n_in_w - kernel + 1) + j;
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
            for(int k = 0; k < layerSize; k++)
            {
                for(int i = 0; i < n_in_h - kernel + 1; i++)
                {
                    for(int j = 0; j < n_in_w - kernel + 1; j++)
                    {
                        idx = k * (n_in_w - kernel + 1) * (n_in_h - kernel + 1) + i * (n_in_w - kernel + 1) + j;
                        error[idx] = 0;
                        int ker_sz = sqrt(weightLast[0].size() / n_in_d);
                        for(int fil = 0; fil < weightLast.size(); fil++)
                        {
                            for(int h = 0; h < ker_sz; h++)
                            {
                                for(int w = 0; w < ker_sz; w++)
                                {
                                    if(i + ker_sz - h - 1 >= n_in_h - kernel + 1 ||
                                       j + ker_sz - w - 1 >= n_in_w - kernel + 1)
                                        continue;
                                    if(i - h < 0 || j - w < 0)
                                        break;
                                    error[idx] += weightLast[fil][k * ker_sz * ker_sz + h * ker_sz + w] * errorLast[fil * (errorLast.size() / weightLast.size()) + (i - h) * (n_in_w - kernel + 2 - ker_sz) + (j - w)];
                                }
                            }
                        }
                    }
                }
            }
        }

        ///activation derivative
        if(activation == "sigmoid")
        {
            int idx;
            for(int k = 0; k < layerSize; k++)
            {
                for(int i = 0; i < n_in_h - kernel + 1; i++)
                {
                    for(int j = 0; j < n_in_w - kernel + 1; j++)
                    {
                        idx = k * (n_in_w - kernel + 1) * (n_in_h - kernel + 1) + i * (n_in_w - kernel + 1) + j;
                        error[idx] *= sigmoid_der_double(z[idx]);
                    }
                }
            }
        }
        else if(activation == "relu")
        {
            int idx;
            for(int k = 0; k < layerSize; k++)
            {
                for(int i = 0; i < n_in_h - kernel + 1; i++)
                {
                    for(int j = 0; j < n_in_w - kernel + 1; j++)
                    {
                        idx = k * (n_in_w - kernel + 1) * (n_in_h - kernel + 1) + i * (n_in_w - kernel + 1) + j;
                        if(z[idx] <= 0)
                            error[idx] = 0.0;
                    }
                }
            }
        }

        int errorIdx;
        for(int fil = 0; fil < layerSize; fil++)
        {
            for(int i = 0; i < n_in_h - kernel + 1; i++)
            {
                for(int j = 0; j < n_in_w - kernel + 1; j++)
                {
                    errorIdx = fil * (n_in_h - kernel + 1) * (n_in_w - kernel + 1) + i * (n_in_w - kernel + 1) + j;
                    delta_b[fil] += error[errorIdx];
                    for(int k = 0; k < n_in_d; k++)
                    {
                        for(int h = 0; h < kernel; h++)
                        {
                            for(int w = 0; w < kernel; w++)
                            {
                                delta_w[fil][k * kernel * kernel + h * kernel + w] += error[errorIdx] * actPrev[k * n_in_h * n_in_w + (i + h) * n_in_w + (j + w)];
                            }
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
        return z;
    }

    virtual string& getActivation()
    {
        return activation;
    }

    virtual vector<double>& getError()
    {
        return error;
    }

    virtual vector<double>& getBias()
    {
        return bias;
    }

    virtual void correctVars(const double& eta, const int& batch_sz, const double& weight_lambda, const int& input_sz)
    {
        for(int i = 0; i < layerSize; i++)
        {
            bias[i] -= delta_b[i] * eta / batch_sz;
            delta_b[i] = 0;

            for(int j = 0; j < weight[i].size(); j++)
            {
                weight[i][j] -= eta * weight[i][j] * weight_lambda / input_sz;
                weight[i][j] -= delta_w[i][j] * eta / batch_sz;
                delta_w[i][j] = 0;
            }

        }
    }

    virtual void load(ifstream& in)
    {
        for(int i = 0; i < layerSize; i++)
        {
            in >> bias[i];
        }
        for(int i = 0; i < layerSize; i++)
        {
            for(int j = 0; j < weight[i].size(); j++)
            {
                in >> weight[i][j];
            }
        }
    }

    virtual void save(ofstream& out)
    {
        for(int j = 0; j < bias.size(); j++)
        {
            out << bias[j] << " ";
        }
        out << endl;
        for(int j = 0; j < weight.size(); j++)
        {
            for(int l = 0; l < weight[j].size(); l++)
            {
                out << weight[j][l] << " ";
            }
            out << endl;
        }
    }
};


#endif // CONVLAYER_H_INCLUDED
