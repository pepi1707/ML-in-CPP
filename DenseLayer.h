#ifndef DENSELAYER_H_INCLUDED
#define DENSELAYER_H_INCLUDED

#include <cstring>
#include "Layer.h"

class DenseLayer : public Layer
{

public:

    string activation;

    int layerSize;
    int n_in;

    double dropout;

    bool *dropped;

    vector<vector<double>> weight;
    vector<double> bias;
    vector<double> z;
    vector<double> act;
    vector<double> error;
    vector<double> delta_b;
    vector<vector<double>> delta_w;

    DenseLayer(int _n_in, int _layerSize, double _dropout = 0.0, string _activation = "sigmoid") : Layer(), n_in(_n_in), layerSize(_layerSize), activation(_activation)
    {
        dropout = _dropout;
        dropped = new bool[layerSize];

        for(int i = 0; i < layerSize; i++)
            dropped[i] = 0;

        bias.resize(layerSize);
        z.resize(layerSize);
        act.resize(layerSize);
        error.resize(layerSize);
        delta_b.resize(layerSize);

        weight.resize(layerSize);
        delta_w.resize(layerSize);
        for(int i = 0; i < layerSize; i++)
        {
            weight[i].resize(n_in);
            delta_w[i].resize(n_in);
        }

        ///random initialization of weights and biases
        unsigned seed = chrono::system_clock::now().time_since_epoch().count();
        default_random_engine generator (seed);

        normal_distribution<double> distribution (0.0, sqrt(1.0 / layerSize));
        for(int i = 0; i < layerSize; i++)
        {
            bias[i] = distribution(generator);
            for(int j = 0; j < weight[i].size(); j++)
            {
                weight[i][j] = distribution(generator);
            }
            fill(delta_w[i].begin(), delta_w[i].end(), 0);
        }

        fill(delta_b.begin(), delta_b.end(), 0);

    }

    DenseLayer(const DenseLayer& toCopy)
    {
        layerSize = toCopy.layerSize;
        n_in = toCopy.n_in;
        dropout = toCopy.dropout;

        activation = toCopy.activation;

        dropped = new bool[layerSize];
        for(int i = 0; i < layerSize; i++)
            dropped[i] = toCopy.dropped[i];

        weight = toCopy.weight;
        bias = toCopy.bias;
        z = toCopy.z;
        act = toCopy.act;
        error = toCopy.error;
        delta_b = toCopy.delta_b;
        delta_w = toCopy.delta_w;
    }

    virtual ~DenseLayer()
    {
        delete[] dropped;
    }

    virtual string type()
    {
        return "dense";
    }

    virtual void init()
    {

        ///initialize dropout neurons
        if(dropout < 1e-6)
            return;
        vector<int> help;
        help.resize(layerSize);

        for(int i = 0; i < layerSize; i++)
            dropped[i] = 0;

        for(int i = 0; i < layerSize; i++)
            help[i] = i;

        random_shuffle(help.begin(), help.end());

        int br = 0;
        while(1.0 * br + 1e-9 < layerSize * dropout)
        {
            dropped[help[br]] = true;
            br++;
        }
    }

    virtual void noDropout()
    {
        for(int i = 0; i < layerSize; i++)
            dropped[i] = 0;
    }

    virtual void feed_input(const vector<double>& actLast)
    {
        if(activation == "sigmoid"){
            for(int i = 0; i < layerSize; i++)
            {
                z[i] = 0;
                act[i] = 0;

                if(dropped[i])
                {
                    continue;
                }

                for(int j = 0; j < actLast.size(); j++)
                {
                    z[i] += weight[i][j] * actLast[j];
                }

                z[i] += bias[i];

                act[i] = sigmoid_double(z[i]) / (1 - dropout);
            }
        }
        else if(activation == "relu")
        {
            for(int i = 0; i < layerSize; i++)
            {
                z[i] = 0;
                act[i] = 0;
                if(dropped[i])
                    continue;
                for(int j = 0; j < actLast.size(); j++)
                {
                    z[i] += weight[i][j] * actLast[j];
                }
                z[i] += bias[i];
                act[i] = max(0.0, z[i]) / (1 - dropout);
            }
        }
        else if(activation == "softmax")
        {
            double sum = 0;
            double max_z = 0;
            for(int i = 0; i < layerSize; i++)
            {
                z[i] = 0;
                act[i] = 0;
                for(int j = 0; j < actLast.size(); j++)
                {
                    z[i] += weight[i][j] * actLast[j];
                }
                z[i] += bias[i];
                if(z[i] > max_z)
                    max_z = z[i];
            }
            for(int i = 0; i < layerSize; i++)
            {
                sum += exp(z[i] - max_z);
            }
            sum = max_z + log(sum);
            for(int i = 0; i < layerSize; i++)
            {
                act[i] = exp(z[i] - sum);
            }
        }
    }

    virtual void backprop(const vector<double>& actPrev, const vector<vector<double>>& weightLast, const vector<double>& errorLast, const string& typeLast)
    {
        if(activation == "sigmoid"){
            for(int i = 0; i < layerSize; i++)
            {
                 error[i] = 0;

                 if(dropped[i])
                 {
                     continue;
                 }

                 for(int j = 0; j < errorLast.size(); j++)
                 {
                     error[i] += errorLast[j] * weightLast[j][i];
                 }
                 error[i] *= sigmoid_der_double(z[i]) / (1 - dropout);

                 delta_b[i] += error[i];
                 for(int j = 0; j < actPrev.size(); j++)
                 {
                     delta_w[i][j] += error[i] * actPrev[j];
                 }
            }
        }
        else if(activation == "relu")
        {
            for(int i = 0; i < layerSize; i++)
            {
                error[i] = 0;
                if(dropped[i])
                    continue;

                for(int j = 0; j < errorLast.size(); j++)
                {
                    error[i] += errorLast[j] * weightLast[j][i];
                }

                if(z[i] <= 0){
                    error[i] = 0.0;
                    continue;
                }
                error[i] /= (1 - dropout);

                delta_b[i] += error[i];
                for(int j = 0; j < actPrev.size(); j++)
                {
                    delta_w[i][j] += error[i] * actPrev[j];
                }
            }

        }
    }

    virtual void setAct(const vector<double>& toCopy)
    {
        for(int i = 0; i < act.size(); i++)
            act[i] = toCopy[i];
    }

    virtual void backpropLastLayer(const vector<double>& actPrev)
    {
        for(int i = 0; i < layerSize; i++)
        {
             delta_b[i] += error[i];
             for(int j = 0; j < actPrev.size(); j++)
             {
                 delta_w[i][j] += error[i] * actPrev[j];
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
        //cout << "in correctVars" << endl;
        for(int i = 0; i < layerSize; i++)
        {
            bias[i] -= delta_b[i] * eta / batch_sz;
            delta_b[i] = 0;
            for(int j = 0; j < weight[i].size(); j++)
            {
                weight[i][j] -= weight_lambda * eta * weight[i][j] / input_sz;
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
        for(int i = 0; i < weight.size(); i++)
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

#endif // DENSELAYER_H_INCLUDED
