#ifndef NEURALNETWORK_H_INCLUDED
#define NEURALNETWORK_H_INCLUDED

#include "QuadraticCostFunction.h"
#include "CrossentropyCostFunction.h"
#include "LogCostFunction.h"
#include "DenseLayer.h"
#include "ConvLayer.h"
#include "PoolLayer.h"
#include <cstring>
#include <vector>
#include <algorithm>
#include <ctime>
#include <random>

using namespace std;

class NeuralNetwork
{

public:

    int numLayers;
    double eta;
    double weight_lambda;

    vector<int> layerSize;
    vector<Layer*> layers;

    CostFunction* costFunction;

    NeuralNetwork(double _eta = 0.1, double _lambda = 0, string costFunctionName = "crossentropy") : eta(_eta), weight_lambda(_lambda), numLayers(0)
    {
        if(costFunctionName == "quadratic")
        {
            costFunction = new QuadraticCostFunction();
        }

        if(costFunctionName == "crossentropy")
        {
            costFunction = new CrossentropyCostFunction();
        }

        if(costFunctionName == "log-likehood")
        {
            costFunction = new LogCostFunction();
        }

    }

    ~NeuralNetwork()
    {
        delete costFunction;
        for(int i = 0; i < layers.size(); i++)
        {
            delete layers[i];
        }
    }

    void addLayer(const DenseLayer& toAdd)
    {
        numLayers ++;
        Layer* add = new DenseLayer(toAdd);

        layers.push_back(add);
    }

    void addLayer(const ConvLayer& toAdd)
    {
        numLayers ++;
        Layer* add = new ConvLayer(toAdd);

        layers.push_back(add);
    }

    void addLayer(const PoolLayer& toAdd)
    {
        numLayers ++;
        Layer* add = new PoolLayer(toAdd);

        layers.push_back(add);
    }

    void feed_input(const vector<double>& input)
    {
        layers[0]->setAct(input);
        for(int i = 1; i < numLayers; i++)
        {
            layers[i]->feed_input(layers[i-1] -> getAct());
        }
    }

    void backprop(const vector<double>& output)
    {
        costFunction -> derivative(layers[numLayers - 1] -> getError(), layers[numLayers - 1] -> getAct(), output, layers[numLayers - 1] -> getZ(), layers[numLayers - 1] -> getActivation());
        layers[numLayers - 1] -> backpropLastLayer(layers[numLayers - 2] -> getAct());
        for(int i = numLayers - 2; i > 0; i--)
        {
            layers[i] -> backprop(layers[i - 1] -> getAct(), layers[i + 1] -> getWeight(), layers[i + 1] -> getError(), layers[i + 1] -> type());
        }
    }

    void handle_minibatch(const vector<pair< vector<double>, vector<double> > >& train_data, const int& batch_idx, const int& batch_sz)
    {
        for(int i = 0; i < numLayers; i++)
        {
            layers[i] -> init();
        }
        //cerr << "about to feed and backprop" << endl;
        for(int i = 0; i < batch_sz; i++)
        {
            feed_input(train_data[batch_idx * batch_sz + i].first);
            backprop(train_data[batch_idx * batch_sz + i].second);
        }
        //cerr << "about to correctVars" << endl;
        for(int i = 1; i < numLayers; i++)
        {
            layers[i] -> correctVars(eta, batch_sz, weight_lambda, train_data.size());
        }
    }

    int predict(const vector<double>& input)
    {
        feed_input(input);
        vector<double> nnOut = layers[numLayers - 1] -> getAct();

        int max_id = 0;
        for(int j = 1; j < (int)nnOut.size(); j++)
        {
            if(nnOut[j] > nnOut[max_id])
            {
                max_id = j;
            }
        }

        return max_id;
    }

    double evaluate(const vector<pair< vector<double>, vector<double> > >& data)
    {
        for(int i = 1; i < numLayers - 1; i++)
        {
            layers[i]->noDropout();
        }

        const double eps = 1e-9;

        int correct = 0;
        int sz = data.size();
        vector<double> nnOut;
        double loss = 0;
        for(int i = 0; i < sz; i++)
        {
            feed_input(data[i].first);

            nnOut = layers[numLayers - 1] -> getAct();
            loss += costFunction->calculate(nnOut, data[i].second);

            int max_id = 0;
            for(int j = 1; j < (int)nnOut.size(); j++)
            {
                if(nnOut[j] > nnOut[max_id])
                {
                    max_id = j;
                }
            }
            if(data[i].second[max_id] > eps)
            {
                correct++;
            }
        }

        loss /= data.size();
        cerr << "Loss: " << loss << endl;

        return (100.0 * correct / sz);
    }

    void train(vector<pair<vector<double>, vector<double> > > train_data, const int& NUM_EPOCHS, const int& BATCH_SIZE)
    {
        for(int j = 0; j < NUM_EPOCHS; j++)
        {
            random_shuffle(train_data.begin(), train_data.end());
            int num_batches = train_data.size() / BATCH_SIZE;
            for(int i = 0; i < num_batches; i++)
            {
                handle_minibatch(train_data, i, BATCH_SIZE);
                //cerr << i + 1 << "-th batch complete" << endl;
            }
            cerr << "Epoch " << j+1 << ": complete.\n";

            double result = evaluate(train_data);
            cerr << "Result: " << result << "%" << endl;

            char toSave[30] = "models/epoch2_";
            char epochStr[5];
            itoa(j+1, epochStr, 10);
            strcat(toSave, epochStr);
            strcat(toSave, ".txt");
            save(toSave);
            cerr<<"Time: "<<(int)(clock() * 1000.0 / CLOCKS_PER_SEC)<<" ms."<<endl;
            cerr << "------------------------------" << endl;
        }
    }

    void save(const char filename[])
    {
        ofstream out(filename);
        for(int i = 1; i < numLayers; i++)
        {

            layers[i] -> save(out);
        }
        out.close();
    }

    void load(const char filename[])
    {
        ifstream in(filename);
        for(int i = 1; i < numLayers; i++)
        {
            layers[i]->load(in);
        }
        in.close();
    }

};


#endif // NEURALNETWORK_H_INCLUDED
