#ifndef NEURALNETWORK_H_INCLUDED
#define NEURALNETWORK_H_INCLUDED

#include "QuadraticCostFunction.h"
#include "CrossentropyCostFunction.h"
#include <cstring>
#include <vector>
#include <algorithm>

using namespace std;

class NeuralNetwork
{

public:

    int numLayers;
    double eta;
    double weight_lambda;

    vector<int> layerSize;
    vector<Matrix> weight;
    vector<Matrix> bias;
    vector<Matrix> z;
    vector<Matrix> act;
    vector<Matrix> error;
    vector<Matrix> nabla_b;
    vector<Matrix> nabla_w;
    vector<Matrix> delta_b;
    vector<Matrix> delta_w;

    CostFunction* costFunction;

    NeuralNetwork(double _eta, double _lambda, string costFunctionName);

    ~NeuralNetwork();

    void addLayer(int _layerSize);

    void feed_input(const Matrix& input);

    void backprop(const Matrix& output);

    void handle_minibatch(const vector<Matrix>& inputs, const vector<Matrix>& outputs, int batch_num, int batch_sz);

    double evaluate(const vector<Matrix>& inputs, const vector<Matrix>& outputs);

    void train(const vector<Matrix>& inputs, const vector<Matrix>& outputs, const int NUM_EPOCHS, const int BATCH_SIZE);

    int predict(const Matrix& input);
};


#endif // NEURALNETWORK_H_INCLUDED
