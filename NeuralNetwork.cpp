#include "NeuralNetwork.h"

#include <ctime>


NeuralNetwork::NeuralNetwork(double _eta, double _lambda, string costFunctionName) : eta(_eta), weight_lambda(_lambda), numLayers(0)
{
    if(costFunctionName == "quadratic")
    {
        costFunction = new QuadraticCostFunction();
    }

    if(costFunctionName == "crossentropy")
    {
        costFunction = new CrossentropyCostFunction();
    }

}

NeuralNetwork::~NeuralNetwork()
{
    delete costFunction;
}

void NeuralNetwork::addLayer(int _layerSize)
{
    numLayers ++;

    layerSize.push_back(_layerSize);

    if(layerSize.size() == 1)
    {
        z.push_back(Matrix(0,0));
        error.push_back(Matrix(0,0));
        weight.push_back(Matrix(0,0));
        bias.push_back(Matrix(0,0));
        act.push_back(Matrix(1,layerSize[0]));
        delta_b.push_back(Matrix(0,0));
        delta_w.push_back(Matrix(0,0));
    }

    else
    {
        Matrix inp_w(layerSize[layerSize.size()-2], _layerSize);
        inp_w.randomize();
        weight.push_back(inp_w);
        Matrix inp_b(1, _layerSize);
        inp_b.randomize();
        bias.push_back(inp_b);
        z.push_back(Matrix(1, _layerSize));
        act.push_back(Matrix(1, _layerSize));
        error.push_back(Matrix(1, _layerSize));
        delta_b.push_back(Matrix(1, _layerSize));
        delta_w.push_back(Matrix(layerSize[layerSize.size() - 2], _layerSize));
    }
}

void NeuralNetwork::feed_input(const Matrix& input)
{
    act[0] = input;
    for(int i = 1; i < numLayers; i++)
    {
        z[i] = act[i-1] * weight[i] + bias[i];
        act[i] = z[i].sigmoid();
    }
}

void NeuralNetwork::backprop(const Matrix& output)
{
    error[numLayers - 1] = costFunction -> derivative(act[numLayers - 1], output, z[numLayers - 1]);
    delta_b[numLayers - 1] += error[numLayers - 1];
    delta_w[numLayers - 1] += ((act[numLayers - 2].transpose()) * error[numLayers - 1]);
    for(int i = numLayers - 2; i > 0; i--)
    {
        error[i] = ( error[i+1] * (weight[i + 1].transpose()) ) % (z[i].sigmoid_derivative());
        delta_b[i] += error[i];
        delta_w[i] += ((act[i-1].transpose()) * error[i]);
    }
}

void NeuralNetwork::handle_minibatch(const vector<Matrix>& inputs, const vector<Matrix>& outputs, int batch_num, int batch_sz)
{
    for(int i = 0; i < numLayers; i++)
    {
        delta_b[i].zero();
        delta_w[i].zero();
    }
    for(int i = 0; i < batch_sz; i++)
    {
        feed_input(inputs[batch_sz * batch_num + i]);
        backprop(outputs[batch_sz * batch_num + i]);
    }

    for(int i = 0; i < numLayers; i++)
    {
        delta_b[i] *= (eta/batch_sz);
        delta_w[i] *= (eta/batch_sz);

        weight[i] *= (1 - weight_lambda * eta / inputs.size()); /// if you want smaller weights

        weight[i] -= delta_w[i];
        bias[i] -= delta_b[i];
    }
}

double NeuralNetwork::evaluate(const vector<Matrix>& inputs, const vector<Matrix>& outputs)
{
    const double eps = 1e-9;

    int correct = 0;
    int sz = inputs.size();
    for(int i = 0; i < sz; i++)
    {
        feed_input(inputs[i]);
        int max_id = 0;
        for(int j = 1; j < (int)layerSize[numLayers -1 ]; j++)
        {
            if(act[numLayers-1].a[0][j] > act[numLayers-1].a[0][max_id])
            {
                max_id = j;
            }
        }
        if(outputs[i].a[0][max_id] > eps)
        {
            correct++;
        }
    }
    return (100.0 * correct / sz);
}

int NeuralNetwork::predict(const Matrix& input)
{
    feed_input(input);
    int max_id = 0;
    for(int j = 1; j < (int)layerSize[numLayers - 1]; j++)
    {
        if(act[numLayers-1][0][j] > act[numLayers-1][0][max_id])
        {
            max_id = j;
        }
    }

    return max_id;
}

void NeuralNetwork::train(const vector<Matrix>& inputs, const vector<Matrix>& outputs, const int NUM_EPOCHS, const int BATCH_SIZE)
{
    vector<Matrix> batch_input;
    vector<Matrix> batch_output;
    vector<int> idx;
    for(int i = 0; i < (int)inputs.size(); i++)
    {
        idx.push_back(i);
    }
    for(int j = 0; j < NUM_EPOCHS; j++)
    {
        int num_batches = inputs.size() / BATCH_SIZE;
        random_shuffle(idx.begin(), idx.end());
        for(int i = 0; i < num_batches; i++)
        {
            /*batch_input.clear();
            batch_output.clear();
            for(int l = 0; l < BATCH_SIZE; l++)
            {
                batch_input.push_back(inputs[idx[i*BATCH_SIZE + l]]);
                batch_output.push_back(outputs[idx[i*BATCH_SIZE + l]]);
            }*/
            handle_minibatch(inputs, outputs, i, BATCH_SIZE);
            //cerr << i << "-th batch complete" << endl;
        }
        cerr << j+1 << "-th epoch complete.\n";

        double result = evaluate(inputs, outputs);
        cerr << "Result:" << result << "%" << endl;

        cerr<<"Time: "<<(int)(clock() * 1000.0 / CLOCKS_PER_SEC)<<" ms."<<endl;
    }
}
