#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>

#include "NeuralNetwork.h"

using namespace std;

NeuralNetwork NN(0.1, 2, "crossentropy");
vector<pair< vector<double>, vector<double> > > train_data;

vector<int> split(string s) {
  int i, curr = 0;
  vector < int > ans;

  for(i=0;i<(int)(s.size());i++) {
    if(s[i]==',') {
      ans.push_back(curr);
      curr = 0;
    }
    else {
      curr *= 10;
      curr += s[i] - '0';
    }
  }

  ans.push_back(curr);

  return ans;
}

void parse_train_data()
{
    ifstream IN("data/train.csv");
    int i, j;
    string trash;
    vector < int > v;
    vector<double> input, output;
    input.resize(784);
    output.resize(10);

    train_data.reserve(42000);

    IN>>trash;
    for(i=0;i<42000;i++) {
    IN>>trash;

    v = split(trash);

    for(int i = 0; i < output.size(); i++)
        output[i] = 0;
    output[v[0]] = 1.0;

    for(j=1;j<785;j++) {
      input[j - 1] = v[j] / 255.0;
    }

    train_data.push_back({input, output});
    }

    cerr<<"Training data loaded!"<<endl;
}

void test() {
    ifstream IN("data/test.csv");
    ofstream OUT("data/ans2.csv");
    string trash;
    vector < int > v;
    int i, j, idx;
    double max_value;
    vector<double> curr_input, curr_output;
    curr_input.resize(784);

    OUT<<"ImageId,Label"<<endl;

    IN>>trash;

    for(i=0;i<28000;i++) {
        IN>>trash;
        v = split(trash);

        for(j=0;j<784;j++) {
            curr_input[j] = v[j] / 255.0;
        }

        int predicted_output = NN.predict(curr_input);

        OUT<< i+1 << "," << predicted_output <<endl;
    }

    OUT.close();
}

int main()
{
	/*ios_base::sync_with_stdio(0);
	cin.tie(0);*/


	NN.addLayer(DenseLayer(0, 784, 0, "sigmoid"));
	NN.addLayer(ConvLayer(1, 28, 28, 5, 5, "sigmoid"));
	NN.addLayer(PoolLayer(5, 24, 24, 2, "max"));
	//NN.addLayer(ConvLayer(20, 12, 12, 20, 5, "sigmoid"));
	//NN.addLayer(PoolLayer(20, 8, 8, 2, "max"));
	NN.addLayer(DenseLayer(5*12*12, 100, 0.5, "sigmoid"));
	NN.addLayer(DenseLayer(100, 10, 0, "sigmoid"));


	parse_train_data();

	NN.load("models/epoch2_26.txt");
	//NN.train(train_data, 60, 10);

	test();


	return 0;
}
