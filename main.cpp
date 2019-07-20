#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>

#include "NeuralNetwork.h"

using namespace std;

NeuralNetwork NN(0.5, 4.2, "crossentropy");
vector<Matrix> train_input, train_output;

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
    Matrix input(1, 784), output(1, 10);

    train_input.reserve(42000);
    train_output.reserve(42000);

    IN>>trash;
    for(i=0;i<42000;i++) {
    IN>>trash;

    v = split(trash);

    output.zero();
    output[0][v[0]] = 1.0;

    for(j=1;j<785;j++) {
      input[0][j - 1] = v[j] / 255.0;
    }

    train_input.push_back(input);
    train_output.push_back(output);
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
    Matrix curr_input(1, 784), curr_output;

    OUT<<"ImageId,Label"<<endl;

    IN>>trash;

    for(i=0;i<28000;i++) {
        IN>>trash;
        v = split(trash);

        for(j=0;j<784;j++) {
            curr_input[0][j] = v[j] / 255.0;
        }

        int predicted_output = NN.predict(curr_input);

        OUT<< i+1 << "," << predicted_output <<endl;
    }

    OUT.close();
}

int main()
{
	ios_base::sync_with_stdio(0);
	cin.tie(0);


	NN.addLayer(784);
	NN.addLayer(100);
	NN.addLayer(10);

	parse_train_data();

	NN.train(train_input, train_output, 60, 10);

	test();


	return 0;
}
