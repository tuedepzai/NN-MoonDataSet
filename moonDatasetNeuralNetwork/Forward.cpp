#include "Headers/Forward.hpp"
#include "Headers/Global.hpp"
#include "Headers/Activation.hpp"
#include <vector>
#include<array> 
#include <iostream>
#include <iomanip>
//don't add bias yet!
std::vector<double> ForwardProp(int neurons, std::vector<double>& inputs, std::vector<std::vector<double>>& weights, std::vector<double> bias) {
	std::vector<double> Result;
	int input_size = inputs.size();
	//	std::cerr << "work here FIRST" << '\n';
	int wcols = weights.size();
	int wrows = weights[1].size();


	//std::cerr << input_size << " " << wcols << " " << wrows << " " << bias.size() << '\n';

	for (int i = 1; i < wcols; i++) {
		double res = 0;
		for (int j = 1; j < wrows; j++) {
			res += (double)weights[i][j] * (double)inputs[j];
			//	std::cerr << "STILL WORK HERE: " << weights[i][j] << '\n';
		}

		double sig = sigmoid(res + bias[i]);
		Result.push_back(sig);
		//	std::cerr << "STILL WORK HERE TOO" << '\n';
	}
	//	std::cerr << "END HERE" << '\n';

	return Result;
}