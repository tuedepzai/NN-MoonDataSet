#include <chrono>
#include <random>
#include <vector>
#include <iostream>


#include "SFML/Graphics.hpp"
#include "Headers/Global.hpp"
#include "Headers/Forward.hpp"
#include "rapidcsv.h"

float precision(float f, int places)
{
	float n = std::pow(10.0f, places);
	return std::round(f * n) / n;
}


int main()
{
	sf::RenderWindow window(sf::VideoMode(SCREEN_WIDTH, SCREEN_HEIGHT), "untited window");


	// Init random engine //
	std::mt19937_64 gen(std::chrono::system_clock::now().time_since_epoch().count());
	std::uniform_real_distribution<double> randx(0.0, 1.0);

	//Process Inputfile //
	rapidcsv::Document  doc("C:/Users/doget/source/repos/moonDatasetNeuralNetwork/moonDatasetNeuralNetwork/Resources/Data/moonDataset.csv");
	std::vector<double> X1 = doc.GetColumn<double>("X1");
	std::vector<double> X2 = doc.GetColumn<double>("X2");
	std::vector<double> X3 = doc.GetColumn<double>("X3");
	std::vector<double> label = doc.GetColumn<double>("label");

	for (int i = 0; i < X1.size(); i++) {
		X1[i] = precision(X1[i], 2);
	}
	for (int i = 0; i < X2.size(); i++) {
		X2[i] = precision(X2[i], 2);
	}
	for (int i = 0; i < X3.size(); i++) {
		X3[i] = precision(X3[i], 2);
	}
	std::vector<std::vector<double>> y_hat;
	for (int i = 0; i < label.size(); i++) {
		double yes = label[i] == 1, no = label[i] == 0;
		y_hat.push_back({ no, yes });
	}


	// Init neurons and random weight and random bias;

	std::vector<std::vector<double>> l1weights;
	std::vector<double> l2bias(LayerNumbers[1] + 1);
	l1weights.resize(LayerNumbers[1] + 1, std::vector<double>(LayerNumbers[0] + 1));
	for (int i = 1; i <= LayerNumbers[1]; i++) {
		for (int j = 1; j <= LayerNumbers[0]; j++) {
			l1weights[i][j] = randx(gen);
		}
	}
	for (int i = 1; i < l2bias.size(); i++) {
		l2bias[i] = randx(gen);
	}


	// Forward Layer1 //
	std::vector<std::vector<double>> layer2Activation = { {} };
	for (int i = 1; i <= (int)X1.size(); i++) {
		std::vector<double> inputs(LayerNumbers[0] + 1);
		inputs[1] = X1[i - 1];
		inputs[2] = X2[i - 1];
		inputs[3] = X3[i - 1];
		std::vector<double> out = ForwardProp(3, inputs, l1weights, l2bias);
		layer2Activation.push_back(out);
	}
	std::cout << "LAYER 2 RESULT: " << "\n";
	for (std::vector<double> k : layer2Activation) {
		for (double q : k) {
			std::cout << q << " ";
		}
		std::cout << '\n';
	}
	// Layer2 forward //
	std::vector<std::vector<double>> l2weights;
	l2weights.resize(LayerNumbers[2] + 1, std::vector<double>(LayerNumbers[1] + 1));
	for (int i = 1; i <= LayerNumbers[2]; i++) {
		for (int j = 1; j <= LayerNumbers[1]; j++) {
			l2weights[i][j] = randx(gen);
		}
	}

	std::vector<double> l3bias(LayerNumbers[2] + 3);
	for (int i = 1; i < l3bias.size(); i++) {
		l3bias[i] = randx(gen);
	}
	std::vector<std::vector<double>> layer3Activation = { {} };
	for (int i = 1; i < (int)layer2Activation.size(); i++) {
		std::vector<double> inputs(LayerNumbers[1] + 1);
		inputs[1] = layer2Activation[i][0];
		inputs[2] = layer2Activation[i][1];
		inputs[3] = layer2Activation[i][2];
		inputs[4] = layer2Activation[i][3];

		std::vector<double> out = ForwardProp(3, inputs, l2weights, l3bias);
		layer3Activation.push_back(out);
	}

	std::cout << "LAYER 3 RESULT: " << "\n";
	for (std::vector<double> k : layer3Activation) {
		for (double q : k) {
			std::cout << q << " ";
		}
		std::cout << '\n';
	}
	// Setup MSE //

	double sum = 0;
	for (int i = 1; i < layer3Activation.size(); i++) {
		if (layer3Activation[i].size() == 0) continue;
		sum += (layer3Activation[i][0] - y_hat[i - 1][0]) * (layer3Activation[i][0] - y_hat[i - 1][0]) + (layer3Activation[i][1] - y_hat[i - 1][1]) * (layer3Activation[i][1] - y_hat[i - 1][1]);

	}
	std::cout << "Mean Squared Error (MSE): " << sum / (layer3Activation.size() - 1) << '\n';


	// Keep window open //
	while (window.isOpen()) {
		sf::Event event;
		while (window.pollEvent(event)) {
			if (event.type == sf::Event::Closed) {
				window.close();
			}
		}
		window.clear(sf::Color::Black);
		window.display();
	}
}
