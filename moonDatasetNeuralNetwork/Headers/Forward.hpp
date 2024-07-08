#pragma once
#include <vector>
#include "Global.hpp"
std::vector<double> ForwardProp(int neurons, std::vector<double>& inputs, std::vector<std::vector<double>>& weights, std::vector<double> bias);