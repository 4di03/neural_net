// header for building blocks of neural network
#include "network.h"
#include "constants.h"


using namespace operation;


void print_value(std::shared_ptr<Value> v, const std::string& name) {
    std::cout << name << ": ";
    std::cout << v->get_data() << "";
    std::cout << "\n";
}


void print_vector(network_input_t vec, const std::string& name) {
    std::cout << name << ": [ ";
    for (const auto& v : vec) {
        if (!v) {
            std::cout << "nullptr ";
        } else {
            std::cout << v->get_data() << " ";
        }
    }
    std::cout << "]\n";
}

// initialize a neuron that takes in num_inputs inputs
Neuron::Neuron(int num_inputs, int layer_index, int neuron_index)
{   
    weights.reserve(num_inputs);
    // initialize weights and bias
    for (int weight_index = 0; weight_index < num_inputs; weight_index++)
    {
        // rand number between -1 and 1
        auto neuron_label = "L" + std::to_string(layer_index) + "N" + std::to_string(neuron_index) + "W" + std::to_string(weight_index);
        weights.push_back(make_value(static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2 - 1, neuron_label));
    }
    bias = make_value(0.0f, "L" + std::to_string(layer_index) + "N" + std::to_string(neuron_index) + "B");
}

std::shared_ptr<Value> Neuron::operator()(network_input_t x) const
{
    // compute weighted sum of inputs + bias
    if (x.size() != weights.size())
    {
        throw std::invalid_argument("Input size does not match weight size, input size: " + std::to_string(x.size()) + ", weight size: " + std::to_string(weights.size()));
    }

    DBG(
        print_vector(x, "Input to neuron");
    );



    std::shared_ptr<Value> out = bias;
    for (size_t i = 0; i < x.size(); i++)
    {
        out = (out + ((weights[i]) * (x[i])));
    }

    DBG(
        print_value(out, "Output from dot product");
    );

    // apply tanh activation
    out = tanh(out);


    DBG(
        print_value(out, "Output from tanh");
    );
    return out;
};

// last parameter is bias
std::vector<std::shared_ptr<const Value>> Neuron::parameters() const
{
    std::vector<std::shared_ptr<const Value>> out;
    out.reserve(weights.size() + 1);

    for (const auto &w : weights)
    { // shared_ptr<Value> → shared_ptr<const Value>
        out.push_back(w);
    }

    if (bias)
    {
        out.push_back(bias);
    }

    return out;
}

// initialize a layer with num_inputs inputs and num_outputs outputs, creating num_outputs neurons that each take in num_inputs inputs
FullyConnectedLayer::FullyConnectedLayer(int num_inputs, int num_outputs, int layer_index)
{
    neurons.reserve(num_outputs);
    for (int neuron_index = 0; neuron_index < num_outputs; neuron_index++)
    {
        neurons.emplace_back(num_inputs, layer_index, neuron_index); // construct neuron in place
    }
}

network_output_t FullyConnectedLayer::operator()(network_input_t x) const
{
    
    DBG(
        print_vector(x, "Input to layer");
    );

    network_output_t out;
    out.reserve(neurons.size());
    for (const auto &neuron : neurons)
    {
        out.push_back(
            neuron(x)
        );
    }

    DBG(
        print_vector(out, "Output from layer");
    );

    return out;
};

// initialize a fully connected network with layer_sizes defining the number of neurons in each layer, and num_inputs defining the number of inputs to the network
FullyConnectedNetwork::FullyConnectedNetwork(int num_inputs, const std::vector<int> &layer_sizes)
{
    layers.reserve(layer_sizes.size());
    int current_input_size = num_inputs;
    for (size_t i = 0; i < layer_sizes.size(); i++)
    {
        int layer_size = layer_sizes[i];
        layers.emplace_back(current_input_size, layer_size, i);
        current_input_size = layer_size;
    }
}

network_output_t FullyConnectedNetwork::operator()(network_input_t x) const
{
    network_output_t out(x.begin(), x.end());


    for (const auto &layer : layers)
    {   
        DBG(
        print_vector(out, "Input to network layer");
        );
        out = layer(out);
        DBG(
        print_vector(out, "Output from network layer");
        );
    }
    return out;
}
