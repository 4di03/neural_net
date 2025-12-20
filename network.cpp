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

const std::vector<std::shared_ptr<Value>> Neuron::trainable_parameters() const
{
    std::vector<std::shared_ptr<Value>> out;
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

const std::vector<std::shared_ptr<Value>>  FullyConnectedLayer::trainable_parameters() const
{
    std::vector<std::shared_ptr<Value>> out;
    for (const auto &neuron : neurons)
    {
        auto neuron_params = neuron.trainable_parameters();
        out.insert(out.end(), neuron_params.begin(), neuron_params.end());
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

const std::vector<std::shared_ptr<Value>>& FullyConnectedNetwork::trainable_parameters() const
{
    // return reference to the cached parameers
    return trainable_params_cache;
}

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

    // cache trainable parameters, TODO: if we implement/allow dynamic layers with changing sizes/nodes after instantiation, we need to update this cache
    for (const auto &layer : layers)
    {
        auto layer_params = layer.trainable_parameters();
        trainable_params_cache.insert(trainable_params_cache.end(), layer_params.begin(), layer_params.end());
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


std::vector<network_output_t> FullyConnectedNetwork::operator()(std::vector<network_input_t>& x) const
{
    std::vector<network_output_t> outputs;
    outputs.reserve(x.size());

    for (const auto& single_input : x)
    {
        auto out = (*this)(single_input);
        outputs.push_back(out);
    }

    return outputs;
}


void Optimizer::step()
{
    for (const auto& param : parameters)
    {
        float current_value = param->get_data();
        float grad = param->get_grad(); // grad w.r.t some loss

        /* nudge the parameter in the direction that reduces the loss
        * if the gradient is positive, the loss is increasing as the parameter increases. To minimize, we need to decrease the value for the parameter.
        * if the gradient is negative, the loss is decreasing as the parameter increases. To minimize, we need to increase the value for the parameter.
        * so we subtract learning_rate * grad from the current value, as we just move by some small amount in the direction opposite to the gradient
        */
        float new_value = current_value - learning_rate * grad;

        param->set_data(new_value);
    }
}


void Optimizer::zero_grad()
{
    for (const auto& param : parameters)
    {
        param->set_grad(0.0f);
    }
}

