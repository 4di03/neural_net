// header for building blocks of neural network
#include "autograd.h"
#include "operation.h"


/**
 * Spans for inputs to force lightweight views on the pointers, and vectors for outputs for flexibility.
 */
using network_input_t = std::span<const std::shared_ptr<Value>>;
using network_output_t = std::vector<std::shared_ptr<Value>>;



class Neuron {

public:
    
    // initialize a neuron that takes in num_inputs inputs. we also store the index of the layer and neuron for visualization purposes
    Neuron(int num_inputs, int layer_index, int neuron_index);
    std::shared_ptr<Value> operator()(network_input_t x) const;
    const std::vector<std::shared_ptr<Value>> trainable_parameters() const; // a list of all trainable parameters in the network
private:
    network_output_t weights;
    std::shared_ptr<Value> bias;
    std::vector<std::shared_ptr<Value>> get_trainable_parameters() const;

};

class FullyConnectedLayer{
    public:
    // initialize a layer with num_inputs inputs and num_outputs outputs, creating num_outputs neurons that each take in num_inputs inputs
    FullyConnectedLayer(int num_inputs, int num_outputs, int layer_index);
    network_output_t operator()(network_input_t x) const ;
    const std::vector<std::shared_ptr<Value>> trainable_parameters() const; // a list of all trainable parameters in the layer

    private:
    // no shared_ptr since the neurons are owned by the layer
    std::vector<Neuron> neurons;

};


class FullyConnectedNetwork {
public:
    // initialize a fully connected network with layer_sizes defining the number of neurons in each layer, and num_inputs defining the number of inputs to the network
    FullyConnectedNetwork(int num_inputs, const std::vector<int>& layer_sizes);
    network_output_t operator()(network_input_t x) const;
    std::vector<network_output_t> operator()(std::vector<network_input_t>& x) const;
    const std::vector<std::shared_ptr<Value>>& trainable_parameters() const; // a list of all trainable parameters in the network

private:
    std::vector<FullyConnectedLayer> layers;
    std::vector<std::shared_ptr<Value>> trainable_params_cache;
};



class Optimizer {
public:
    Optimizer(const std::vector<std::shared_ptr<Value>>& parameters, float learning_rate) : parameters(parameters), learning_rate(learning_rate) {}
    
    // updates the parameters using their gradients and the learning rate
    void step();

    // zeros out all gradients in the parameters, to be used before a new backward pass
    void zero_grad();
private:
    const std::vector<std::shared_ptr<Value>>& parameters;
    float learning_rate;
};