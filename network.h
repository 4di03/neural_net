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
    std::vector<std::shared_ptr<const Value>> parameters() const;
private:
    network_output_t weights;
    std::shared_ptr<Value> bias;

};

class FullyConnectedLayer{
    public:
    // initialize a layer with num_inputs inputs and num_outputs outputs, creating num_outputs neurons that each take in num_inputs inputs
    FullyConnectedLayer(int num_inputs, int num_outputs, int layer_index);
    network_output_t operator()(network_input_t x) const ;
    private:
    // no shared_ptr since the neurons are owned by the layer
    std::vector<Neuron> neurons;
};


class FullyConnectedNetwork {
public:
    // initialize a fully connected network with layer_sizes defining the number of neurons in each layer, and num_inputs defining the number of inputs to the network
    FullyConnectedNetwork(int num_inputs, const std::vector<int>& layer_sizes);
    network_output_t operator()(network_input_t x) const;

private:
    std::vector<FullyConnectedLayer> layers;
};

