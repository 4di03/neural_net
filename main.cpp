#include "autograd.h"
#include "vis.h"
#include "network.h"
using namespace operation; 

int main()
{



    // comp graph of a single neuron
    {
    auto x1 = make_value(2.0, "x1");
    auto x2 = make_value(0.0, "x2");

    auto w1 = make_value(-3.0, "w1");
    auto w2 = make_value(1.0, "w2");

    auto bias = make_value(6.88137358, "b");

    auto x1w1 = x1 * w1;
    x1w1 -> set_label("x1 * w1");
    auto x2w2 = x2 * w2;
    x2w2 -> set_label("x2 * w2");

    auto x1w1x2w2 = x1w1 + x2w2;
    x1w1x2w2 -> set_label("x1w1 + x2w2");
    auto n = x1w1x2w2 + bias;
    n -> set_label("n");

    auto out = tanh(n);
    out -> set_label("out");

    // backprop for the fixed values
    out->backward();
    
    WRITE_PNG(out, "neuron_comp_graph.png");   // produces graph.png
    }

    // comp graph without tanh
    {
    auto x1 = make_value(2.0, "x1");
    auto x2 = make_value(0.0, "x2");

    auto w1 = make_value(-3.0, "w1");
    auto w2 = make_value(1.0, "w2");

    auto bias = make_value(6.88137358, "b");

    auto x1w1 = x1 * w1;
    x1w1 -> set_label("x1 * w1");
    auto x2w2 = x2 * w2;
    x2w2 -> set_label("x2 * w2");

    auto x1w1x2w2 = x1w1 + x2w2;
    x1w1x2w2 -> set_label("x1w1 + x2w2");
    auto n = x1w1x2w2 + bias;
    n -> set_label("n");    
    // tanh(n) = exp(2n) - 1 / (exp(2n) + 1)
    auto out = (exp(2 * n) - 1) / (exp(2 * n) + 1);
    out -> set_label("out");

    // backprop for the fixed values
    out->backward();

    WRITE_PNG(out, "neuron_comp_graph_no_tanh.png");   // produces graph.png
    }
    // test case where we reuse dependency, grad should be 2
    {
    auto a = make_value(3.0, "a");
    auto b = a + a;
    b->set_label("b = a + a");
    
    b->backward();
    WRITE_PNG(b, "reuse_dep_graph.png");   // produces graph.png
    }
    {
      auto a = make_value(-2.0, "a");
      auto b = make_value(3.0, "b");
      auto d = a* b;
      auto e = a+b;
      auto f = d*e;
      f->backward();
      WRITE_PNG(f, "complex_graph.png");
    }

    // make a simple fully connected network and do a forward pass
    {
      FullyConnectedNetwork net(3, {4,4,1}); // 3 inputs, 2 hidden layers of 4 neurons each, 1 output

      network_output_t inputs;
      inputs.push_back(make_value(1.0, "input1"));
      inputs.push_back(make_value(0.0, "input2"));
      inputs.push_back(make_value(-1.0, "input3"));
      auto outputs = net(inputs);
      outputs[0]->set_label("network_output");
      outputs[0]->backward();
      //WRITE_PNG(outputs[0], "fcc_network_comp_graph.png");

    }
    // compute loss over a batch and backpropagate
    {
      FullyConnectedNetwork net(3, {4,4,1}); // 3 inputs, 2 hidden layers of 4 neurons each, 1 output

      std::array<std::shared_ptr<Value>, 3> x1_arr = {
        make_value(1.0, "input1.1"), make_value(0.0, "input1.2"), make_value(-1.0, "input1.3")
      };
      std::array<std::shared_ptr<Value>, 3> x2_arr = {
        make_value(0.0, "input2.1"), make_value(1.0, "input2.2"), make_value(2.0, "input2.3")
      };
      std::array<std::shared_ptr<Value>, 3> x3_arr = {
        make_value(-1.0, "input3.1"), make_value(-1.0, "input3.2"), make_value(1.0, "input3.3")
      };

      std::vector<network_input_t> X{
        x1_arr, x2_arr, x3_arr
      };

      std::vector<float> expected_outputs = {1.0f, -1.0f, 0.0f};

      size_t n_steps = N_EPOCHS;
      float learning_rate = LEARNING_RATE;
      Optimizer opt(net.trainable_parameters(), learning_rate);
      for (size_t i = 0 ; i < n_steps; i++) {

        auto outputs = net(X);
        std::shared_ptr<Value> loss = make_value(0.0, "loss");
        for (size_t i = 0; i < outputs.size(); i++) {
          auto diff = outputs[i][0] - expected_outputs[i];
          loss  = loss + (diff * diff);
        }

        std::cout << "Step " << i << ", loss: " << loss->get_data() << std::endl;

        // zero grads
        opt.zero_grad();
        // backprop from loss
        loss->backward();

        // take an optimizer step
        opt.step();
        if (i%5 == 0 || i == n_steps - 1) {
          WRITE_PNG(loss, "fcc_trained_network_after_step[" + std::to_string(i) + "].png");
        }
      } 

      // final predictions after training
      auto final_outputs = net(X);
      for (size_t i = 0; i < final_outputs.size(); i++) {
        std::cout << "Final output for input " << i << ": " << final_outputs[i][0]->get_data() << std::endl;
        std::cout << "Expected output: " << expected_outputs[i] << std::endl;
      }


    }

    

    return 0;
}
