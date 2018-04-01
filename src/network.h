/*MIT License

Copyright (c) 2018 Minglei Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.*/

#ifndef __network_h
#define __network_h
#include <stdio.h>
#include "fast_go.h"
#include "board.h"
#include <tensorflow/core/public/session.h>
#include <tensorflow/cc/saved_model/loader.h>

using namespace tensorflow;

class Network
{
private:
    SavedModelBundle bundle;
    SessionOptions session_options;
    RunOptions run_options;
    bool model_loaded;
public:
    Network()
    {
        model_loaded = false;
    }

    int LoadModel(std::string path, std::string tag_name)
    {
        session_options.config.mutable_gpu_options()->set_allow_growth(true);

        std::string export_dir = path; //"/Users/wangm/PycharmProjects/gozero/metagraphb/";
        Status status = LoadSavedModel(session_options, run_options, export_dir, {tag_name}, &bundle);
        if (!status.ok()) {
            std::cout << status.ToString() << "\n";
            return 1;
        }
        model_loaded = true;
        return 0;
    }
    
    bool Forward_Simulator(Tensor& states, std::vector<std::array<float, NN+1>>& action_probs_vector, std::vector<float>& values)
    {
        std::mt19937 gen((unsigned int)time(NULL));
        std::uniform_real_distribution<> dis(0.0, 1.0);
        
        for(int c=0; c<states.dim_size(0); c++)
        {
            std::array<float, NN+1> action_probs;
            
            for(int i=0; i<NN+1; i++)
                action_probs[i] = dis(gen);
            action_probs_vector.push_back(action_probs);
            
            float value = std::tanh(10*(dis(gen)-0.5));
            values.push_back(value);
        }
        
        return true;
    }
    
    bool Forward(Tensor& states, std::vector<std::array<float, NN+1>>& action_probs_vector, std::vector<float>& values)
    {
        
        if(!model_loaded)
            return false;
        
        std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
            { "training_network/states", states },
        };

        // The session will initialize the outputs
        std::vector<tensorflow::Tensor> outputs;

        // Run the session, evaluating our "c" operation from the graph
        Status status = bundle.session->Run(inputs, {"training_network/policy_head/out_action_prob", "training_network/value_head/out_value"}, {}, &outputs);
        if (!status.ok()) {
            std::cout << status.ToString() << "\n";
            return 1;
        }
        
        auto probs_mapped = outputs[0].tensor<float,2>();
        for(int c=0; c<states.dim_size(0); c++)
        {
            std::array<float, NN+1> action_probs;
            
            for(int i=0; i<NN+1; i++)
                action_probs[i] = probs_mapped(c,i);
            action_probs_vector.push_back(action_probs);
            
            auto values_mapped = outputs[1].tensor<float, 2>();
            float value = values_mapped(c,0);
            values.push_back(value);
        }
        return true;
    }
};

#endif
