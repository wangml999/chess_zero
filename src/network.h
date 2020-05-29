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
//#include "fast_go.h"
//#include "board.h"
#include "bit_board.h"
#include "chess_board.h"
#include <tensorflow/core/public/session.h>
#include <tensorflow/cc/saved_model/loader.h>

using namespace tensorflow;

void dump1(Tensor* pt)
{
	float* ptensor_data = pt->flat<float>().data();
	cout << setprecision(5);
	cout << "-------START------" << endl; 
	for(int i=0; i<pt->dim_size(0); i++)
	{
		for(int j=0; j<pt->dim_size(1); j++)
		{
			for(int m=0; m<pt->dim_size(2); m++)
			{
				for(int n=0; n<pt->dim_size(3); n++)
				{
					cout << *ptensor_data << " ";
					ptensor_data++;
				}
				cout << endl;
			}
			cout << endl;
		}
		cout << endl;
		cout << "=================" << endl; 
		getchar();
	}
	cout << "-------END------" << endl; 
}

void dump2(Tensor* pt)
{
	float* ptensor_data = pt->flat<float>().data();
	cout << setprecision(5);
	for(int i=0; i<pt->dim_size(0); i++)
	{
		for(int j=0; j<pt->dim_size(1); j++)
		{
			cout << *ptensor_data;
			ptensor_data++;
		}
		cout << endl;
		getchar();
	}
}

class Network
{
private:
	Session* session;
    SavedModelBundle bundle;
    SessionOptions session_options;
    RunOptions run_options;
    bool model_loaded;
	int model_version;
	bool warmed;
public:
	int dim1;
    Network()
    {
        model_loaded = false;
		warmed = false;
    }

	~Network()
	{
		session->Close();
		delete session;
	}

    int LoadModel(std::string path, std::string tag_name)
    {
        session_options.config.mutable_gpu_options()->set_allow_growth(true);
		Status status = NewSession(session_options, &session);
		if (!status.ok()) {
			std::cout << status.ToString() << "\n";
			return 1;
		}

		// Read in the protobuf graph we exported
		// (The path seems to be relative to the cwd. Keep this in mind
		// when using `bazel run` since the cwd isn't where you call
		// `bazel run` but from inside a temp folder.)
		GraphDef graph_def;
		status = ReadBinaryProto(Env::Default(), path + "/frozen_model.pb", &graph_def);
		if (!status.ok()) {
			std::cout << status.ToString() << "\n";
			return 1;
		}

		model_version = 1;
		for(int i=0; i<graph_def.node_size(); i++)
		{
			if(graph_def.node(i).name() == "training_network/states")
			{
				auto shape = graph_def.node().Get(0).attr().at("shape").shape();
				dim1 = shape.dim(1).size();
			}
			if(graph_def.node(i).name() == "training_network/actions_pi")
			{
				model_version = 2;
			}
			if(graph_def.node(i).name() == "training_network/training")
			{
				model_version = 3;
			}
		}

		//cout << path << is_new_model << endl;
		// Add the graph to the session
		status = session->Create(graph_def);
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
                action_probs[i] = 1/NN; //dis(gen); 
			float p = dis(gen);
			int x = (int)(p*NN);
			action_probs[x] = p;
            action_probs_vector.push_back(action_probs);
            
            float value = std::tanh(10*(dis(gen)-0.5));
            values.push_back(value);
        }
        
        return true;
    }
    
    bool Forward(Tensor& states, Tensor& action_mask, std::vector<std::array<float, NN>>& action_probs_vector, std::vector<float>& values)
    {
        if(!model_loaded)
            return false;
        
		tensorflow::Tensor training(tensorflow::DT_BOOL, tensorflow::TensorShape());
		training.scalar<bool>()() = false;
        std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
            { "training_network/states:0", states },
        };

		if(model_version>=2)
		{
			inputs.push_back({ "training_network/actions_pi:0", action_mask});
		}
		if(model_version>=3)
		{
			inputs.push_back({ "training_network/training:0", training });
		}
            

        // The session will initialize the outputs
        std::vector<tensorflow::Tensor> outputs;

        // Run the session, evaluating our "c" operation from the graph
        Status  status;
		if(!warmed)
		{
			for(int i=0; i<100; i++)
				status = session->Run(inputs, {"training_network/policy_head/out_action_prob", "training_network/value_head/out_value"}, {}, &outputs);
			warmed = true;
		}
		else
			status = session->Run(inputs, {"training_network/policy_head/out_action_prob", "training_network/value_head/out_value"}, {}, &outputs);
        if (!status.ok()) {
            std::cout << status.ToString() << "\n";
            return false;
        }
        
		action_probs_vector.clear();
		values.clear();
        /*std::mt19937 gen((unsigned int)time(NULL));
        std::uniform_real_distribution<> dis(0.0, 1.0);
        for(int c=0; c<states.dim_size(0); c++)
        {
            std::array<float, NN> action_probs;
            for(int i=0; i<NN; i++)
                action_probs[i] = 1/NN;
			float p = dis(gen);
			int x = (int)(p*NN);
			action_probs[x] = p;
            action_probs_vector.push_back(action_probs);
            values.push_back(std::tanh(10*(dis(gen)-0.5)));
		}*/
        auto probs_mapped = outputs[0].tensor<float,2>();
        for(int c=0; c<states.dim_size(0); c++)
        {
            std::array<float, NN> action_probs;
			float sum=0;
            for(int i=0; i<NN; i++)
			{
                action_probs[i] = probs_mapped(c,i);
				//cout << action_probs[i] << " ";
				//assert(action_probs[i]>=0 && action_probs[i]<=1);
				sum+=action_probs[i];
			}

			/*if(abs(sum-1.0)>0.001 && sum > 0)
			{
		        //for(int i=0; i<NN; i++)
					//cout << action_probs[i] << " ";

				dump1(&states);
				dump2(&action_mask);
				assert(abs(sum-1.0)<=0.001);
			}*/
            action_probs_vector.push_back(action_probs);
            
            auto values_mapped = outputs[1].tensor<float, 2>();
            float value = values_mapped(c,0);
            values.push_back(value);

			/*if( abs(sum) < 0.000001)
			{
				cout << "vector size " << states.dim_size(0) << endl;
				cout << "tensorflow error. action probs sum is zero. sum = " << sum << endl;
				std::for_each(action_probs.begin(), action_probs.end(), [=](float x){
					cout << x << " ";
				});
				cout << endl;
				std::for_each(values.begin(), values.end(), [=](float x){
					cout << x << " ";
				});
				cout << endl;
				return false;

				//exit(0);
			}*/
        }
        return true;
    }
};

#endif
