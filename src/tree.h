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

#ifndef tree_h
#define tree_h

#include <stdio.h>
#include <string>
#include <vector>
#include <array>
#include <limits>
#include "board.h"
#include "fast_go.h"
#include <math.h>
#include <algorithm>
#include <cassert>
#include <numeric>
#include <random>
#include "fast_go.h"
#include "network.h"
#include <tensorflow/cc/saved_model/loader.h>
#include <iomanip>
#include <thread>

using namespace std;
using namespace tensorflow;

class dihedral_method
{
public:
    array<array<int, NN>, 8>  data;
    dihedral_method()
    {
        for(int i=0; i<NN; i++)
            data[0][i] = i;
        
        for(int j=0; j<data.size(); j+=2)
        {
            transpose(data[j], data[j+1]);
            
            if(j<6)
                reverse_row(data[j+1], data[j+2]);
        }
    }
    
    void reverse_row(array<int, NN>& input, array<int, NN>& output)
    {
        for(int row=0; row<WN; row++)
            for(int col=0; col<WN; col++)
            {
                output[row*WN+WN-col-1] = input[row*WN+col];
            }
    }

    void transpose(array<int, NN>& input, array<int, NN>& output)
    {
        for(int row=0; row<WN; row++)
            for(int col=0; col<WN; col++)
            {
                output[col*WN+row] = input[row*WN+col];
            }
    }
};


class TreeNode
{
private:
public:
    float total_value;
    float mean_value;
    int id;
    Position pos;
    TreeNode *children;
    int visits;
    int level;
    char player;
    float prob;
    TreeNode* parent;
    
    TreeNode()
    {
        id = -1;
        
        visits = 0;
        total_value = 0.0;
        mean_value = 0.0;
        level = 0;
        prob = 0.0;
        parent = nullptr;
        children = nullptr;
    }
    
    TreeNode(TreeNode* parent) : TreeNode()
    {
        if(parent == nullptr)
            level = 0;
        else
            level = parent->level + 1;
        this->parent = parent;
    }

    ~TreeNode()
    {
        if(children != nullptr)
        {
            delete[] children;
            children = nullptr;
        }
    }
    
    float puct_value(float total_visits_sqrt, float dir=0.0, float epsilon=0.0)
    {
        if(prob < 0.00001)
            return -std::numeric_limits<int>::max();
        else
            return -mean_value + CPUCT * ((1-epsilon)*prob+epsilon*dir) * total_visits_sqrt / (1+visits);
    }
    
    int all_children_visits()
    {
        int all_visits = 0;
        std::for_each(children, children+NN+1, [&] (TreeNode& node) {
            all_visits += node.visits;
        });
        
        return all_visits;
    }

    bool is_leaf()
    {
        return children==nullptr;
    }
    
    void update(int add_visits, float add_value)
    {
        visits = visits + add_visits;
        assert(visits>0);
        total_value = total_value + add_value;
        mean_value = total_value / visits;
    }
};

class Tree
{
private:
    float TEMPERATURE;
    std::mt19937 generator;
    dihedral_method dihedral;
    vector<TreeNode*> expand_buffer;
    bool multi_threading;
    std::mutex m;
    std::condition_variable producer_condition, consumer_condition;
    float resign_threshold;
    float resign_prob;
    std::gamma_distribution<float> gamma_dist;
    vector<float> dirichlet_noise; 
    int mcts_reps;
    //atomic<int> count;
public:
    TreeNode* root;
    Network* pNetwork;
    
    Tree(float temp=1.0, float re_th=-0.9, float re_pro=0.9, int reps=MCTS_REPS) :
            dihedral(),
            generator(std::random_device{}()),
            TEMPERATURE(temp),
            resign_threshold(re_th),
            resign_prob(re_pro),
            gamma_dist(ALPHA),
            dirichlet_noise(NN+1),
            mcts_reps(reps)
    {
        root = new TreeNode();
        pNetwork = nullptr;
    }

    ~Tree()
    {
        delete_tree(root);
        delete root;
    }
    
    void delete_tree(TreeNode* rootNode)
    {
        if(rootNode != nullptr)
        {
            if(rootNode->children != nullptr)
            {
                for(int i=0; i<NN+1; i++)
                {
                    delete_tree(&rootNode->children[i]);
                }
                
                delete[] rootNode->children;
                rootNode->children = nullptr;
            }
        }
    }
    
    void change_root(TreeNode& node)
    {
        TreeNode* newRoot = new TreeNode();
        *(newRoot) = node;
        node.children = nullptr;
        newRoot->parent = nullptr;
        if(newRoot->children != nullptr)
        {
            for(int i=0; i<NN+1; i++)
                newRoot->children[i].parent = newRoot;
        }
        
        delete_tree(root);
        delete root;
        root = newRoot;
    }
    
    int search(Board& board, array<float, NN+1> &search_probs, float &value)
    {
        this->root->pos = board.position;
        this->root->player = board.current;
        
        if(this->root->is_leaf())
        {
            float v = expand(this->root);
            this->root->update(1, v);
        }

        dirichlet(dirichlet_noise);
        expand_buffer.clear();
        for(int i=0; i<this->mcts_reps; i++)
        {
            simulate(board);
            if( expand_buffer.size() == TENSORFLOW_BATCH_SIZE || i==(this->mcts_reps-1))
            {
                if( expand_buffer.size() > 0)
                {
                    vector<float> values;
                    expand(expand_buffer, values);
                    for(int j=0; j<expand_buffer.size(); j++)
                    {
                        backup_tree(expand_buffer[j], values[j]);
                    }
                    expand_buffer.clear();
                }
            }
        }
        
        std::bernoulli_distribution distribution(resign_prob);

        float sum = 0.0;
        std::array<float, NN+1> values;
        for(int i=0; i<NN+1; i++)
        {
            if(this->mcts_reps > 0)
                search_probs[i] = root->children[i].visits;
            else
                search_probs[i] = root->children[i].prob;
            
            values[i] = -root->children[i].mean_value;
            sum += search_probs[i];
        }
        for(int i=0; i<NN+1; i++)
        {
            search_probs[i] = search_probs[i] / sum;
        }

        int n;
        if(TEMPERATURE > 0.1 && root->level < int(0.1*NN))
        {
            std::discrete_distribution<int> distribution(search_probs.begin(), search_probs.end());
            n = distribution(generator);
        }
        else
        {
            n = (int)std::distance(search_probs.begin(), std::max_element(search_probs.begin(), search_probs.end()));
        }
        
        value = -root->children[n].mean_value;

        if(distribution(generator))
        {
            if (root->mean_value < resign_threshold)
            {
                n = NN;
            }
        }
        
        return n;
    }
    
    void simulate(Board local_board)
    {
        TreeNode* current_node = root;
        current_node->update(1+VIRTUAL_LOSS, VIRTUAL_LOSS);
        while(!current_node->is_leaf())
        {
            int action = this->select(current_node);

            if(action==NN)
                local_board.pass_count++;
            else
                local_board.pass_count = 0;
            
            local_board.steps++;
            
            current_node = &(current_node->children[action]);
            current_node->update(1+VIRTUAL_LOSS, VIRTUAL_LOSS);
        }
        //cout << endl;

        if(current_node->parent != nullptr)
        {
            local_board.position = current_node->parent->pos;
            if(current_node->id != NN)
                local_board.position = local_board.position.play_move(current_node->id, current_node->parent->player);
            local_board.current =  local_board.position.swap_colors(current_node->parent->player);
        }
        
        int status = local_board.status();
        
        float v = 0.0;
        if(status == -1)
        {
            current_node->pos = local_board.position;
            expand_buffer.push_back(current_node);
        }
        else
        {
            /*either player actioned will result a win and -1 probability to the next player
            e.g. after player 1 played, player 1 wins and currennt player becomes 2, 2's winning prob is -1
            or after player 2 played, player 2 wins and currennt player becomes 1, 1's winning prob is -1 too*/
            if(status != 0)
                if(status == (int)(local_board.current - '0'))
                    v = 1.0;   //the winner is player to play
                else
                    v = -1.0;  //the winner is the player just played.
            else
                v = 0.0;
            backup_tree(current_node, v);
        }
    }
    
    /*
        update on the root node is nonsense as it is not standing for an edge
        the root visit count is 1 plus all the edge visits
    */
    void backup_tree(TreeNode *node, float value)
    {
        while (node != nullptr)
        {
            node->update(0-VIRTUAL_LOSS, value-VIRTUAL_LOSS);
            node = node->parent;
            value = -value;
        }
    }
    
    string transform_state(string& state, int method)
    {
        array<char, NN> newstate;
        for(int i=0; i<NN; i++)
            newstate[i] = state[dihedral.data[method][i]];
        return string(newstate.begin(), newstate.end());
    }
    
    float expand(TreeNode* leaf)
    {
        vector<TreeNode*> nodes;
        vector<float> values;
        nodes.push_back(leaf);
        expand(nodes, values);
        
        return values[0];
    }
    
    void expand(vector<TreeNode*>& leaves, vector<float>& values)
    {
	vector<int> methods;
	std::uniform_int_distribution<int> distribution(0,7);
        for(int node_id=0; node_id<leaves.size(); node_id++)
            methods.push_back(distribution(generator));
	Tensor* p_states = nullptr;

#ifdef TENSORFLOW_BENCHMARK        
	auto game_start1 = std::chrono::high_resolution_clock::now();
	for(int i=0; i<MCTS_REPS/TENSORFLOW_BATCH_SIZE; i++)
#endif
            p_states = MakeTensor(leaves, methods);

#ifdef TENSORFLOW_BENCHMARK        
    	auto game_end1 = std::chrono::high_resolution_clock::now();
    	auto diff1 = game_end1-game_start1;
    	std::chrono::nanoseconds game_ns1 = std::chrono::duration_cast<std::chrono::nanoseconds>(diff1);
    	std::cout << MCTS_REPS/TENSORFLOW_BATCH_SIZE << "x maketensor time: " << game_ns1.count()*1.0/1000000000 << " seconds" << std::endl;
#endif
        std::vector<std::array<float, NN+1>> tmpprob_vector;

#ifdef TENSORFLOW_BENCHMARK        
	auto game_start2 = std::chrono::high_resolution_clock::now();
	for(int i=0; i<MCTS_REPS/TENSORFLOW_BATCH_SIZE; i++)
#endif
            pNetwork->Forward(*p_states, tmpprob_vector, values);
#ifdef TENSORFLOW_BENCHMARK
    	auto game_end2 = std::chrono::high_resolution_clock::now();
    	auto diff2 = game_end2-game_start2;
    	std::chrono::nanoseconds game_ns2 = std::chrono::duration_cast<std::chrono::nanoseconds>(diff2);
    	std::cout << MCTS_REPS/TENSORFLOW_BATCH_SIZE << "x forward time: " << game_ns2.count()*1.0/1000000000 << " seconds" << std::endl;
#endif
        //pNetwork->Forward_Simulator(states, tmpprob_vector, values);
	delete p_states;
     
        for(int node_id=0; node_id<leaves.size(); node_id++)
        {
            TreeNode* leaf = leaves[node_id];
            if(leaf->children == nullptr)
            {
                std::array<float, NN+1> prob;
                float sum_of_probs = 0.0;
                std::for_each(tmpprob_vector[node_id].begin(), tmpprob_vector[node_id].end(), [&] (float v) {
                    sum_of_probs += v;
                });
            
                std::for_each(tmpprob_vector[node_id].begin(), tmpprob_vector[node_id].end(), [=] (float& v) {
                    v = v / sum_of_probs;
                });

		int method = methods[node_id];
                if (method != 0)
                {
                    for(int i=0; i<NN; i++)
                        prob[i] = tmpprob_vector[node_id][dihedral.data[method][i]];
                    prob[NN] = tmpprob_vector[node_id][NN];
                }
                else
                    prob = tmpprob_vector[node_id];

                std::vector<int> moves = leaf->pos.not_allowed_actions(leaf->player);
            
                std::for_each(moves.begin(), moves.end(), [&](int x){
                    prob[x] = 0.0;
                });

                //normalize prior probilities
                sum_of_probs = 0.0;
                std::for_each(prob.begin(), prob.end(), [&] (float v) {
                    sum_of_probs += v;
                });
            
                std::for_each(prob.begin(), prob.end(), [=] (float& v) {
                    v = v / sum_of_probs;
                });

                TreeNode *tmp = new TreeNode[NN+1];
                char next_player = leaf->pos.swap_colors(leaf->player);
                for(int i=0; i<NN+1; i++)
                {
                    tmp[i].id = i;
                    tmp[i].parent = leaf;
                    tmp[i].prob = prob[i];
                    tmp[i].player = next_player;
                    tmp[i].level = leaf->level + 1;
                }
                leaf->children = tmp;
            }
        }
    }

    void dirichlet(std::vector<float> &result)
    {
        float sum = 0.0;
        for(int i=0; i<result.size(); i++)
        {
            result[i] = gamma_dist(generator);
            sum += result[i];
        }
        
        for(int i=0; i<result.size(); i++)
        {
            result[i] = result[i] / sum;
        }
    }

    int select(TreeNode* parent)
    {
        assert(parent->children != nullptr);
        
        float total_visits_sqrt = sqrt(parent->visits);
        array<float, NN+1> uct;
#ifdef ADD_DIRICHLET_NOISE
        if (parent->parent == nullptr && TEMPERATURE > 0.1)
        {
            for(int i=0; i<NN+1; i++)
                uct[i] = parent->children[i].puct_value(total_visits_sqrt=total_visits_sqrt, dirichlet_noise[i], 0.25);
        }
        else
#endif
        {
            for(int i=0; i<NN+1; i++)
                uct[i] = parent->children[i].puct_value(total_visits_sqrt=total_visits_sqrt);
        }

        int action = (int)std::distance(uct.begin(), std::max_element(uct.begin(), uct.end()));
        //assert(parent->children[action].prob > 0.0001);
        /*if(action==NN)
        {
            std::cout<<"action pass"<<std::endl;
        }*/
        
        return action;
    }
    
    Tensor* MakeTensor(vector<TreeNode*>& nodes, vector<int>& methods)
    {
        assert(nodes.size() > 0);
        int batch_size = nodes.size();
        Tensor* p_states = new Tensor(DT_FLOAT, TensorShape({batch_size, (SLICES+1)*2+1, WN, WN}));
        float* ptensor_data = p_states->flat<float>().data();
        
        //std::thread t[TENSORFLOW_BATCH_SIZE];
        
        for(int b=0; b<batch_size; b++)
        {
	    int method = methods[b];	
            //t[b] = std::thread([=](){
                /*thread_local*/ TreeNode *p = nodes[b];
                char current = p->player;
                char opponent = p->pos.swap_colors(current);
                
                for(int slice=0; slice<(SLICES+1)*2; slice+=2)
                {
                    std::string state;
                    
                    float* pdata1 = ptensor_data + b * ((SLICES+1)*2+1) * NN + slice * NN;
                    float* pdata2 = ptensor_data + b * ((SLICES+1)*2+1) * NN + (slice+1) * NN;
                    if( p!=nullptr )
                    {
                        state = p->pos.get_board();
                        for(int i=0; i<NN; i++)
                        {
                            char c = state[dihedral.data[method][i]];
                            *pdata1++ = (c == current);
                            *pdata2++ = (c == opponent);
                        }
                    }
                    else
                    {
                        for(int i=0; i<NN; i++)
                        {
                            *pdata1++ = 0;
                            *pdata2++ = 0;
                        }
                    }
                    
                    if(p != root && p != nullptr)
                        p = p->parent;
                    else
                        p = nullptr;
                }
                
                //last slice is the color of current player. 1 black or 0 white
                float* pdata = ptensor_data + b * ((SLICES+1)*2+1) * NN + (SLICES+1)*2 * NN;
                for(int i=0; i<NN; i++)
                {
                    *pdata++ = (current == BLACK);
                }
            //});
        }
        
        //for(int b=0; b<batch_size; b++)
            //t[b].join();
        return p_states;
    }
    
};

#endif
