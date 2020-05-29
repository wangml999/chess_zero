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
//#include "board.h"
//#include "fast_go.h"
#include "bit_board.h"
#include "chess_board.h"
#include "config.h"
#include <math.h>
#include <algorithm>
#include <cassert>
#include <numeric>
#include <random>
#include "network.h"
#include <tensorflow/cc/saved_model/loader.h>
#include <iomanip>
#include <thread>

using namespace std;
using namespace tensorflow;

extern double timing;
extern int ActionToInternalMove(int side, int n);
extern string InternalMoveToString(int m);

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

bool myfunction (int i,int j) { return (i<j); }
class TreeNode
{
private:
public:
    float total_value;
    float mean_value;
	float original_value;
    int id;
    Position pos;
    TreeNode *children;
	int children_num;
    int visits;
    int level;
    char player;
    float prob;
	//string debugMove;
    TreeNode* parent;
	BYTE	reps;
    
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
		children_num = 0;
		reps = 0;
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
    
    float puct_value(float total_visits_sqrt, float cpuct = CPUCT, float dir=0.0, float epsilon=0.0)
    {
        if(prob < 0.00001)
            return -std::numeric_limits<int>::max();
        else
		{
			float c = log((total_visits_sqrt*total_visits_sqrt + CPUCT_BASE + 1)/CPUCT_BASE) + cpuct;
			c *= total_visits_sqrt / (1+visits);
            return -mean_value + (c) * ((1-epsilon)*prob+epsilon*dir);
		}
    }
    
    int all_children_visits()
    {
        int all_visits = 0;
        std::for_each(children, children+children_num, [&] (TreeNode& node) {
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

	TreeNode* GetChild(int n)
	{
		for(int i=0; i<children_num; i++)
			if(children[i].id == n)
				return &(children[i]);
		return nullptr;
		/*int first = 0,                 //first array element
		last = children_num - 1,            //last array element
		middle;                        //mid point of search

		while (first <= last)
		{
		    middle = (first + last) / 2; //this finds the mid point
		    if (children[middle].id == n) {
		        return &(children[middle]);
		    }
		    else if (children[middle].id > n) // if it's in the lower half
		    {
		        last = middle - 1;
		    }
		    else {
		        first = middle + 1;                 //if it's in the upper half
		    }
		}*/
		//cout << n << " is not found" << endl;
		//for(int i=0; i<children_num; i++)
		//	cout << children[i].id << " ";
		//cout << endl;
		return nullptr;  // not found
	}
};

class Tree
{
private:
    std::mt19937 generator;
    dihedral_method dihedral;
    vector<TreeNode*> expand_buffer;
    bool multi_threading;
    std::mutex m;
    std::condition_variable producer_condition, consumer_condition;
    float resign_threshold;
    float resign_prob;
    std::gamma_distribution<float> gamma_dist;
    int mcts_reps;
    float cpuct;
public:
    vector<float> dirichlet_noise; 
    int mode;
    TreeNode* root;
    Network* pNetwork;
    atomic<int> spare;
    
    Tree(int m=EVALUATE, float re_th=-0.9, float re_pro=0.9, int reps=MCTS_REPS, float c=CPUCT) :
            dihedral(),
            generator(std::random_device{}()),
            mode(m),
            resign_threshold(re_th),
            resign_prob(re_pro),
            gamma_dist(ALPHA),
            //dirichlet_noise(NN+1),
            mcts_reps(reps),
	    	cpuct(c)
    {
        root = new TreeNode();
        pNetwork = nullptr;
		spare.store(0, memory_order_relaxed);
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
                for(int i=0; i<rootNode->children_num; i++)
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
            for(int i=0; i<newRoot->children_num; i++)
                newRoot->children[i].parent = newRoot;
        }
        
        delete_tree(root);
        delete root;
        root = newRoot;
    }
    
    int search(Board& board, vector<int> &actions, vector<float> &search_probs, float &value, std::vector<float>& child_values)
    {
        this->root->pos = board.position;
        //this->root->player = board.current;
        
        if(this->root->is_leaf())
        {
			this->root->reps = board.repetitions[board.position.ZobristHash()];
            float v = expand(this->root);
            this->root->update(1, v);
        }

		dirichlet_noise.resize(root->children_num);
        dirichlet(dirichlet_noise);
        expand_buffer.clear();
		auto timer_start = std::chrono::high_resolution_clock::now();
		int finish_reps = (this->mode==EVALUATE)?200000:this->mcts_reps;
        for(int i=0; i<finish_reps; i++)
        {
            simulate(board);
            if( expand_buffer.size() == TENSORFLOW_BATCH_SIZE || i==(finish_reps-1))
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
			if(mode == EVALUATE)
			{
				int sp = spare.load(memory_order::memory_order_acquire);
		
				if(sp==1) // if still waiting for human to input, continue to search
					continue;
				if(sp==2) // if human has entered move, stop spare time search
					break;
				auto now = std::chrono::high_resolution_clock::now();
				std::chrono::milliseconds ns = std::chrono::duration_cast<std::chrono::milliseconds>(now-timer_start);
				if(ns.count()>this->mcts_reps) // in milliseconds
					break;
			}
        }
		if(spare.load(memory_order::memory_order_acquire)==1)
		{
			cout << ":::" << flush;
			//while(spare.load(memory_order::memory_order_acquire)==1)
				//std::this_thread::yield();
		}
        std::bernoulli_distribution resign_distribution(resign_prob);

        //std::array<float, NN+1> values;
        for(int i=0; i<root->children_num; i++)
        {
            if(this->mcts_reps > 0)
                search_probs.push_back(root->children[i].visits);
            else
                search_probs.push_back(root->children[i].prob); //if it is root, use the base prob. ***this is wrong here. will fix later
            
			actions.push_back(root->children[i].id);
            child_values.push_back(-root->children[i].mean_value);
        }

        int n;
		float max_visit = *std::max_element(search_probs.begin(), search_probs.end());
		//if( max_visit < 1 )
		//	max_visit = 1;

		vector<float> decision_probs;
        for(int i=0; i<root->children_num; i++)
			decision_probs.push_back(search_probs[i] / (float)max_visit);

		float temperature = 0.01;
        if(mode == SELF_PLAY && root->level < 30)  //first 10% or first 4 steps are not deterministic to create more randomness
			temperature = 1.0;
		else
			temperature = 0.01;

        float sum = 0.0;
        for(int i=0; i<root->children_num; i++)
        {
			decision_probs[i] = pow(decision_probs[i], 1.0/temperature);
			sum = sum + decision_probs[i];
    	}
        for(int i=0; i<root->children_num; i++)
			decision_probs[i] = decision_probs[i] / sum;

        std::discrete_distribution<int> distribution(decision_probs.begin(), decision_probs.end());
        n = distribution(generator);
        n = root->children[n].id;

        sum = 0.0;
        for(int i=0; i<root->children_num; i++)
        {
			sum = sum + search_probs[i];
    	}
        for(int i=0; i<root->children_num; i++)
        {
			search_probs[i] = search_probs[i] / sum;
    	}

        value = root->mean_value;

		if (root->mean_value < resign_threshold && root->level > NN)  // disable resign prior to NN steps
        {
        	if(resign_distribution(generator))
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
            int index = this->select(current_node);
			int action = current_node->children[index].id;

            if(action==NN)
                local_board.pass_count++;
            else
                local_board.pass_count = 0;
            
			if(current_node != root)
				local_board.repetitions[current_node->pos.ZobristHash()]++;
            local_board.steps++;
            
            current_node = &(current_node->children[index]);
            current_node->update(1+VIRTUAL_LOSS, VIRTUAL_LOSS);
        }
        //cout << endl;

        if(current_node->parent != nullptr)
        {
            local_board.position = current_node->parent->pos;
            if(current_node->id != NN)
			{
                local_board.position = local_board.position.play_move(current_node->id, current_node->parent->player);
			}
			local_board.position.side=(local_board.position.side+1)%2;
            local_board.current =  local_board.position.swap_colors(current_node->parent->player);
			local_board.repetitions[local_board.position.ZobristHash()]++;
        }
        
		if(local_board.position.bitboards[KING|local_board.position.side] == 0)
			assert(local_board.position.bitboards[KING|local_board.position.side] != 0);
        int status = local_board.status();
        
        float v = 0.0;
        if(status == -1)
        {
            current_node->pos = local_board.position;
			current_node->reps = local_board.repetitions[local_board.position.ZobristHash()];
            expand_buffer.push_back(current_node);
        }
        else
        {
            /*either player actioned will result a win and -1 probability to the next player
            e.g. after player 1 played, player 1 wins and currennt player becomes 2, 2's winning prob is -1
            or after player 2 played, player 2 wins and currennt player becomes 1, 1's winning prob is -1 too*/
            if(status == 0 || status == 1)
                if(status == local_board.current)
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
			if(mode == EVALUATE)	
            	methods.push_back(0); //distribution(generator));  // we need some randomness in evaluation
			else
	            methods.push_back(0); //distribution(generator));  decided not to transform board in play mode. instead this should be done in train to avoid bias. 
	Tensor* p_states = nullptr;
	Tensor* p_action_masks = nullptr;

#ifdef TENSORFLOW_BENCHMARK        
	//auto game_start1 = std::chrono::high_resolution_clock::now();
	//for(int i=0; i<MCTS_REPS/TENSORFLOW_BATCH_SIZE; i++)
#endif
        p_states = MakeTensor(leaves, methods);
		p_action_masks = MakeActionMasksTensor(leaves);

#ifdef TENSORFLOW_BENCHMARK        
    	//auto game_end1 = std::chrono::high_resolution_clock::now();
    	//auto diff1 = game_end1-game_start1;
    	//std::chrono::nanoseconds game_ns1 = std::chrono::duration_cast<std::chrono::nanoseconds>(diff1);
    	//std::cout << MCTS_REPS/TENSORFLOW_BATCH_SIZE << "x maketensor time: " << game_ns1.count()*1.0/1000000000 << " seconds" << std::endl;
#endif
        std::vector<std::array<float, NN>> tmpprob_vector;

#ifdef TENSORFLOW_BENCHMARK        
	auto game_start2 = std::chrono::high_resolution_clock::now();
	for(int i=0; i<100; i++)
#endif
//	auto game_start2 = std::chrono::high_resolution_clock::now();
        assert(pNetwork->Forward(*p_states, *p_action_masks, tmpprob_vector, values));
//	auto game_end2 = std::chrono::high_resolution_clock::now();
//	auto diff2 = game_end2-game_start2;
//	std::chrono::nanoseconds game_ns2 = std::chrono::duration_cast<std::chrono::nanoseconds>(diff2);
//	timing += game_ns2.count()*1.0/1000000000;		
#ifdef TENSORFLOW_BENCHMARK
    	auto game_end2 = std::chrono::high_resolution_clock::now();
    	auto diff2 = game_end2-game_start2;
    	std::chrono::nanoseconds game_ns2 = std::chrono::duration_cast<std::chrono::nanoseconds>(diff2);
    	std::cout << MCTS_REPS/TENSORFLOW_BATCH_SIZE << "x forward time: " << game_ns2.count()*1.0/1000000000 << " seconds" << std::endl;
#endif
        //pNetwork->Forward_Simulator(*p_states, tmpprob_vector, values);
	delete p_states;
	delete p_action_masks;
     
        for(int node_id=0; node_id<leaves.size(); node_id++)
        {
            TreeNode* leaf = leaves[node_id];
            if(leaf->children == nullptr)
            {
                std::array<float, NN> prob;
                float sum_of_probs = 0.0;

               /* std::for_each(tmpprob_vector[node_id].begin(), tmpprob_vector[node_id].end(), [&] (float v) {
                    sum_of_probs += v;
                });
            

                std::for_each(tmpprob_vector[node_id].begin(), tmpprob_vector[node_id].end(), [=] (float& v) {
                    v = v / sum_of_probs;
                });*/

				/*int method = methods[node_id];
                if (method != 0)
                {
                    for(int i=0; i<NN; i++)
                        prob[i] = tmpprob_vector[node_id][dihedral.data[method][i]];
                    prob[NN] = tmpprob_vector[node_id][NN];
                }
                else*/
                    prob = tmpprob_vector[node_id];


                /*for(int i=0; i<NN; i++)  // test code. clip any prob smaller than 0.01 to give chance being selected in MCTS.
					if( prob[i] < 0.01 )
						prob[i] = 0.01;*/

				if(leaf->player != leaf->pos.side)
				{
					assert(leaf->player == leaf->pos.side);
				}
                std::vector<int> moves = leaf->pos.legal_actions(leaf->player);


                sum_of_probs = 0.0;
                std::for_each(moves.begin(), moves.end(), [&](int x){
					sum_of_probs += exp(prob[x]);
				});

                std::for_each(moves.begin(), moves.end(), [&](int x){
					prob[x] = exp(prob[x]) / sum_of_probs;
				});



				/*std::array<int, NN> action_mask;
				std::fill(action_mask.begin(), action_mask.end(), 0);

                std::for_each(moves.begin(), moves.end(), [&](int x){
                    //prob[x] = -prob[x];
					action_mask[x] = 1;
                });

				for(int i=0; i<NN; i++)
				{
					if(prob[i] == 0 && action_mask[i] != 0)
					{
						cout << "found zero action prob" << endl;
					}
					prob[i] = prob[i] * action_mask[i];
				}

				prob[NN]=0.0;

                //normalize prior probilities
                sum_of_probs = 0.0;

                std::for_each(prob.begin(), prob.end(), [&] (float v) {
					if(v!=0)
                    	sum_of_probs += exp(v);
                });
            
                std::for_each(prob.begin(), prob.end(), [=] (float& v) {
					if(v!=0)
                		v = exp(v) / sum_of_probs;
                });*/

                TreeNode *tmp = new TreeNode[moves.size()];
                char next_player = leaf->pos.swap_colors(leaf->player);
                for(int i=0; i<moves.size(); i++)
                {
                    tmp[i].id = moves[i];
                    tmp[i].parent = leaf;
                    tmp[i].prob = prob[moves[i]];
                    tmp[i].player = next_player;
                    tmp[i].level = leaf->level + 1;
					//tmp[i].debugMove = InternalMoveToString(ActionToInternalMove(leaf->player, moves[i]));
                }
                leaf->children = tmp;
				leaf->children_num = moves.size();
				leaf->original_value = values[node_id];
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
        vector<float> uct(parent->children_num, 0.0f);

#ifdef ADD_DIRICHLET_NOISE
        if (parent->parent == nullptr && mode == SELF_PLAY)
        {
            for(int i=0; i<parent->children_num; i++)
                uct[i] = parent->children[i].puct_value(total_visits_sqrt=total_visits_sqrt, cpuct, dirichlet_noise[i], 0.25);
        }
        else
#endif
        {
            for(int i=0; i<parent->children_num; i++)
                uct[i] = parent->children[i].puct_value(total_visits_sqrt=total_visits_sqrt, cpuct);
        }

        int action = (int)std::distance(uct.begin(), std::max_element(uct.begin(), uct.end()));
        //assert(parent->children[action].prob > 0.0001);
        /*if(action==NN)
        {
            std::cout<<"action pass"<<std::endl;
        }*/

        return action;
    }
    
	Tensor* MakeActionMasksTensor(vector<TreeNode*>& nodes)
	{
        int batch_size = nodes.size();
        Tensor* p_tensor = new Tensor(DT_FLOAT, TensorShape({TENSORFLOW_BATCH_SIZE, 73*64}));
		
        float* ptensor_data = p_tensor->flat<float>().data();
		memset(ptensor_data, 0, sizeof(float) * TENSORFLOW_BATCH_SIZE * 73 * 64);
        for(int b=0; b<batch_size; b++)
        {
			TreeNode *p = nodes[b];
            std::vector<int> moves = p->pos.legal_actions(p->player); 
			assert(moves.size()!=0);
            std::for_each(moves.begin(), moves.end(), [&](int x){
                ptensor_data[b*73*64+x] = 1;
            });
		}
		return p_tensor;
	}

    Tensor* MakeTensor(vector<TreeNode*>& nodes, vector<int>& methods)
    {
        assert(nodes.size() > 0);
        int batch_size = nodes.size();
		int slice_num = (pNetwork->dim1-7)/14-1;
			
        Tensor* p_states = new Tensor(DT_FLOAT, TensorShape({TENSORFLOW_BATCH_SIZE, (slice_num+1)*14+7, 8, 8}));
		
        float* ptensor_data = p_states->flat<float>().data();
		memset(ptensor_data, 0, sizeof(float) * (TENSORFLOW_BATCH_SIZE * ((slice_num+1)*14+7) * 8 * 8));
        
        //std::thread t[TENSORFLOW_BATCH_SIZE];
        
        for(int b=0; b<batch_size; b++)
        {
		    int method = methods[b];	
            //t[b] = std::thread([=](){
                /*thread_local*/ 
			TreeNode *p = nodes[b];
            char current = p->pos.side;
            char enermy = p->pos.swap_colors(current);
			assert(p->reps>=0 && p->reps<3);
            
            float* pslicehead = ptensor_data + b * ((slice_num+1)*14+7)*64;
			float* pp = pslicehead;
            for(int slice=0; slice<slice_num+1; slice++)
            {
                std::string state;

                if( p!=nullptr )
                {
					BYTE work_board_array[64];
					if(current==WHITE)
					{
						for(int i=0; i<64; i++)
							work_board_array[i] = p->pos.board_array[i];
					}
					else
					{
						for(int i=0; i<64; i++)
							work_board_array[63-i] = p->pos.board_array[i];
					}

					for(int i=0; i<64; i++)
					{
						if(work_board_array[i]==EMPTY)
							continue;
						assert(work_board_array[i]>=PAWN);
						int piece=work_board_array[i] & 0xfe;
						int color = work_board_array[i] & 0x1;
						int is_enermy = (color==current)?0:1;
						int k = ((piece>>1)-1)+is_enermy*6;

						pp[k*64+i] = 1;
					}
					pp+=(12*64);
					//slide 12th and 13th are repetitions. same for both blackwhite and 

					std::fill(pp, pp+64, (p->reps-1)&0b01);  // repetition. 2 planes. 0, 1, 2 stands for 1, 2, 3
					pp+=64;
					std::fill(pp, pp+64, ((p->reps-1)&0b10)>>1);
					pp+=64;
					assert(pp-pslicehead == (slice+1)*14*64);
                }
                else
                {
                    /*for(int i=0; i<NN; i++)
                    {
                        *pdata1++ = 0;
                        *pdata2++ = 0;
                    }*/
					//memset(pp, 0, 14*64);
					pp+=(14*64);
                }
                
                if(p != root && p != nullptr)
                    p = p->parent;
                else
                    p = nullptr;
            }
			
			std::fill(pp, pp+64, current);  // current side
			pp+=64;
			std::fill(pp, pp+64, nodes[b]->level); // total move count
			pp+=64;
			std::fill(pp, pp+64, (nodes[b]->pos.castling_rights>>(current*2))&0x1);		// p1 king castling
			pp+=64;
			std::fill(pp, pp+64, (nodes[b]->pos.castling_rights>>(current*2+1))&0x1);	// p1 queen castling
			pp+=64;
			std::fill(pp, pp+64, (nodes[b]->pos.castling_rights>>(enermy*2))&0x1);		// p2 king castling
			pp+=64;
			std::fill(pp, pp+64, (nodes[b]->pos.castling_rights>>(enermy*2+1))&0x1);	// p2 queen castling
			pp+=64;
			std::fill(pp, pp+64, nodes[b]->pos.fifty_moves);
			pp+=64;
            //});
        }
        
        //for(int b=0; b<batch_size; b++)
            //t[b].join();
        return p_states;
    }
    
};

#endif
