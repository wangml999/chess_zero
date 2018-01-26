//
//  tree.h
//  t1cpp
//
//  Created by Minglei Wang on 9/01/2018.
//  Copyright © 2018 Minglei Wang. All rights reserved.
//
#ifndef tree_h
#define tree_h

#include <stdio.h>

//#include <boost/range/combine.hpp>

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

using namespace std;

#define CPUCT 0.2

template <typename T> class dihedral_method
{
public:
    dihedral_method()
    {
    }

    std::vector<T> dihedral(const std::vector<T>& t1, int i)
    {
        std::vector<T> t2 = t1;
        return t2;
    }
    
    std::array<T, NN> inverse_dihedral(const std::array<T, NN>& t1, int i)
    {
        std::array<T, NN> t2 = t1;
        return t2;
    }
};


class TreeNode
{
private:
public:
    double total_value;
    double mean_value;
    int id;
    Position pos;
    TreeNode *children;
    int visits;
    int level;
    double prob;
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
    
    double puct_value(double total_visits_sqrt, double dir=0.0, double epsilon=0.0)
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
    
    void update(double v)
    {
        visits += 1;
        total_value += v;
        mean_value = total_value / visits;
    }
};

class Tree
{
private:
    double TEMPERATURE;
    
public:
    TreeNode* root;
    
    Tree()
    {
        root = new TreeNode();
        TEMPERATURE = 1.0;
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
    
    int search(Board& board, double &value, int rep = 0)
    {
        this->root->pos = board.position;
        
        for(int i=0; i<rep; i++)
            simulate(board);
        
        if (root->mean_value < -0.8)
        {
            value = -root->children[NN].mean_value;
            return NN;
        }

        array<double, NN+1> v;
        double sum = 0.0;
        for(int i=0; i<NN+1; i++)
        {
            double t = pow (root->children[i].visits, 1.0/TEMPERATURE);
            v[i] = t;
            sum += t;
        }

        for_each(v.begin(), v.end(), [&](double item) {
            return item/sum;
        });
        
        int n = (int)distance(v.begin(), max_element(v.begin(), v.end()));
        value = -root->children[n].mean_value;
        return n;
    }
    
    void simulate(Board& board)
    {
        Board local_board = board;

        TreeNode* current_node = root;
        //current_node->visits++;
        while(!current_node->is_leaf())
        {
            int action = this->select(current_node);
            
            Step s;
            
            s.state = current_node->pos.get_board();
            s.player = '0' + (current_node->level % 2 + 1);
            s.action = current_node->id;
            s.value = 0.0;
            
            local_board.history.push_back(s);
            local_board.steps++;
            
            current_node = &(current_node->children[action]);
            //current_node->visits++;
        }

        if(current_node->parent != nullptr)
        {
            local_board.position = current_node->parent->pos;
            local_board.current = BLACK+(current_node->parent->level)%2;
            
            //since the above code did not actually action on the board, we need to take one step back
            local_board.steps--;
            local_board.action(current_node->id);
        }
        
        int status = local_board.status();
        
        double v = 0.0;
        if(status == -1)
            v = expand(current_node, local_board);
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
        }
        /*
        
            update on the root node is nonsense as it is not standing for an edge
            the root visit count is 1 plus all the edge visits
        */
        while (current_node != nullptr)
        {
            current_node->update(v);
            current_node = current_node->parent;
            v = -v;
        }
    }
    
    double expand(TreeNode* leaf, Board& board)
    {
        /*if(leaf->parent != nullptr)
        {
            board.position = leaf->parent->pos;
            board.current = BLACK+(leaf->parent->level)%2;
            board.action(leaf->id);
            leaf->pos = board.position;
        }*/
        leaf->pos = board.position;
        

        std::string s;
        dihedral_method<char> d1;
        
        int method = -1;
        std::string state = leaf->pos.get_board();
        if (board.symmetrical)
        {
            method = rand() % 8;
            vector<char> v = d1.dihedral(std::vector<char>(state.begin(), state.end()), method);
            s = string(v.begin(), v.end());
        }
        else
            s = state;
        
        //get image slices
        
        //evaluate in neural network
        std::array<double, NN+1> prob;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(1.0, 2.0);
        for(int i=0; i<prob.size(); i++)
            prob[i] = dis(gen);
        double sum_of_probs = 0.0;
        std::for_each(prob.begin(), prob.end(), [&] (double v) {
            sum_of_probs += v;
        });
        
        std::for_each(prob.begin(), prob.end(), [=] (double& v) {
            v = v / sum_of_probs;
        });

        double value = std::tanh(10*(dis(gen)-1.5));
        
        // end of evaluation
        
        if (method != -1)
        {
            dihedral_method<double> d2;
            //prob = d2.inverse_dihedral(prob, method);
        }
        else
            prob = prob;

        std::vector<int> moves = board.not_allowed_actions();
        std::for_each(moves.begin(), moves.end(), [&](int x){
            prob[x] = 0.0;
        });

        //normalize prior probilities
        sum_of_probs = 0.0;
        std::for_each(prob.begin(), prob.end(), [&] (double v) {
            sum_of_probs += v;
        });
        
        std::for_each(prob.begin(), prob.end(), [=] (double& v) {
            v = v / sum_of_probs;
        });

        leaf->children = new TreeNode[NN+1];
        for(int i=0; i<NN+1; i++)
        {
            leaf->children[i].id = i;
            leaf->children[i].parent = leaf;
            leaf->children[i].prob = prob[i];
            leaf->children[i].level = leaf->level + 1;
        }

        return value;
    }

    array<double, NN+1> dirichlet(double alpha)
    {
        std::array<double, NN+1> v;
        
        /*std::generate(v.begin(), v.end(), [](){
            return (double) rand() / RAND_MAX;
        });
        
        double sum = 0.0;
        std::for_each(v.begin(), v.end(), [&] (double v) {
            sum += v;
        });
        
        std::for_each(v.begin(), v.end(), [=](double& el){
            el = el / sum;
        });*/
        return v;
    }
    
    int select(TreeNode* parent)
    {
        double total_visits_sqrt = sqrt(parent->visits);
        array<double, NN+1> uct;
        if (parent->parent == nullptr)
        {
            array<double, NN+1> dir = dirichlet(0.03);
            for(int i=0; i<NN+1; i++)
                uct[i] = parent->children[i].puct_value(total_visits_sqrt=total_visits_sqrt, dir[i], 0.25);
        }
        else
        {
            for(int i=0; i<NN+1; i++)
                uct[i] = parent->children[i].puct_value(total_visits_sqrt=total_visits_sqrt);
        }

        int action = (int)std::distance(uct.begin(), std::max_element(uct.begin(), uct.end()));
        /*if(action==NN)
        {
            std::cout<<"action pass"<<std::endl;
        }*/
        
        return action;
    }
};

#endif
