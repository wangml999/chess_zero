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

#include "fast_go.h"
#include "tree.h"
#include <cassert>
#include <thread>

#include "tensorflow/core/public/session.h"
//#include "tensorflow/core/platform/env.h"
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/cc/saved_model/loader.h>
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "network.h"

#include <random>
#include <chrono>
#include <fstream>
#include <iomanip>

#include <dirent.h>
#include <stdio.h>


int fast_go_test()
{
//    Position p = Position("1111111111111111111101101", NONE);
//    std::vector<int> a = p.possible_moves(WHITE);
//    std::vector<int> b = p.not_allowed_actions(WHITE);
//    //int score = p.score();
//    return 0;

    for(int rep=0; rep<100; rep++)
    {
        char color = BLACK;
        int previous_move = -1;
        
        Position p = Position::initial_state();
        for(int step=0; step<NN; step++)
        {
            std::vector<int> moves = p.possible_moves(color);
            int m = NN;
            if (moves.size() > 0)
            {
                int x = rand() % moves.size();
                m = moves[x];
            }
            
            if(m == NN && previous_move == NN)
                break;
            
            if(m != NN)
            {
                p = p.play_move(m, color);
            }
            color = p.swap_colors(color);
            previous_move = m;
        }
        //std::cout << p.str();
        //std::cout << "score " << p.score() << "\n" << std::endl;
    }
    return 0;
}

void tree_test(int n)
{
/*    Tree  t;
    
    assert(t.root->parent == nullptr);
    
    Board board;
    float value;
    
    for(int i=0; i<n; i++)
        t.search(board, value, 800);*/
}

struct logitem
{
    std::string state;
    char player;
    array<float, NN+1> probs;
    int action;
    float reward;
};

string get_last_model(string path)
{
    DIR *dir = opendir(path.c_str());

    struct dirent *entry = readdir(dir);

    string model = "";
    while (entry != NULL)
    {
        if (entry->d_type == DT_DIR)
        {
            string dir_name = string(entry->d_name);
            if(dir_name.length() > 10 && dir_name.substr(0, 10)=="metagraph-")
            {
                if(dir_name > model)
                    model = dir_name;
            }
        }

        entry = readdir(dir);
    }

    closedir(dir);
    return model;
}

int gtp_to_int(string str)
{
    int n = -1;
    std::transform(str.begin(), str.end(),str.begin(), ::toupper);
    if(str=="PASS")
        n = NN;
    else
    {
        if(str.length() == 2)
        {
            int row, col;
            col = (str[0] - 'A');
            if(str[0]>='J')
                col--;
            if(col >= 0 && col < WN)
            {
                row = str[1] - '1';
                if(row >= 0 && row < WN)
                {
                    row = WN - row - 1;
                    n = row * WN + col;
                }
            }
        }
    }
    return n;
}

string int_to_gtp(int n)
{
    int row, col;
    string gtp = "";
    
    if(n>=0 && n<=NN)
    {
        if(n == NN)
            return "PASS";
    
        row = n / WN;
        col = n % WN;
        
        char column_header = col + 'A';
        if(column_header>='I')
           column_header++;
        
        gtp = std::string(1, column_header)+std::to_string(WN-row);
    }
    
    return gtp;
}

void play(Network* pNetwork1, Network* pNetwork2, vector<logitem>& logs, bool verbose=true)
{
    auto game_start = std::chrono::high_resolution_clock::now();
    Tree *pt1=nullptr, *pt2=nullptr;
    bool sharing = (pNetwork1 == pNetwork2);
    if(!sharing)
    {
        if(pNetwork1!=nullptr)
        {
            pt1 = new Tree(0.01, -0.9, 1.0);  //allow resign if value is less than -0.
            pt1->pNetwork = pNetwork1;
        }
        
        if(pNetwork2!=nullptr)
        {
            pt2 = new Tree(0.01, -0.9, 1.0);
            pt2->pNetwork = pNetwork2;
        }
    }
    else
    {
        if(pNetwork1!=nullptr)
        {
            pt1 = new Tree(1.0, -2.0, 1.0);
            pt1->pNetwork = pNetwork1;
        
            pt2 = pt1;
        }
    }
    
    Board board;
    
    int status;
    Tree* current_tree, *opponent_tree;
    
    current_tree = pt1;
    opponent_tree = pt2;
    while((status=board.status())==-1)
    {
        float value;

        auto t1 = std::chrono::high_resolution_clock::now();
        
        logitem item;
        item.state = board.position.get_board();
        item.player = board.current;
        
        int n;
        if(current_tree != nullptr)
            n = current_tree->search(board, item.probs, value, 1600);
        else
        {
            vector<int> available_actions;

            do
            {
                string str;
                
                if(board.current + '0' == BLACK)
                    std::cout << "(x) next: ";
                else
                    std::cout << "(o) next: ";
                std::getline (std::cin, str);

                n = gtp_to_int(str);
                if( n == -1)
                    continue;
                
                available_actions = board.possible_actions();
            }while(!std::any_of(available_actions.begin(), available_actions.end(), [=](int i){return i==n;}));
        }
        item.action = n;
        logs.push_back(item);
        
        auto t2 = std::chrono::high_resolution_clock::now();
        auto diff = t2-t1;
        std::chrono::nanoseconds ns = std::chrono::duration_cast<std::chrono::nanoseconds>(diff);
        
        if(verbose)
        {
            std::cout << "step: " << board.steps << ", current: " << board.current << ", action: ";
        
            std::cout << int_to_gtp(n);
        
            //std::cout << ", score: " << board.score();
            std::cout << ", value: " << value;
            std::cout << ", time: " << ns.count()*1.0/1000000000 << " seconds" << std::endl;
            board.display(n);
            /*if(n==NN)
            {
                if((board.score()>0 && board.current==WHITE)
                 ||(board.score()<0 && board.current==BLACK))
                 {
                    std::cout << "pause" << std::endl;
                 }
                
            }*/
        }
        board.action(n);
        if(!sharing)
        {
            if(current_tree != nullptr)
                current_tree->change_root(current_tree->root->children[n]);
            
            if(opponent_tree != nullptr)
            {
                if(opponent_tree->root->children != nullptr)
                    opponent_tree->change_root(opponent_tree->root->children[n]);
            }
            std::swap(current_tree, opponent_tree);
        }
        else
        {
            if(current_tree != nullptr)
                current_tree->change_root(current_tree->root->children[n]);
        }
    }

    for (vector<logitem>::reverse_iterator i = logs.rbegin(); i != logs.rend(); ++i )
    {
        if(status == 0)
            i->reward = 0.0;
        else if(status == (i->player-'0'))
            i->reward = 1.0;
        else
            i->reward = -1.0;
    }
    

    if(!sharing)
    {
        delete pt1;
        delete pt2;
    }
    else
        delete pt1;
    

    auto game_end = std::chrono::high_resolution_clock::now();
    auto diff = game_end-game_start;
    std::chrono::nanoseconds game_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(diff);
    std::cout << "winner: " << status << " score: " << std::fixed << std::setprecision(1) << board.score() << " steps: " << board.steps << " time: " << game_ns.count()*1.0/1000000000 << " seconds" << std::endl;
}

int main(int argc, const char * argv[])
{
    unsigned int number_to_play = 1;
    string player1_model, player2_model;
    bool verbose = true;

    string root_path = "";
    string model_path = "";
    string data_path = "";
    int i = 0;
    while(i<argc)
    {
        if(string(argv[i]) == "-n")
        {
            number_to_play = atoi(argv[++i]);
        }
        
        if(string(argv[i]) == "-m1")
        {
           player1_model = string(argv[++i]);
        }

        if(string(argv[i]) == "-m2")
        {
           player2_model = string(argv[++i]);
        }
        
        if(string(argv[i]) == "-s")  //silent
        {
           verbose = false;
        }
        
        if(string(argv[i]) == "-md")
        {
            model_path = string(argv[i+1]);
        }
        
        if(string(argv[i]) == "-dd")
        {
            data_path = string(argv[i+1]);
        }
        
        if(string(argv[i]) == "-root")
        {
            root_path = string(argv[i+1]);
        }

        i++;
    }

    if(root_path=="")
        root_path = "../";

    root_path = root_path + std::to_string(WN)+"x"+std::to_string(WN)+"/";
    if(model_path == "")
        model_path = root_path + "models/";
    if(data_path == "")
        data_path = root_path + "data/";

    bool auto_refresh_model = false;
    if(player1_model == "")
    {
        auto_refresh_model = true;
        string model = get_last_model(model_path);
        if(model == "")
        {
            cout << "no network model specified in " << model_path << endl;
            return -1;
        }
        else
            player1_model = model;
    }
    
    if(player2_model == "" || auto_refresh_model)
    {
        if( player2_model != "" )
            cout << "warning: " << player2_model << " is ignored!" << endl;
        player2_model = player1_model;
    }

    bool is_self_play = (player1_model == player2_model) && (player1_model != "human");
    
    Network *pnetwork1, *pnetwork2;
    
    pnetwork1 = nullptr;
    if(player1_model != "human")
    {
        pnetwork1 = new Network();
        if( pnetwork1->LoadModel(player1_model.length()>18?player1_model:model_path+player1_model, "SERVING") != 0 )
        {
            delete pnetwork1;
            return -1;
        }
    }
    
    pnetwork2 = nullptr;
    if(player2_model != "human")
    {
        if(player2_model != player1_model)
        {
            pnetwork2 = new Network();
            if( pnetwork2->LoadModel(player2_model.length()>18?player2_model:model_path+player2_model, "SERVING") != 0 )
            {
                if(pnetwork1 != nullptr)
                    delete pnetwork1;
                delete pnetwork2;
                return -1;
            }
        }
        else
        {
            pnetwork2 = pnetwork1;
        }
    }

    vector<vector<logitem>> games;
    int games_played = 0;
    
    while(true)
    {
        if(auto_refresh_model)
        {
            string model = get_last_model(model_path);
            if(model > player1_model)
            {
                player1_model = model;
                pnetwork1->LoadModel(model_path+player1_model, "SERVING");
            }
        }

        vector<logitem> logs;
        logs.clear();
        try
        {
            play(pnetwork1, pnetwork2, logs, verbose);
        }
        catch(...)
        {
            continue;
        }
        
        games_played++;
        if(number_to_play == 0 && is_self_play)
        {
            games.push_back(logs);
            if(games.size()==100)
            {
                int generation = atoi(player1_model.substr(player1_model.size() - 8).c_str());
                std::time_t now = std::time(NULL);
                std::tm * ptm = std::localtime(&now);
                char buffer[32];
                // Format: Mo, 2018-03-17 20-20-00
                std::strftime(buffer, 32, "%Y-%m-%d %H-%M-%S", ptm);

            
                //auto t1 = std::chrono::high_resolution_clock::now();
                std::ofstream output_file;
                ostringstream oss;
                oss << "selfplay-" << std::setfill('0') << std::setw(8) << generation << "-" + string(buffer) + ".txt";

                output_file.open(data_path + oss.str());
                for(auto& game : games)
                {
                    for(auto& step : game)
                    {
                        output_file << step.state << "," << step.action << ",";
                        for(auto& p : step.probs)
                            output_file << std::fixed << std::setprecision(2) << p << ",";
                        output_file << step.reward << "\n";
                    }
                }
                output_file.close();
                std::cout << "saved " << oss.str() << std::endl;
                
                games.clear();
                //auto t2 = std::chrono::high_resolution_clock::now();
                //auto diff = t2-t1;
                //std::chrono::nanoseconds ns = std::chrono::duration_cast<std::chrono::nanoseconds>(diff);
                //std::cout << ns.count()*1.0/1000000000 << " seconds" << std::endl;
            }
        }
        else if(number_to_play == 0) // if not self play, no need to record play history
            continue;
        else if(games_played >= number_to_play)
            break;
    }
    return 0;
}
