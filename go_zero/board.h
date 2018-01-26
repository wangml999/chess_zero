//
//  board.cpp
//  t1cpp
//
//  Created by Minglei Wang on 9/01/2018.
//  Copyright © 2018 Minglei Wang. All rights reserved.
//

#ifndef __board_h
#define __board_h

#include <string>
#include <queue>
#include "fast_go.h"
#include "boost/circular_buffer.hpp"


using namespace std;
using namespace boost;


struct Step
{
    string state;
    int player;
    int action;
    double value;
};

#define SLICES 17
class Board
{
public:
    boost::circular_buffer<Step> history;
    Position position;
    char current;
    string state;
    bool symmetrical;
    int action_space;
    int steps;
    
    Board() : position(Position::initial_state())
    {
        symmetrical = true;
        action_space = NN + 1;
        current = BLACK;
        steps = 0;
        state = position.get_board();
        history = boost::circular_buffer<Step>(3);
    }
    
    void Reset()
    {
        position = Position::initial_state();
        state = position.get_board();
        current = BLACK;
        history.clear();
        steps = 0;
    }
    
    vector<int> possible_actions()
    {
        return position.possible_moves(current);
    }

    vector<int> not_allowed_actions()
    {
        return position.not_allowed_actions(current);
    }

    void action(int n)
    {
        Step s;
        
        s.state = this->state;
        s.player = this->current;
        s.action = n;
        s.value = 0.0;
        
        history.push_back(s);

        steps += 1;

        if(n != NN)
            position = position.play_move(n, current);
            state = position.get_board();
        current = position.swap_colors(current);
    }
    
    int status()
    {
        /* Get the game result from the viewpoint of player
         if maximum steps reached, or no action to take or last two actions were pass
        */

        bool game_finished = false;

        if (history.size()>=2 && (history.end()[-2]).action == NN && history.end()[-1].action == NN)
            game_finished = true;
        else if (steps >= NN * 2)
            game_finished = true;
        /*else
        {
            vector<int> moves = possible_actions();
            if (moves.size() == 1 || (moves.size()==1 && moves[0] == position.ko))
                game_finished = true;
        }*/

        if (game_finished)
        {
            int score = position.score();
            if (score >0)
                return 1;
            else if (score < 0)
                return 2;
            else
                return 0;
        }
        return -1; //unknown
    }
    
    int score()
    {
        return this->position.score();
    }
    
    void display(int next_move=-1, int optimal_move=-1)
    {
        enum class Color {BBLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WWHITE};

        int board_color = 37;
        std::string w = "\033[1;" + std::to_string(board_color) + "m+" + "\033[0m";
        std::string a = "\033[1;" + std::to_string(31) + "mx" + "\033[0m";
        std::string b = "\033[1;" + std::to_string(32) + "mo" + "\033[0m";
        std::string y = "\033[1;" + std::to_string(33) + "m*" + "\033[0m";

        std::string s = "";
        for(int i =0; i<NN; i++)
        {
            if (i == next_move)
            {
                if (this->current == BLACK)
                {
                    std::string tmp = y;
                    std::replace(tmp.begin(), tmp.end(), '*', 'X');
                    s += tmp;
                }
                else
                {
                    std::string tmp = y;
                    std::replace(tmp.begin(), tmp.end(), '*', 'O');
                    s += tmp;
                }
            }
            else
            {
                if ((i == optimal_move) && (next_move != optimal_move))
                    s += y;
                else
                {
                    switch(state[i])
                    {
                        case EMPTY:
                            s += w;
                            break;
                        case BLACK:
                            s += a;
                            break;
                        case WHITE:
                            s += b;
                            break;
                    }
                }
            }

            if ((i % WN) == (WN-1))
                s += "\n";
            else
            {
                s += "\033[1;" + std::to_string(board_color) + "m-" + "\033[0m";
            }
        }

        std::cout << s << std::endl;
    }
};

#endif
