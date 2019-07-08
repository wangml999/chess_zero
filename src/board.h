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


#ifndef __board_h
#define __board_h

#include <string>
#include <queue>
#include "fast_go.h"
#include <iomanip>

using namespace std;

struct Step
{
    string state;
    char player;
    int action;
    double value;
};

#define SLICES 7  //maximum 7 history steps plus current state equals 8 to what the paper defines 
class Board
{
public:
    int pass_count;
    
    Position position;
    char current;
    //string state;
    bool symmetrical;
    int action_space;
    int steps;
    
    Board()
    {
        position = Position::initial_state();
        symmetrical = true;
        action_space = NN + 1;
        current = BLACK;
        steps = 0;
        
        pass_count = 0;
    }
    
    Board(const char *initial_state, char player) : Board()
    {
        this->position = Position(string(initial_state), NONE);
        current = player;
    }
    
    void Reset()
    {
        position = Position::initial_state();
        current = BLACK;
        
        steps = 0;

        pass_count = 0;
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
        if(n != NN)
        {
            position = position.play_move(n, current);
            pass_count = 0;
        }
        else
		{
			position.ko = NONE;
            pass_count++;
		}

        steps += 1;
        current = position.swap_colors(current);
    }
    
    int status()
    {
        /* Get the game result from the viewpoint of player
         if maximum steps reached, or no action to take or last two actions were pass
        */

        bool game_finished = false;

        //if (history.size()>=2 && (history.end()[-2]).action == NN && history.end()[-1].action == NN)
        if(pass_count == 2)
            game_finished = true;
        else if (steps >= NN * 2)
            game_finished = true;

        if (game_finished)
        {
            float score = position.score();
            if (score >0)
                return 1;
            else if (score < 0)
                return 2;
            else
                return 0;
        }
        return -1; //unknown
    }
    
    float score()
    {
        return this->position.score();
    }
    
    void display(int next_move=-1, int optimal_move=-1, array<float, NN+1>* probs=NULL, array<float, NN+1>* values=NULL)
    {
        enum class Color {BBLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WWHITE};

        int board_color = 37;
        std::string w = "\033[1;" + std::to_string(board_color) + "m." + "\033[0m";
        std::string a = "\033[1;" + std::to_string(31) + "mx" + "\033[0m";
        std::string b = "\033[1;" + std::to_string(32) + "mo" + "\033[0m";
        std::string y = "\033[1;" + std::to_string(33) + "m*" + "\033[0m";

        std::string s = "";
        std::string header = "    ";
        //the header line
        for(int i=0; i<WN; i++)
        {
            std::string tmp = w;
            char column_header = 'A' + i;
            if(i>=8)
                column_header++;
            std::replace(tmp.begin(), tmp.end(), '.', column_header);
            header += tmp;
            header += "\033[1;" + std::to_string(board_color) + "m " + "\033[0m";
        }

        std::cout << header;
	std::cout << std::setw(12) << " ";
	std::cout << std::setw(4) << std::setprecision(1) << std::fixed << (*probs)[NN]*100 << " ";	
	std::cout << std::setw(23) << " ";
	std::cout << std::setw(4) << std::setprecision(1) << std::fixed << (*values)[NN] << " ";	
	std::cout << std::endl;

        std::vector<int> moves = this->position.not_allowed_actions(this->current);

        for(int i=0; i<NN; i++)
        {
            if( i % WN == 0)
            {
                
            }
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
                    std::string state = this->position.get_board();
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
            {
                std::cout << std::setw(3) << WN-i/WN << " " << s;

		if(probs!=NULL)
		{
			std::cout << std::setw(3) << " ";
			for(int k=0; k<WN; k++)
			{
			    if(	std::find(moves.begin(), moves.end(), i-WN+1+k) != moves.end())
				std::cout << std::setw(4) << "-" << " ";	
			    else	
			        std::cout << std::setw(4) << std::setprecision(1) << std::fixed << (*probs)[i-WN+1+k]*100 << " ";	
			}
			//if( i == NN-1 )  
			//    std::cout << std::setw(4) << std::setprecision(1) << std::fixed << (*probs)[NN]*100 << " ";	
		}

		if(values!=NULL)
		{
			std::cout << std::setw(3) << " ";
			for(int k=0; k<WN; k++)
			{
			    if(	std::find(moves.begin(), moves.end(), i-WN+1+k) != moves.end())
				std::cout << std::setw(4) << "-" << " ";	
			    else	
			        std::cout << std::setw(4) << std::setprecision(1) << std::fixed << (*values)[i-WN+1+k] << " ";	
			}
			//if( i == NN-1 )  
			    //std::cout << std::setw(4) << std::setprecision(1) << std::fixed << (*values)[NN] << " ";	
		}

		std::cout << std::endl;
                s = "";
            }
            else
            {
                s += "\033[1;" + std::to_string(board_color) + "m " + "\033[0m";
            }
        }
        std::cout << std::endl;
    }
};

#endif
