/*MIT License

Copyright (c) 2019 Minglei Wang

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


#ifndef __chess_board_h
#define __chess_board_h

#include <string>
#include <unordered_map>
#include "bit_board.h"

using namespace std;

struct Step
{
    string state;
    char player;
    int action;
    double value;
};

#define NN 64*73

#define STATUS_DRAW			2
#define STATUS_REP3_DRAW	3

#define STATUS_STALEMATE	10
#define STATUS_UNKNOWN		-1

extern void show(BYTE *b);
class PassthroughHash {
   public:
    std::size_t operator()(uint64_t x) const {
      return static_cast<std::size_t>(x);
    }
};

#define SLICES 7  //maximum 7 history steps plus current state equals 8 to what the paper defines 
class Board
{
public:
	action_space_of_a_piece actions[64];
	int legal_moves = 0;
    int pass_count;
    
    Position position;
    char current;
    bool symmetrical;
    int action_space;
    int steps;
	std::unordered_map<uint64_t, int, PassthroughHash> repetitions;	
    
    Board()
    {
        position = Position::initial_state();
        symmetrical = false;
        action_space = NN + 1;
        current = WHITE;
        steps = 0;
        pass_count = 0;
		repetitions.clear();
    }
    
    Board(const char *initial_state, char player) : Board()
    {
		if(string(initial_state) != "")
	        this->position = Position(string(initial_state), EMPTY);
		else
        	this->position = Position::initial_state();
        current = this->position.side;

		repetitions[position.ZobristHash()]=1;
    }
    
    /*vector<int> get_actions(int selector)
	{
        std::vector<int> selected_action; // vector with all moves

		int *p = (int *)&(actions[0]);
		for(int i=0; i<sizeof(actions); i++)
			if(p[i] == selector)
				selected_action.push_back(i);

		return selected_action;
	}

    vector<int> possible_actions()
	{
		return get_actions(1);
	}

    vector<int> not_allowed_actions()
    {
		return get_actions(0);
    }*/

    void action(int n)
	{
        if(n != NN)
        {
            position = position.play_move(n, current);
            pass_count = 0;
        }
        else
		{
            pass_count++;
		}

        steps += 1;
        current = (current+1)%2;
        position.side = current;
		repetitions[position.ZobristHash()]++;
	}

	//status after enermy finishes a move. so this should always be called after function action()
    int status(float value=0.0)
    {
		assert(current == position.side);
		if(position.fifty_moves == 50)
			return STATUS_DRAW;
		else if(pass_count==1) // in chess, pass means resign
			return (current+1)%2;	//return the winner
		else
		{
			if( position.insufficient_material() )
				return STATUS_DRAW;
			BYTE *p = (BYTE *)&actions[0];
			memset(p, 0, sizeof(actions));
			legal_moves = position.LegalMoves(actions);
			//if no legal move, stop the game
			if(!legal_moves)
			{
				if(!position.FindCheckers(position.side)) // stalemate it is not in check
					return STATUS_STALEMATE;
				else
					return (position.side+1)%2;	//current side loss the game. return the winner
			}
			if(steps>512)
				return STATUS_DRAW;

  			const auto entry = repetitions.find(position.ZobristHash());
			if(entry == repetitions.end())
			{
				show(position.board_array);
				cout << hex << position.hash << endl;
				assert(entry != repetitions.end());
			}
			if(entry->second >= 3)
			{
				//cout << std::hex << "Rep 3 times draw: " << entry->first << std::dec << endl;
				//Display();				
				return STATUS_REP3_DRAW;
			}
			return STATUS_UNKNOWN;
		}
    }
    
    float score()
    {
        return 0.0; // score should be 1, 0 or -1 
    }
    
    void display(int next_move=-1, int optimal_move=-1, array<float, NN+1>* probs=NULL, array<float, NN+1>* values=NULL)
    {
		Display();
    }

    void Display()
    {
        enum class Color {BBLACK=30, RED=31, GREEN=32, YELLOW=33, BLUE=34, MAGENTA=35, CYAN=36, WWHITE=37};

        std::string header = "    ";
		std::string current_piece;

        //the header line
        for(int i=0; i<8; i++)
        {
            std::string tmp = "\033[0;" + std::to_string((int)Color::WWHITE) + "m." + "\033[0m";  //white;
            char column_header = 'A' + i;
            if(i>=8)
                column_header++;
            std::replace(tmp.begin(), tmp.end(), '.', column_header);
            header += tmp;
            header += "\033[1;" + std::to_string((int)Color::WWHITE) + "m " + "\033[0m";
        }
        std::cout << header;

		std::cout << " " << position.ToFen() << std::endl;

		for(int row=7; row>=0; row--)
		{
        	std::string s = "";
			for(int col=0; col<8; col++)
			{
				char c = PIECE_CHARS[position.board_array[row*8+col]];
				Color piece_color;
				if(c=='.')
				{
					if((col+row%2) % 2)
						piece_color=Color::WWHITE;
					else
						piece_color=Color::BLUE;
				}
				else
				{
					if(c < 'a')  // white
						piece_color=Color::WWHITE;
					else
						piece_color=Color::BLUE;
				}
				if(c=='k'||c=='K')
		        	current_piece = "\033[1;" + std::to_string((int)piece_color) + "m" + string(1, c) + "\033[0m";
				else
		        	current_piece = "\033[0;" + std::to_string((int)piece_color) + "m" + string(1, c) + "\033[0m";

				if(row*8+col == position.last_move)
					current_piece[2] = '7';  // highlight the current move

				s += current_piece;
	            s += "\033[1;" + std::to_string((int)Color::WWHITE) + "m " + "\033[0m";
			}

            std::cout << std::setw(3) << row+1 << " " << s;
			std::cout << std::endl;
		}
        std::cout << std::endl;	
	}
};

#endif
