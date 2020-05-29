#include <vector>
#include <random>
#include <chrono>
#include "bit_board.h"
#include "chess_board.h"
#include "tree.h"
#include "config.h"
#include <dirent.h>
#include <stdio.h>
#include <fstream>
#include <iomanip>
#include <unistd.h>
#include <thread>
#include <execinfo.h>
#include <signal.h>
#include "tensorflow/compiler/tf2tensorrt/utils/trt_logger.h"
#include "NvInfer.h"
#include "NvUffParser.h"

using namespace std;
using namespace nvuffparser;
using namespace nvinfer1;

#include "logger.h"
#include "common.h"

#define HUMAN	"eric"

std::mt19937 generator(std::random_device{}());

void print_bits(uint64_t x)
{
	for(int i=7; i>=0; i--)
	{
		for(int j=0; j<8; j++)
		{
			if( (x & (0x1ull<<(i*8+j))) != 0)
				cout << "1 ";
			else
				cout << ". ";
		}
		cout << endl;
	}
	cout << endl;
}

void rotate_board(BYTE *b, BYTE *n)
{
	for(int i=0; i<8; i++)
	{
		for(int j=0; j<8; j++)
		{
			BYTE c = EMPTY;
			if( b[i*8+j] != EMPTY)
				c = b[i*8+j] ^ 1;
			n[63-(i*8+j)] = c;
		}
	}
}

void show(BYTE *b)
{
	cout << "   ";
	for(int j=0; j<8; j++)
		cout << char('A'+j) << " ";

	cout << endl;
	for(int i=7; i>=0; i--)
	{
		cout << " " << i+1 << " ";
		for(int j=0; j<8; j++)
		{
			cout << PIECE_CHARS[b[i*8+j]] << " ";
		}
		cout << endl;
	}
}

void show_board(BYTE *b)
{
	show(b);
	cout << endl;
	BYTE w[64];
	rotate_board(b, w);
	show(w);
}

void show_tensor(float *p)
{
	for(int s=0; s<119; s++)
	{
		for(int i=0; i<8; i++)
		{
			for(int j=0; j<8; j++)
				cout << p[s*64+i*8+j] << " ";

			if(i==4)
				cout << "   " << s;
			cout << endl;
		}
		cout << endl;
	}
}

float kl(vector<float>& a, vector<float>& b)
{
	assert(a.size()==b.size());

	vector<float> c;
	float sum = 0.0;
	for(int i=0; i<a.size(); i++)
	{
		sum += a[i] * log(a[i]/b[i]);
	}
	return sum;
}

vector<float> get_uct(TreeNode* parent, Tree* pT)
{
    assert(parent->children != nullptr);
    
    float total_visits_sqrt = sqrt(parent->visits);
    vector<float> uct(parent->children_num, 0.0f);

    if (parent->parent == nullptr && pT->mode == SELF_PLAY)
    {
        for(int i=0; i<parent->children_num; i++)
            uct[i] = parent->children[i].puct_value(total_visits_sqrt=total_visits_sqrt, CPUCT, pT->dirichlet_noise[i], 0.25);
    }
    else
    {
        for(int i=0; i<parent->children_num; i++)
            uct[i] = parent->children[i].puct_value(total_visits_sqrt=total_visits_sqrt, CPUCT);
    }

    return uct;
}


void show_node(TreeNode*p, vector<float>& ref)
{
	vector<float> a;
	for(int i=0; i<p->children_num; i++)
	{
		cout << p->children[i].id << " " << InternalMoveToString(ActionToInternalMove(p->pos.side, p->children[i].id));
		cout << " " << p->children[i].visits << " " << setprecision(5) << ref[i]*100 << " " << p->children[i].prob*100 << " " << p->children[i].mean_value*100;
		cout << " " << p->children[i].original_value << endl;
		a.push_back(p->children[i].prob);
	}
	cout << "kl=" << kl(a, ref) << endl;
	cout << "value=" << p->original_value << endl;
	show(p->pos.board_array);
}

void show_node(TreeNode* p, Tree* pT=nullptr)
{
	show(p->pos.board_array);
	cout << endl;
	cout << "total visits = " << p->visits << endl;
	cout << "index\tmove\taction\tvisits\tprob\tmean\tprior\tvalue";
	if(pT!=nullptr)
		cout << "\tdir\tpuct";
	cout << endl;
	cout << "------------------------------------------------------------------" << endl;
	vector<float> uct = get_uct(p, pT);
	for(int i=0; i<p->children_num; i++)
	{
		cout << i << "\t" << InternalMoveToString(ActionToInternalMove(p->pos.side, p->children[i].id)) << "\t" << p->children[i].id << "\t"; 
		cout << p->children[i].visits << "\t" << setprecision(3) << p->children[i].visits*100.0/(p->visits-1) << "\t" << setprecision(3) << p->children[i].mean_value*100.0 << "\t" << setprecision(5) << p->children[i].prob*100.0 << "\t";
		cout << setprecision(3) << p->children[i].original_value*100.0 << "\t";
		if(pT!=nullptr)
		{
			cout << setprecision(3) << pT->dirichlet_noise[i]*100.0 << "\t";
			cout << uct[i];
		}
		cout << endl;
	}
}


string InternalMoveToString(int m)
{
	int from_col = (m&0xff)%8;
	int from_row = (m&0xff)/8;
	int to_col = ((m>>8)&0xff)%8;
	int to_row = ((m>>8)&0xff)/8;
	int promotion = (m>>16)&0xff;

	if(from_col < 0 || from_col >= 8
	 ||from_row < 0 || from_row >= 8
	 ||to_col < 0 || to_col >= 8
	 ||to_row < 0 || to_row >= 8)
		throw "Invalid move!";

	if(promotion != 0
  	 &&promotion != QUEEN
     &&promotion != ROOK
     &&promotion != KNIGHT
     &&promotion != BISHOP)
		throw "Invalid promotion!";

	string s = string(1, char('a'+from_col))+to_string(from_row+1)+string(1, char('a'+to_col))+to_string(to_row+1);

	if(promotion!=0)
		s += PIECE_CHARS[promotion|0x1]; //always use the lower case
	return s;
}

int StringToInternalMove(string str)
{
	if(str.length()!=4 && str.length()!=5)
		throw "Invalid move!";

	int from_col = toupper(str[0]) - 'A';
	int from_row = toupper(str[1]) - '1';
	int to_col = toupper(str[2]) - 'A';
	int to_row = toupper(str[3]) - '1';

	if(from_col < 0 || from_col >= 8
	 ||from_row < 0 || from_row >= 8
	 ||to_col < 0 || to_col >= 8
	 ||to_row < 0 || to_row >= 8)
		throw "Invalid move!";
    
    int promotion = 0;
	if(str.length()==5)
		switch(toupper(str[4]))
		{
			case 'Q': promotion=QUEEN; break;
			case 'R': promotion=ROOK; break;
			case 'N': promotion=KNIGHT; break;
			case 'B': promotion=BISHOP; break;
			default:
				throw "Invalid promotion!";
		}
	return ((from_row*8)+from_col) + (((to_row*8)+to_col)<<8) + (promotion << 16);
}

int ActionToInternalMove(int side, int n)
{
	if(n<0 || n>=73*64)
		throw "Invalid action " + to_string(n);
	int pos = n / 73;  // position

	int from_col = pos % 8;
	int from_row = pos / 8;
	int to_row, to_col;
	
	int promotion = 0;
	int m = n % 73;
	if( m < 56 ) {
		//sliding move or promoting to queen
		int ox[] = { 1, 1, 0,-1,-1,-1, 0, 1};
		int oy[] = { 0, 1, 1, 1, 0,-1,-1,-1};
		int direction = m / 7;
		int steps = m % 7;
		to_row = from_row + oy[direction]*(steps+1);
		to_col = from_col + ox[direction]*(steps+1);
	}
	else if (m < 64 ) {
		//knight move
		int direction = m - 56;
		int ox[] = { 2, 1,-1,-2,-2,-1, 1, 2};
		int oy[] = { 1, 2, 2, 1,-1,-2,-2,-1};
		to_row = from_row + oy[direction];
		to_col = from_col + ox[direction];
	}
	else {
		//promoting to rook/knight/bishop
		int direction = (m-64) / 3; 
		promotion = (m-64) % 3; //from left to right rook, knight, bishop
		switch(promotion)
		{
			case 0:
				promotion=ROOK;
				break;
			case 1:
				promotion=KNIGHT;
				break;
			case 2:
				promotion=BISHOP;
				break;
		}
		to_row = from_row + 1;
		to_col = from_col + direction - 1;
	}

	if(side==BLACK)
	{
		from_row = 7-from_row;
		from_col = 7-from_col;
		to_row = 7-to_row;
		to_col = 7-to_col;
	}
	return ((from_row*8)+from_col) + (((to_row*8)+to_col)<<8) + (promotion << 16);
}

int InternalMoveToAction(int side, int m)
{
	int from = m&0xff;  // rotate the board to always the current player facing the board
	int to = (m>>8)&0xff;
	BYTE promotion = (m>>16)&0xff;

	if(side==BLACK)
	{
		from = 63-from;
		to = 63-to;
	}


	int x_offset = to%8 - from%8;
	int y_offset = to/8 - from/8;

	int direction=-1, step=0;
    int index;

	if(promotion ==0 || promotion==QUEEN)
	{
		if( x_offset * y_offset == 0 || abs(x_offset) == abs(y_offset) )
		{
			if(y_offset == 0) // direction 0 or 4
			{
				if(x_offset>0) // 0
				{
					direction = 0;
					step = x_offset;
				}
				else 
				{
					direction = 4;
					step = abs(x_offset);
				}
			}
			else if( x_offset == 0 ) // direction 2 or 6
			{
				if(y_offset>0) // 2
				{
					direction = 2;
					step = y_offset;
				}
				else
				{
					direction = 6;
					step = abs(y_offset);
				}
			}
			else if( abs(x_offset) == abs(y_offset) ) // direction 1, 3, 5, 7
			{
				step = abs(y_offset);
				if(x_offset>0) // 1
				{
					if(y_offset>0)
						direction = 1;
					else
						direction = 7;
				}
				else
				{
					if(y_offset>0)
						direction = 3;
					else
						direction = 5;
				}
			}
			index = from*73 + direction*7 + step - 1;
		}
		else if( min(abs(x_offset), abs(y_offset))==1 && max(abs(x_offset), abs(y_offset)) == 2) // knight
		{
			if(x_offset==2 && y_offset==1)
				direction = 0;
			else if(x_offset==1 && y_offset==2)
				direction = 1;
			else if(x_offset==-1 && y_offset==2)
				direction = 2;
			else if(x_offset==-2 && y_offset==1)
				direction = 3;
			else if(x_offset==-2 && y_offset==-1)
				direction = 4;
			else if(x_offset==-1 && y_offset==-2)
				direction = 5;
			else if(x_offset==1 && y_offset==-2)
				direction = 6;
			else if(x_offset==2 && y_offset==-1)
				direction = 7;
			else
				throw "incorrect coordinates";
			index = from*73 + 7*8 + direction;
		}
		else
			throw "incorrect coordinates";
	}
	else
	{
		if(to<from+7 || to>from+9)
			throw "Invalid promotion";

		index = from*73+64;
		index += (to-from-7)*3;
		switch(promotion)
		{
			case ROOK: break;
			case KNIGHT: index+=1; break;
			case BISHOP: index+=2; break;
			default: throw "Invalid promotion";
		}
	}

	return index;
}

struct logitem
{
    std::string state;
    char player;
    int action;
	string move;
	vector<int>	  actions;
    vector<float> probs;
    vector<float> original_probs;
	float original_value;
    float value;
    float reward;
	int	reps;
	
	logitem()
	{
		state="";
		move="";
		player=WHITE;
		action=-1;
		actions.clear();
		probs.clear();
		original_probs.clear();
		value=0.0;
		reward=0.0;
		original_value = 0.0;
		reps = 0;
	}
};

std::ostream& operator<<(std::ostream& os, const logitem& item)
{
	std::streamsize ss = std::cout.precision();
	os << item.state << std::endl;
	os << int(item.player) << "," << item.action << "," << item.move << "," << item.original_value << "," << item.value << "," << int(item.reward) << "," << item.reps << std::endl;
    for(auto& p : item.actions)
        os << p << ",";
	os << std::endl;
    for(auto& p : item.probs)
        os << std::fixed << std::setprecision(2) << p * 100 << ",";
	os << std::endl;
    for(auto& p : item.original_probs)
        os << std::fixed << std::setprecision(2) << p * 100 << ",";
	os << std::setprecision(ss);
	return os;
}

void showmoves(int side, vector<int> &actions)
{
	for(int i=0; i<actions.size(); i++)
		cout << actions[i] << " " << InternalMoveToString(ActionToInternalMove(side, actions[i])) << endl;
}

int sample_move(Board& board, Tree *pT, action_space_of_a_piece actions[], int side, logitem& item)
{
    //vector<float> probs;
    //float value;

    int n, suggestion;
    vector<float> child_values;
    n = pT->search(board, item.actions, item.probs, item.value, child_values);
//
//	for(int i=0; i<pT->root->children_num; i++)
//		cout << item.probs[i] << ",";
//	cout << endl;
//
    item.state=board.position.ToFen();
    item.player = board.current;
	item.reps=board.repetitions[board.position.ZobristHash()];

	for(int i=0; i<pT->root->children_num; i++)
		item.original_probs.push_back(pT->root->children[i].prob);

	item.original_value = pT->root->original_value;
    item.action = n;  	/*for(int i=0; i<pT->root->children_num; i++)
	{
		cout << InternalMoveToString(ActionToInternalMove(pT->root->pos.side, pT->root->children[i].id)) << "=" << probs[i] << " "; 
	}
	cout << endl;*/
	int m = ActionToInternalMove(side, n);
	item.move = InternalMoveToString(m);
	return m;
}

int manual_move(action_space_of_a_piece actions[], int side, int default_move=-1)
{
	string str;		
	do{
		if(side == WHITE)		
			std::cout << "(WHITE) next: ";
		else
			std::cout << "(BLACK) next: ";
		std::getline (std::cin, str);
		if(str=="")
			//continue;
			return default_move;
		try{
			int m = StringToInternalMove(str);
			BYTE *p = (BYTE *)&actions[0];
			int action = InternalMoveToAction(side, m);
			if(p[action] != 1)
				throw "Illegal move. Try again.";
			return m;
		}
		catch(char const* error)
		{
			cout << error << endl;
			continue;
		}
	}while(true);
	return 0;
}
 

int init_depth;

uint64_t Perft(Position pos, int depth)
{
    uint64_t nodes = 0;
	string color; 

    if (depth == 0)
	{
		//cb.Display();
	 	return 1;
	}

	int moves[100];
	int move_count = 0;
	int action_num = pos.PossibleActions(pos.side, moves, move_count);
	int c = pos.side;

	for(int x=0; x<move_count; x++)
	{
		int m = moves[x];
		Position tmp = pos;
		if(tmp.Move(m))
		{
			if(!tmp.FindCheckers(c))
			{
				// legal move. swap side to continue
				tmp.side = (tmp.side+1)%2;
				uint64_t count = Perft(tmp, depth - 1);
				if(depth == init_depth)
				{
					string s = InternalMoveToString(m);
					cout << s << " " << count;
					//cout << " " << Perft_cb.ToFen();
					cout << endl;
				}
				nodes += count;
			}
		}
		//Perft_cb.UndoMove();
	}
	
    return nodes;
}

double timing;
const char *SIDENAME[] = {"WHITE", "BLACK"};
int play(Network *pN1, Network *pN2, Board cb, vector<logitem>* game_log, int evaluation_time, bool verbose=true)
{
	if(game_log!=nullptr)
		game_log->clear();
	Tree *pT1=nullptr, *pT2=nullptr;

	if(pN1==pN2 && pN1!=nullptr)
	{
		pT1 = new Tree(SELF_PLAY, -1.0, 0.9, 800, CPUCT);  //allow resign if value is less than -0.
		pT1->pNetwork = pN1;
		pT2 = pT1;
	}
	else
	{
		if(pN1 != nullptr)
		{
			pT1 = new Tree(EVALUATE, -1.0, 0.9, evaluation_time, CPUCT);  //allow resign if value is less than -0.
			pT1->pNetwork = pN1;
		}

		if(pN2 != nullptr)
		{
			pT2 = new Tree(EVALUATE, -1.0, 0.9, evaluation_time, CPUCT);  //allow resign if value is less than -0.
			pT2->pNetwork = pN2;
		}
	}

	if(pT1 != nullptr)
		pT1->root->player = cb.current;

	if(pT2 != nullptr && pT2 != pT1)
		pT2->root->player = (cb.current+1)%2;

	if(verbose)
	{
		cout << "Game Start" << endl;
		cb.Display();
	}

	int status;
	
    auto t1 = std::chrono::high_resolution_clock::now();
	Tree* current_tree = pT1;
	Tree* opponent_tree = pT2;	
	while((status=cb.status())==-1)
	{
		int move;
		timing = 0;
		logitem item;
		int visits = 0;
		int spare_time_loop = 0;
		if(current_tree != nullptr)
		{
			int v1 = current_tree->root->visits;
			move = sample_move(cb, current_tree, cb.actions, cb.current, item);
#ifdef DEBUG
show_node(current_tree->root, current_tree);
#endif
//exit(0);
			visits = current_tree->root->visits - v1;
		}
		else // below is for evaluation with human only 
		{
			std::thread machine_player;
			int suggestion;
			if(opponent_tree!=nullptr)
			{	
				spare_time_loop = opponent_tree->root->visits;
				opponent_tree->root->player = cb.current;
				opponent_tree->spare.store(1);
				machine_player = std::thread([&]() {
				    vector<float> child_values;
					suggestion = opponent_tree->search(cb, item.actions, item.probs, item.value, child_values);
				});		    
			}
			move = manual_move(cb.actions, cb.current);
			if(opponent_tree!=nullptr)
			{
				opponent_tree->spare.store(2, memory_order::memory_order_release);
				machine_player.join();
				if(move==-1)
					move = ActionToInternalMove(cb.current, suggestion);
				opponent_tree->spare.store(0); //restore to 0 for next move
				spare_time_loop = opponent_tree->root->visits - spare_time_loop;
			}
		}

		if(game_log!=nullptr)
			game_log->push_back(item);

		int action = InternalMoveToAction(cb.current, move);
		double value = 0.0, prob=0.0;
		double original_value, original_prob;
		if(verbose)
		{
			if(current_tree!=nullptr)
			{
				value=item.value;
				prob = item.probs[current_tree->root->GetChild(action)-current_tree->root->children];
				original_value = current_tree->root->original_value;
				original_prob = item.original_probs[current_tree->root->GetChild(action)-current_tree->root->children];
				
				/*show_node(current_tree->root, item.probs);
				vector<TreeNode*> nodes;
				nodes.push_back(current_tree->root);
				vector<int> methods;
				methods.push_back(0);
				Tensor *p = current_tree->MakeTensor(nodes, methods);
				show_tensor(p->flat<float>().data());
				delete p;*/
			}
			else if (opponent_tree != nullptr)
			{
				value=item.value;
				prob = item.probs[opponent_tree->root->GetChild(action)-opponent_tree->root->children];
				original_value = opponent_tree->root->original_value;
				if( item.original_probs.size() == 0)
				{
					for(int i=0; i<opponent_tree->root->children_num; i++)
						item.original_probs.push_back(opponent_tree->root->children[i].prob);
				}
				original_prob = item.original_probs[opponent_tree->root->GetChild(action)-opponent_tree->root->children];
				/*show_node(opponent_tree->root, item.probs);
				vector<TreeNode*> nodes;
				nodes.push_back(opponent_tree->root);	
				vector<int> methods;
				methods.push_back(0);
				Tensor *p = opponent_tree->MakeTensor(nodes, methods);
				show_tensor(p->flat<float>().data());
				delete p;*/
			}

			std::cout << "step " << cb.steps << ": " << SIDENAME[cb.current] << ", legal moves=" << cb.legal_moves;
			std::cout << ", action=" << InternalMoveToString(move) << "(" << action << ")" << ", search=" << visits << "(" << spare_time_loop << ")" << std::endl;
		}
		cb.action(InternalMoveToAction(cb.current, move));
		if(verbose)
		{
			cb.Display();
			std::cout << "\033[s\033[9A\033[20C hash=" << std::hex << cb.position.ZobristHash() << "(" << cb.repetitions[cb.position.ZobristHash()] << ")" << std::dec << "\033[u" << std::flush;
			std::cout << "\033[s\033[8A\033[20C value=" << value << "(" << original_value << ")" << "\033[u" << std::flush;
			std::cout << "\033[s\033[7A\033[20C prob=" << prob << "(" << original_prob << ")    " << prob/original_prob << "("<< original_prob*1.0*cb.legal_moves << ")" << "\033[u" << std::flush;

			/*
			Tree *p = current_tree;
			if(current_tree==nullptr)
				p = opponent_tree;

			for(int x=0; x< p->root->children_num; x++)
				std::cout << InternalMoveToString(ActionToInternalMove(p->root->pos.side, p->root->children[x].id)) << " - " << p->root->children[x].prob << std::endl;
			*/
		}

		if(current_tree != nullptr)
		{
			TreeNode *child = current_tree->root->GetChild(action);
			current_tree->change_root(*child);
		}
        if(pT1!=pT2)
        {
            if(opponent_tree != nullptr && opponent_tree->root->children != nullptr)
			{
				TreeNode *child = opponent_tree->root->GetChild(action);
				opponent_tree->change_root(*child);
			}
            std::swap(current_tree, opponent_tree);
        }
	}
	if(pT1==pT2)
	{
		delete pT1;
	}
	else
	{
		if(pT1!=nullptr)
			delete pT1;
		if(pT2!=nullptr)
			delete pT2;
	}

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::nanoseconds ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1);
	
	switch(status)
	{
		case STATUS_DRAW:
		case STATUS_REP3_DRAW:
			cout << "steps: " << dec << cb.steps;
			if(status == STATUS_DRAW)
				cout << ". Draw. Play again. Time: ";
			else
				//cout << ". Draw (rep 3). Play again. ZobristHash: " << hex << cb.position.ZobristHash() << "(" << cb.repetitions[cb.position.ZobristHash()] << "). " << "Time: ";
				cout << ". Draw (rep 3). Play again. Time: ";
			cout << ns.count()*1.0/1000000000 << " seconds." << " Average: " << cb.steps*1.0*1000000000/ns.count() << endl;
			//std::cout << "hash=" << std::hex << cb.position.hash << std::dec << std::endl;
			break;
		case STATUS_STALEMATE:
			cout << "steps: " << cb.steps << ". Draw. Stalemate. Play again. Time: " << ns.count()*1.0/1000000000 << " seconds." << " Average: " << cb.steps*1.0*1000000000/ns.count() << endl;
			break;
		case WHITE:
			cout << "steps: " << cb.steps << ". \033[1;31mWhite\033[0m won. Time: " << ns.count()*1.0/1000000000 << " seconds." << " Average: " << cb.steps*1.0*1000000000/ns.count() << endl;
			break;
		case BLACK:
			cout << "steps: " << cb.steps << ". \033[1;32mBlack\033[0m won. Time: " << ns.count()*1.0/1000000000 << " seconds." << " Average: " << cb.steps*1.0*1000000000/ns.count() << endl;
			break;
	}	
	
	return status;
}

void run_perft(string& fen, int depth)
{
	Position perft = Position::initial_state();

	if(fen != "")
	{
		perft.ParseFen(fen);
		assert(fen == perft.ToFen());
	}
	//cout << fen << endl;		
	//cout << perft.ToFen() << endl;		

	init_depth = depth;
   	auto start = std::chrono::high_resolution_clock::now();
	uint64_t n = Perft(perft, depth);
   	auto end = std::chrono::high_resolution_clock::now();

	chrono::nanoseconds ns = chrono::duration_cast<chrono::nanoseconds>(end-start);
	cout << n << ", time: " << ns.count()*1.0/1000000000 << " seconds" << endl;
}

void test_move_conversion()
{
    assert(InternalMoveToString(StringToInternalMove("a2a3"))=="a2a3");
    assert(InternalMoveToString(StringToInternalMove("h7h8"))=="h7h8");
    assert(InternalMoveToString(StringToInternalMove("h7h8r"))=="h7h8r");
    assert(InternalMoveToString(StringToInternalMove("a2a1q"))=="a2a1q");
	try
	{
		assert(InternalMoveToString(ActionToInternalMove(1, InternalMoveToAction(1, StringToInternalMove("a1h8"))))=="a1h8");
		assert(InternalMoveToString(ActionToInternalMove(1, InternalMoveToAction(1, StringToInternalMove("a8h1"))))=="a8h1");
		assert(InternalMoveToString(ActionToInternalMove(1, InternalMoveToAction(1, StringToInternalMove("b1c3"))))=="b1c3");
		assert(InternalMoveToString(ActionToInternalMove(1, InternalMoveToAction(1, StringToInternalMove("b8a6"))))=="b8a6");

		assert(InternalMoveToString(ActionToInternalMove(0, InternalMoveToAction(0, StringToInternalMove("a1h8"))))=="a1h8");
		assert(InternalMoveToString(ActionToInternalMove(0, InternalMoveToAction(0, StringToInternalMove("a8h1"))))=="a8h1");
		assert(InternalMoveToString(ActionToInternalMove(0, InternalMoveToAction(0, StringToInternalMove("b1c3"))))=="b1c3");
		assert(InternalMoveToString(ActionToInternalMove(0, InternalMoveToAction(0, StringToInternalMove("b8a6"))))=="b8a6");

		//assert(InternalMoveToAction(0, StringToInternalMove("a2a3")) == 2672);
		assert(InternalMoveToAction(0, StringToInternalMove("a1b1")) == 0);
		assert(InternalMoveToAction(0, StringToInternalMove("a1c1")) == 1);
		assert(InternalMoveToAction(0, StringToInternalMove("a1d1")) == 2);
		assert(InternalMoveToAction(0, StringToInternalMove("a1e1")) == 3);
		assert(InternalMoveToAction(0, StringToInternalMove("a1f1")) == 4);
		assert(InternalMoveToAction(0, StringToInternalMove("a1g1")) == 5);
		assert(InternalMoveToAction(0, StringToInternalMove("a1h1")) == 6);

		assert(InternalMoveToAction(0, StringToInternalMove("a1b2")) == 7);
		assert(InternalMoveToAction(0, StringToInternalMove("a1c3")) == 8);

		assert(InternalMoveToAction(0, StringToInternalMove("a1a2")) == 14);
		assert(InternalMoveToAction(0, StringToInternalMove("a2a3")) == 73*8+14);
    }
	catch(char const *e)
	{
		cout << e << endl;
	}
}

void fen_test()
{
	Position pos;
	//pos.ParseFen("5bn1/P2B4/3k4/3R2P1/3NKB1n/6P1/7p/3r4 b - - 1 1");
	//action_space_of_a_piece actions[64];
	//int legal_moves = pos.LegalMoves(actions);

	pos.ParseFen("n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1");
	assert(Perft(pos, 1)==24);
	assert(Perft(pos, 2)==496);
	assert(Perft(pos, 3)==9483);
	assert(Perft(pos, 4)==182838);
	assert(Perft(pos, 5)==3605103);
	assert(Perft(pos, 6)==71179139);
	cout << "pass" << endl;

	pos.ParseFen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
	assert(Perft(pos, 1)==48);
	assert(Perft(pos, 2)==2039);
	assert(Perft(pos, 3)==97862);
	assert(Perft(pos, 4)==4085603);
	assert(Perft(pos, 5)==193690690);
	//assert(Perft(6)==8031647685);
	cout << "pass" << endl;

	pos.ParseFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
	assert(Perft(pos, 1)==20);
	assert(Perft(pos, 2)==400);
	assert(Perft(pos, 3)==8902);
	assert(Perft(pos, 4)==197281);
	assert(Perft(pos, 5)==4865609);
	assert(Perft(pos, 6)==119060324);
	//assert(Perft(7)==3195901860);
	cout << "pass" << endl;
}

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
            if(std::isdigit(dir_name[0]))
            {
                if(std::atoi(dir_name.c_str()) > std::atoi(model.c_str()))
                    model = dir_name;
            }
        }

        entry = readdir(dir);
    }

    closedir(dir);
    return model;
}

string generate_log_file_name(string latest_model)
{
	char buffer[32];
	std::time_t now = std::time(NULL);
	std::tm * ptm = std::localtime(&now);
	// Format: Mo, 2018-03-17 20-20-00
	std::strftime(buffer, 32, "%Y-%m-%d %H-%M-%S", ptm);
	ostringstream oss;
	oss << "selfplay-" << std::setfill('0') << std::setw(8) << latest_model << "-" + string(buffer) << "-" << ::getpid() << ".txt";
	return oss.str();
}

void handler(int sig) {
  void *array[10];
  size_t size;

  // get void*'s for all entries on the stack
  size = backtrace(array, 10);

  // print out all the frames to stderr
  fprintf(stderr, "Error: signal %d:\n", sig);
  backtrace_symbols_fd(array, size, STDERR_FILENO);
  exit(1);
}

int main(int argc, const char * argv[])
{
	//cout << "\033[1;31mwhite\033[0m" << endl;
	//cout << "\033[1;32mblack\033[0m" << endl;
	//exit(0);
	InitZobrist();
	//signal(SIGSEGV, handler);

	int i = 0;
	int mode = 0;  // 0 self play, 1 perft, 2 fen test. 3 evaluation (must provide players)
	int depth = 0;
	int manual=0;
	bool verbose=true;
	bool log_games = false;
	bool log_draw_games = true;
	int number_of_games = 1;
	bool self_play = false;
	int evaluation_time = 200; // default to 200 ms

	InitializeMasks();

	string model_path = "../8x8/models/";
	string latest_model = get_last_model(model_path);
	string player1="", player2="";
	string fen = BOARD_START_FEN;
    while(i<argc)
    {
        if(string(argv[i]) == "-p")
        {
			mode = 1; // perft
            depth = atoi(argv[++i]);
        }
		
        if(string(argv[i]) == "-f")
        {
            fen = argv[++i];
        }

        if(string(argv[i]) == "-m1")
        {
            player1 = argv[++i];
        }

        if(string(argv[i]) == "-m2")
        {
            player2 = argv[++i];
        }
		
        if(string(argv[i]) == "-t")
        {
			mode=2; //fen test
		}
        if(string(argv[i]) == "-s")  //silent
        {
           verbose = false;
        }
        if(string(argv[i]) == "-e")
        {
            evaluation_time = atoi(argv[++i]);
        }
        if(string(argv[i]) == "-n")
        {
            number_of_games = atoi(argv[++i]);
        }
        if(string(argv[i]) == "-l" || string(argv[i]) == "-log")
        {
            log_games = true;
        }
        if(string(argv[i]) == "-lne" || string(argv[i]) == "-logne")
        {
            log_games = true;
			log_draw_games = false;	
        }

		i++;
	}

	if(mode==0) // play game/games
	{
///////// workaround to avoid tensorrt malloc assert
		Logger mylogger;
		nvinfer1::IBuilder* builder1 = nvinfer1::createInferBuilder(mylogger);
		builder1->destroy();
/////////
		Network *pnetwork1=nullptr, *pnetwork2=nullptr;

		if(player2=="")
			player2 = player1;
		if(player1 == player2 && player1 != HUMAN)  // same machine players
		{
			self_play = true;
			pnetwork1 = new Network();
			string model=player1;
			if(model=="")
				model=latest_model;
			if( pnetwork1->LoadModel(model_path+model, "serve") != 0 )
				return -1;
			pnetwork2 = pnetwork1;
		}
		else  // 2 different machine players or at least 1 human player
		{
			string model;
			if(player1!=HUMAN)
			{
				pnetwork1 = new Network();
				model=player1;
				if(model=="")
					model=latest_model;
				if( pnetwork1->LoadModel(model_path+model, "serve") != 0 )
					return -1;
			}

			if(player2 != HUMAN)
			{
				pnetwork2 = new Network();
				model=player2;
				if(model=="")
					model=latest_model;

				if( pnetwork2->LoadModel(model_path+model, "serve") != 0 )
					return -1;
			}
		}


		Board cb;
		if(fen!="")
			cb = Board(fen.c_str(), WHITE);

		string data_path = "../8x8/data/";
		string log_file = generate_log_file_name(latest_model);

		//const int thread_num = 1;
		//std::thread runner[thread_num];
		//std::cout << "Starting..." << std::endl;

		//for(int ti=0; ti<thread_num; ti++)
		{
			//runner[i] = std::thread([&](int thread_id, int nGames) {
				vector<logitem> game_log;
				game_log.reserve(256);
				for(int g=0; g<number_of_games; g++)
				{
						cout << "Game " << dec << g+1 << ", ";
						int status;
						if(log_games)
						{
							status = play(pnetwork1, pnetwork2, cb, &game_log, evaluation_time, verbose);  // if two networks are different and not null then self play otherwise evaluate
							if(log_draw_games)
							{
								std::ofstream output_file;
								output_file.open(data_path + log_file, std::ofstream::out | std::ofstream::app);
								output_file << "START" << endl;
								for(auto& item : game_log) 
								{
									if(status==BLACK || status==WHITE)
										item.reward = (status==item.player)?1.0:-1.0;
									else
										item.reward = 0.0;
	
									output_file << item << std::endl;
								}
								output_file << "FINISH" << endl;
							}
						}
						else
							status = play(pnetwork1, pnetwork2, cb, nullptr, evaluation_time, verbose);  // if two networks are different and not null then self play otherwise evaluate
				}
			//}, ti, number_of_games/thread_num);
		}

		//for(int ti=0; ti<thread_num; ti++)
		//	runner[ti].join();
		//std::cout << "Finish..." << std::endl;

		if(pnetwork1==pnetwork2)
		{
			if(pnetwork1!=nullptr)
				delete pnetwork1;
		}
		else
		{
			if(pnetwork1!=nullptr)
				delete pnetwork1;

			if(pnetwork2!=nullptr)
				delete pnetwork2;
		}
	}
	else if (mode == 1)
		run_perft(fen, depth);
	else if( mode == 2)
		fen_test();

	return 0;
}	


