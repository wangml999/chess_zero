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


#ifndef __bit_board_h
#define __bit_board_h

#include <string>
#include <queue>
#include <iomanip>
#include <assert.h>
#include <stack>
#include <iostream>
#include <algorithm>
#include <cstring>
#include "slider_attacks.h"
#include <array>

using namespace std;

#define BYTE	unsigned char 

#define WHITE_PAWN_START	0x000000000000ff00ull
#define WHITE_ROOK_START	0x0000000000000081ull
#define WHITE_KNIGHT_START  0x0000000000000042ull
#define WHITE_BISHOP_START  0x0000000000000024ull
#define WHITE_QUEEN_START	0x0000000000000008ull
#define WHITE_KING_START	0x0000000000000010ull

#define BLACK_PAWN_START	0x00ff000000000000ull
#define BLACK_ROOK_START	0x8100000000000000ull
#define BLACK_KNIGHT_START  0x4200000000000000ull
#define BLACK_BISHOP_START  0x2400000000000000ull
#define BLACK_QUEEN_START	0x0800000000000000ull
#define BLACK_KING_START	0x1000000000000000ull

#define BOARD_START_FEN		"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

#define EMPTY			0

#define WHITE_PAWN		2
#define WHITE_ROOK		6
#define WHITE_KNIGHT	4
#define WHITE_BISHOP	8
#define WHITE_QUEEN		10
#define WHITE_KING		12

#define BLACK_PAWN		3
#define BLACK_ROOK		7
#define BLACK_KNIGHT	5
#define BLACK_BISHOP	9
#define BLACK_QUEEN		11
#define BLACK_KING		13


#define WHITE  0
#define BLACK  1 

#define PAWN   2 
#define KNIGHT 4 
#define ROOK   6 
#define BISHOP 8 
#define QUEEN  10 
#define KING   12

#define WHITE_KING_SIDE_CASTLING_MASK	0x0000000000000060ull
#define WHITE_QUEEN_SIDE_CASTLING_MASK	0x000000000000000eull
#define BLACK_KING_SIDE_CASTLING_MASK	0x6000000000000000ull
#define BLACK_QUEEN_SIDE_CASTLING_MASK	0x0e00000000000000ull

#define WHITE_QUEEN_SIDE	0b0001
#define WHITE_KING_SIDE		0b0010
#define BLACK_QUEEN_SIDE	0b0100
#define BLACK_KING_SIDE		0b1000
#define NO_CASTLING_RIGHTS 	0b0000

#define QUEEN_SIDE			0b01
#define KING_SIDE			0b10

#define NULL_SQUARE 64

#define FILE_A	0x0101010101010101
#define FILE_H	0x8080808080808080

#define RANK_1	0x00000000000000ff
#define RANK_4	0x00000000ff000000
#define RANK_5	0x000000ff00000000
#define RANK_8	0xff00000000000000

#define WHITE_SQUARES 0xAA55AA55AA55AA55

const string PIECE_CHARS = ".EPpNnRrBbQqKk";

const	uint64_t kingside_mask[2] = {WHITE_KING_SIDE_CASTLING_MASK, BLACK_KING_SIDE_CASTLING_MASK};
const	uint64_t queenside_mask[2] = {WHITE_QUEEN_SIDE_CASTLING_MASK, BLACK_QUEEN_SIDE_CASTLING_MASK};

const	BYTE left_rook[2] = {0, 56};
const	BYTE right_rook[2] = {7, 63};
const	uint64_t left_rook_bit[2] = {0x1ull, 0x1ull<<56};
const	uint64_t right_rook_bit[2] = {0x1ull<<7, 0x1ull<<63};

const	BYTE king_start[2] = {4, 60};
const	uint64_t king_start_bit[2] = {0x0000000000000010ull, 0x1000000000000000ull};

const	BYTE en_passant_start_ranks[2] = {1, 6};
const	BYTE en_passant_end_ranks[2] = {3, 4};
const 	int offset[2] = {-8, 8};

uint64_t 	knightAttack_mask[64];
uint64_t 	kingAttack_mask[64];
uint64_t	pawnAttack_mask[2][64];
SliderAttacks slider_attacks;

extern void print_bits(uint64_t x);
extern int InternalMoveToAction(int side, int m);
extern int ActionToInternalMove(int side, int n);

uint64_t hash_piece[64][2][14]; //64 squares, 2 colors, 12 pieces from 2 to 13, 0 as empty for white
uint64_t hash_castling[16];
uint64_t hash_enpassant[8];

/*
	move definition
	4 bytes from high to low
	- QUEEN_SIDE, KING_SIDE, lower 6 bits used for en passant which is the same as TO
	- promotion - EMPTY, QUEEN, ROOK, KNIGHT, BISHOP
	- TO
	- FROM
*/
/*inline int popcount64c(uint64_t x)
{
	//return (__builtin_popcount(x) + __builtin_popcount(x>>32));
	return __builtin_popcountll(x);
}*/	


struct action_space_of_a_piece {
	std::array<std::array<BYTE, 7>, 8> sliding;
	std::array<BYTE, 8> knight;
	std::array<std::array<BYTE, 3>, 3> pawn;

	action_space_of_a_piece()
	{
		set_zeros();
	}

	void set_zeros()
	{
		memset(&sliding, 0, sizeof(sliding));
		memset(&knight, 0, sizeof(knight));
		memset(&pawn, 0, sizeof(pawn));
	}
	
	int count()
	{
		int sum = 0;
		for(int i=0; i<8; i++)
		{
			for(int j=0; j<7; j++)
				sum += sliding[i][j];
			sum += knight[i];
		}
		return sum;
	}
};

struct Position
{
	uint64_t 	bitboards[14]; 
	BYTE 		board_array[64]; 

    int 		side;
	int 		en_passant;
	int 		fifty_moves;
	int 		castling_rights;
	int 		last_move;
	uint64_t	hash;
	
    Position()
    {
		Reset();
    }
    
    Position(std::string fen, int ko=EMPTY)
    {
		ParseFen(fen);
    }

    static Position initial_state()
    {
        return Position(BOARD_START_FEN, EMPTY);
    }

    std::string get_board()
    {
        return ToFen();
    }

	bool insufficient_material()
	{
		uint64_t all_pieces = bitboards[WHITE]|bitboards[BLACK];
		int piece_count = __builtin_popcountll(all_pieces);
		if( piece_count == 2 ) // KING vs KING
			return true;
		else if ( piece_count == 3 ) 
		{
			all_pieces &= ~bitboards[BLACK_KING];
			all_pieces &= ~bitboards[WHITE_KING];

			if( (all_pieces & bitboards[WHITE_KNIGHT]) != 0 )
				return true;

			if( (all_pieces & bitboards[BLACK_KNIGHT]) != 0 )
				return true;

			if( (all_pieces & bitboards[WHITE_BISHOP]) != 0 )
				return true;

			if( (all_pieces & bitboards[BLACK_BISHOP]) != 0 )
				return true;
		}
		else if ( piece_count == 4 ) 
		{
			all_pieces &= ~bitboards[BLACK_KING];
			all_pieces &= ~bitboards[WHITE_KING];

			if( bitboards[WHITE_BISHOP] != 0 && bitboards[BLACK_BISHOP] != 0 )
			{
				if ( (bitboards[WHITE_BISHOP] & WHITE_SQUARES) !=0 && (bitboards[BLACK_BISHOP] & WHITE_SQUARES) !=0 ) 
				{
					return true;
				}
				if ( (bitboards[WHITE_BISHOP] & ~WHITE_SQUARES) !=0 && (bitboards[BLACK_BISHOP] & ~WHITE_SQUARES) !=0 ) 
				{
					return true;
				}
			}
		}
		return false;
	}

	Position play_move(int fc, char color) // fc is the index in action space 
	{
		Position p = *this;

		p.side = color;
		int move = ActionToInternalMove(p.side, fc);
		p.Move(move);
		return p;
	}

	BYTE ParsePiece(char p)
	{
		std::size_t found = PIECE_CHARS.find(p);
		if (found!=std::string::npos)
			return found;
		else
			return EMPTY;
	}

	void ParseFen(string fen)
	{
		Reset();

		int pos = 0; 
		// position in string 
		// 8 rows of pieces 
		for(int row = 7; row >= 0; row--){ 
			while(fen[pos] == '/') pos++; 
			for(int col = 0; col < 8; col++){ 
				char c = fen[pos++]; 
			// if number skip ahead that many columns 
				if (c >= '1' && c <= '8'){ 
					col += c - '1'; 
				} else { 
			// find piece U8 
					uint64_t bit = 0x1ull << (row*8+col);
					BYTE piece = ParsePiece(c);
					board_array[row*8+col] = piece;
					if(piece != EMPTY)
						bitboards[piece] |= bit;
				} 
			} 
		} 
		for(int x=PAWN; x<=(KING|0x1); x++)
			bitboards[x & 1] |= bitboards[x];

		while(fen[pos] == ' ') 
			pos++;
		if(fen[pos]=='w')
			side = WHITE;
		else if(fen[pos]=='b')
			side = BLACK;
		else assert("error");
		pos++;
		
		while(fen[pos] == ' ') 
			pos++;
		
		while(fen[pos] != ' ') 
		{
			switch(fen[pos])
			{
				case 'K':
					castling_rights |= WHITE_KING_SIDE;
					break;
				case 'Q':
					castling_rights |= WHITE_QUEEN_SIDE;
					break;
				case 'k':
					castling_rights |= BLACK_KING_SIDE;
					break;
				case 'q':
					castling_rights |= BLACK_QUEEN_SIDE;
					break;
			}
			pos++;
		}
		//uncompleted here to read en passant

		while(fen[pos] == ' ') 
			pos++;

		string ep = "";
		while(fen[pos] != ' ') 
			ep.append(1, fen[pos++]);

		if(ep != "-")
		{
			int col = toupper(ep[0])-'A';
			int row = ep[1]-'1';
			en_passant = row*8+col;
		}
		else
			en_passant = NULL_SQUARE;

		hash = ZobristHash();
	}

	string ToFen()
	{
		string fen = "";
		for(int i=7; i>=0; i--)
		{
			int empty_count = 0;
			for(int j=0; j<8; j++)
			{
				if((board_array[i*8+j]&0xfe) == EMPTY)
					empty_count++;
				else
				{
					if(empty_count>0)
					{
						fen += to_string(empty_count);
						empty_count=0;
					}
					fen.append(1, PIECE_CHARS[board_array[i*8+j]]);
				}
			}
			if(empty_count>0)
				fen += to_string(empty_count);
			if(i > 0 )
				fen.append(1, '/');
		}

		fen += " ";
		if(side==WHITE)
			fen.append(1, 'w');
		else
			fen.append(1, 'b');
		fen += " ";
		
		string castling = "----";
		if(castling_rights & WHITE_KING_SIDE)
			castling[0] = 'K';
		if(castling_rights & WHITE_QUEEN_SIDE)
			castling[1] = 'Q';
		if(castling_rights & BLACK_KING_SIDE)
			castling[2] = 'k';
		if(castling_rights & BLACK_QUEEN_SIDE)
			castling[3] = 'q';

		if(castling=="----")
			castling = "-";
		fen += castling;
		fen += " ";

		if(en_passant == NULL_SQUARE)
			fen += "-";
		else
		{
			int row, col;
			row = en_passant / 8;
			col = en_passant % 8;

			fen.append(1, char('a'+col));
			fen.append(1, char('1'+row));
		}
		fen += " ";
		fen += to_string(fifty_moves);
		fen += " ";
		fen += "1"; //to_string(steps);
		return fen;
	}

    void Reset()
    {
		for(int i=0; i<14; i++)
			bitboards[i] = 0;

		for(int i=0; i<64; i++)
			board_array[i] = EMPTY;

        side = WHITE;
		fifty_moves = 0;
		en_passant = NULL_SQUARE;
		last_move = -1;
		castling_rights = NO_CASTLING_RIGHTS;	

		hash = ZobristHash();
    }
	
    char swap_colors(const char color)
    {
		return (color+1)%2;
    }

	int LegalMoves(action_space_of_a_piece actions[])
	{
		int moves[100];
		int move_count = 0;
		PossibleActions(side, moves, move_count);

		int count=0;
		BYTE *p = (BYTE *)&actions[0];
		for(int x=0; x<move_count; x++)
		{
			int m = moves[x];
			Position pos=*this; // restore current board position
			if(pos.Move(m))
			{
				if(!pos.FindCheckers(pos.side))
				{
					// legal move
					int index = InternalMoveToAction(pos.side, m); 
					p[index] = 1;
					count++;
				}
			}
			//UndoMove();
		}
		return count;
	}

    int PossibleActions(int side, int moves[], int& move_count)
    {
		assert(bitboards[KING|side] != 0);
		uint64_t all_pieces = bitboards[WHITE]|bitboards[BLACK];
		uint64_t move_mask;

		uint64_t attackers = FindCheckers(side);
		int checkcount = __builtin_popcountll(attackers);
		if(checkcount>1) // double check. can only move king. since king is in check, no castling is allowed 
		{	
			int pos = __builtin_ffsll(bitboards[KING | side])-1;
			move_mask = kingAttack_mask[pos] & ~bitboards[side];
			while(move_mask!=0)
			{
				int to = __builtin_ffsll(move_mask)-1;
				moves[move_count++] = pos+(to<<8);
				move_mask &= move_mask-1;
			}			
			return 0;
		}

		uint64_t m = bitboards[side];
		int enermy = (side+1)&0x1;
		int pos;
		while(m!=0)
		{
			pos = __builtin_ffsll(m)-1;  assert(pos>=0);

			move_mask = 0;
			switch(board_array[pos] & ~0x1) // zero the last bit 
			{
				case ROOK:
					move_mask = slider_attacks.RookAttacks(all_pieces, pos) & ~bitboards[side];
					break;
				case BISHOP:
					move_mask = slider_attacks.BishopAttacks(all_pieces, pos) & ~bitboards[side];
					break;
				case QUEEN:
					move_mask = slider_attacks.QueenAttacks(all_pieces, pos) & ~bitboards[side];
					break;
				case KNIGHT:
					move_mask = knightAttack_mask[pos] & ~bitboards[side];
					break;
				case PAWN:
					{
						uint64_t p = 0x1ull << pos;

						uint64_t push = (side==WHITE)?p<<8 : p>>8;
						push = push & ~all_pieces;

						uint64_t doublepush = (side==WHITE)?push<<8:push>>8;
						uint64_t rank = (side==WHITE)?RANK_4:RANK_5; 
						doublepush = doublepush & ~all_pieces & rank;

						uint64_t enermy_bits = bitboards[enermy];
						if(en_passant != NULL_SQUARE)
							enermy_bits |= (0x1ull << en_passant);
						move_mask = (pawnAttack_mask[side][pos] & enermy_bits) | push | doublepush;
					}
					break;
				case KING:
					{	
						move_mask = kingAttack_mask[pos] & ~bitboards[side];
						int enermy_king = __builtin_ffsll(bitboards[KING | (side+1)%2])-1;
						move_mask &= ~kingAttack_mask[enermy_king];
			
						//check white castling 
						if ((checkcount == 0) && ((bitboards[KING | side] == king_start_bit[side]) ) ) // king is at start position
						{
							bool allow_kingside_castling = ((castling_rights >> (side*2)) & KING_SIDE) != 0;
							bool allow_queenside_castling = ((castling_rights >> (side*2)) & QUEEN_SIDE) != 0;

							if (((all_pieces & kingside_mask[side]) == 0) //no pieces between king and right rook
							 && ((bitboards[ROOK | side] & right_rook_bit[side]) != 0) //right rook is at start position
		                     && allow_kingside_castling) 
							{
								if(!FindCheckers(side, king_start[side]+1))
								{
									moves[move_count++] = pos+((pos+2)<<8)+(KING_SIDE<<24);
								}
							}
							if (((all_pieces & queenside_mask[side]) == 0) //no pieces between king and left rook
							 && ((bitboards[ROOK | side] & left_rook_bit[side]) != 0) //left rook is at start position
							 && allow_queenside_castling)
							{
								if(!FindCheckers(side, king_start[side]-1))
									moves[move_count++] = pos+((pos-2)<<8)+(QUEEN_SIDE<<24);
							}
						}
					}
					break;
			}
			while(move_mask!=0)
			{
				int to = __builtin_ffsll(move_mask)-1;
				if((board_array[pos] & ~0x1) == PAWN)  //add promotions.
				{
					if(to>=56 || to<=7)
					{
						moves[move_count++] = pos+(to<<8)+(QUEEN<<16);
						moves[move_count++] = pos+(to<<8)+(ROOK<<16);  // promote to rook
						moves[move_count++] = pos+(to<<8)+(KNIGHT<<16);  // promote to knight
						moves[move_count++] = pos+(to<<8)+(BISHOP<<16);  // promote to bishop
					}
					else
						moves[move_count++] = pos+(to<<8);  // no promotion
				}
				else
					moves[move_count++] = pos+(to<<8);

				move_mask&=move_mask-1;
			}			
			m &= m -1;
		}

		return move_count;
    }
	uint64_t FindCheckers(int kingscolor, int pos=NULL_SQUARE)
	{
		if(pos==NULL_SQUARE)
			pos = __builtin_ffsll(bitboards[KING | kingscolor])-1;

		if(pos<0)
		{
			assert(pos>=0);
		}

		int enermy = (kingscolor + 1) & 0x1;

		uint64_t all = bitboards[WHITE] | bitboards[BLACK];

		uint64_t knightcheck = knightAttack_mask[pos] & bitboards[KNIGHT|enermy];

		uint64_t rookcheck = slider_attacks.RookAttacks(all, pos) & ~bitboards[kingscolor];
		rookcheck &= bitboards[ROOK|enermy] | bitboards[QUEEN|enermy];

		uint64_t bishopcheck = slider_attacks.BishopAttacks(all, pos) & ~bitboards[kingscolor];
		bishopcheck &= bitboards[BISHOP|enermy] | bitboards[QUEEN|enermy];

		uint64_t pawncheck = pawnAttack_mask[kingscolor][pos];  //if king can attack like pawn, then the positions that king can attack like a pawn is where the enermy pawns can attack king. 
		pawncheck &= bitboards[PAWN|enermy];

		uint64_t kingcheck = kingAttack_mask[pos];
		kingcheck &= bitboards[KING|enermy];

		uint64_t attackers = knightcheck | rookcheck | bishopcheck | pawncheck | kingcheck;

		return attackers;
	}

	bool Move(int move)
	{
		int from = move&0xff;
		int to = (move>>8)&0xff;

		BYTE promotion = (move>>16)&0xff;

		BYTE from_row = from >> 3;		
		BYTE to_row = to >> 3;		

		uint64_t bit_from = (0x1ull<<from);
		uint64_t bit_to = (0x1ull<<to);
		
		BYTE last_en_passant = en_passant;
		en_passant = NULL_SQUARE;

		BYTE from_piece = board_array[from];
		BYTE target_piece = board_array[to];

		int enermy = (side+1)&0x1;

		hash ^= hash_piece[from][side][from_piece];
		hash ^= hash_piece[from][side][EMPTY];
		if(last_en_passant != NULL_SQUARE)
			hash ^= hash_enpassant[last_en_passant%8];

		switch(from_piece&~side)
		{
			case ROOK:
				bitboards[ROOK | side] &= ~bit_from;
				bitboards[ROOK | side] |= bit_to;

				bitboards[side] &= ~bit_from;
				bitboards[side] |= bit_to;

				board_array[to] = board_array[from];
				board_array[from] = EMPTY;

				hash ^= hash_castling[castling_rights];
				if(from==left_rook[side])
					castling_rights &= ~(QUEEN_SIDE << (side*2));  // allow_white_queenside_castling = false;
				else if(from==right_rook[side])
					castling_rights &= ~(KING_SIDE << (side*2)); //allow_white_kingside_castling = false;

				hash ^= hash_piece[to][side][target_piece];
				hash ^= hash_piece[to][side][from_piece];
				hash ^= hash_castling[castling_rights];
				break;
			case BISHOP:
			case QUEEN:
			case KNIGHT:
				bitboards[from_piece] &= ~bit_from;
				bitboards[from_piece] |= bit_to;

				bitboards[side] &= ~bit_from;
				bitboards[side] |= bit_to;

				board_array[to] = board_array[from];
				board_array[from] = EMPTY;

				hash ^= hash_piece[to][side][target_piece];
				hash ^= hash_piece[to][side][from_piece];
				break;
			case PAWN:
			{
				bitboards[PAWN | side] &= ~bit_from;
				if(promotion==0)
				{
					if((from>=48&&side==WHITE) || (from <=15&&side==BLACK))
						promotion=QUEEN;
					else
						promotion=PAWN;
				}
				bitboards[promotion | side] |= bit_to; //if no promotion, promotion = PAWN

				bitboards[side] &= ~bit_from;
				bitboards[side] |= bit_to;

				board_array[to] = promotion | side;
				board_array[from] = EMPTY;

				if(promotion==PAWN)
				{
					if(from_row==en_passant_start_ranks[side] && to_row == en_passant_end_ranks[side]) // if pawn moves from start position and pushes two steps, it is an en passant
						en_passant = to+offset[side];
					else
					{
						if(last_en_passant == to)
						{
							bitboards[PAWN | enermy] &= ~(0x1ull << (last_en_passant+offset[side]));
							bitboards[enermy] &= ~(0x1ull << (last_en_passant+offset[side]));
							board_array[last_en_passant+offset[side]] = EMPTY;
						}
						en_passant = NULL_SQUARE;
					}
				}
				hash ^= hash_piece[to][side][target_piece];
				hash ^= hash_piece[to][side][promotion | side];
				if(en_passant != NULL_SQUARE)
					hash ^= hash_enpassant[en_passant%8];
				break;
			}
			case KING:
			{  
				hash ^= hash_castling[castling_rights];

				BYTE cas = 0; // = (move>>24)&0xff;
				if((to-from)==2)
					cas = KING_SIDE;
				else if((to-from)==-2)
					cas = QUEEN_SIDE;

				bitboards[KING | side] = bit_to; // there is only 1 king 
				bitboards[side] &= ~bit_from;
				bitboards[side] |= bit_to;

				board_array[to] = board_array[from];
				board_array[from] = EMPTY;
				if(cas!=0)  //castling 
				{
					if(cas==KING_SIDE)
					{
						bitboards[ROOK | side] &= ~right_rook_bit[side];
						bitboards[ROOK | side] |= (right_rook_bit[side]>>2);
						bitboards[side] &= ~right_rook_bit[side];
						bitboards[side] |= (right_rook_bit[side]>>2);

						board_array[to-1] = ROOK|side;
						board_array[from+3] = EMPTY;

						hash ^= hash_piece[from+3][side][ROOK|side];
						hash ^= hash_piece[from+3][side][EMPTY];
						hash ^= hash_piece[to-1][side][EMPTY];
						hash ^= hash_piece[to-1][side][ROOK|side];
					}
					else
					{
						bitboards[ROOK | side] &= ~left_rook_bit[side];
						bitboards[ROOK | side] |= (left_rook_bit[side]<<3);
						bitboards[side] &= ~left_rook_bit[side];
						bitboards[side] |= (left_rook_bit[side]<<3);

						board_array[to+1] = ROOK|side;
						board_array[from-4] = EMPTY;

						hash ^= hash_piece[from-4][side][ROOK|side];
						hash ^= hash_piece[from-4][side][EMPTY];
						hash ^= hash_piece[to+1][side][EMPTY];
						hash ^= hash_piece[to+1][side][ROOK|side];
					}
				}
				//as long as king has moved, castling rights is removed.
				castling_rights &= ~(KING_SIDE << (side*2));
				castling_rights &= ~(QUEEN_SIDE << (side*2));

				hash ^= hash_piece[to][side][target_piece];
				hash ^= hash_piece[to][side][from_piece];
				hash ^= hash_castling[castling_rights];
				break;
			}
			default:
				assert("no piece to move!");
		}

		if(target_piece != EMPTY)
		{
			bitboards[target_piece] &= ~bit_to;
			bitboards[target_piece&0x1] &= ~bit_to;
		}

		if(target_piece!=EMPTY || (from_piece&~0x1) == PAWN)
			fifty_moves = 0; // reset to 0 if pawn has moved or capture has been made
		else
			fifty_moves++;

		last_move = to;

		/*if(hash != ZobristHash())
		{		
			assert(hash == ZobristHash());
		}*/
		hash = ZobristHash();
		assert(bitboards[KING|side] != 0);
		return true;
	}

    std::vector<int> legal_actions(char color) // color is the same as this->side. redundant argument
	{
        std::vector<int> selected_action; // = new std::vector<int>(); // vector with all moves
		selected_action.reserve(50);

		int moves[100];
		std::fill(moves, moves+100, 0);
		int move_count = 0;
		PossibleActions(side, moves, move_count);

		int count=0;
		//BYTE *p = (BYTE *)&actions[0];
		for(int x=0; x<move_count; x++)
		{
			int m = moves[x];
			Position pos=*this; // restore current board position
			if(pos.Move(m))
			{
				if(!pos.FindCheckers(pos.side))
				{
					// legal move
					selected_action.push_back(InternalMoveToAction(color, m));
					count++;
				}
			}
			//UndoMove();
		}

		/*action_space_of_a_piece actions[64];	// great! this is already an ordered vector 
		int legal_moves = LegalMoves(actions);*/

        //std::vector<int> *selected_action = new std::vector<int>(); // vector with all moves
		//selected_action->reserve(50);

		/*BYTE *p = (BYTE *)&actions[0];
		for(int i=0; i<sizeof(actions); i++)
			if(p[i] == 1)
				selected_action->push_back(i);*/
		
		return selected_action;
	}

	uint64_t ZobristHash()
	{
		uint64_t _hash = 0;
		for(int sq=0; sq<64; sq++)
		{
			_hash ^= hash_piece[sq][this->side][this->board_array[sq]];
		}
		_hash ^= hash_castling[this->castling_rights];
		if(this->en_passant < NULL_SQUARE)
			_hash ^= hash_enpassant[this->en_passant%8];	

		return _hash;
	}
};

void InitializePawnMoves()
{
	for(int pos=0; pos<64; pos++)
	{
		pawnAttack_mask[WHITE][pos] = 0;
		pawnAttack_mask[BLACK][pos] = 0;

		uint64_t bit = 0x1ull<<pos;

		pawnAttack_mask[WHITE][pos] = (bit & ~FILE_A)<<7;
		pawnAttack_mask[WHITE][pos] |= (bit & ~FILE_H)<<9;

		pawnAttack_mask[BLACK][pos] = (bit & ~FILE_A)>>9;
		pawnAttack_mask[BLACK][pos] |= (bit & ~FILE_H)>>7;
	}
}

void InitializeKnightAttacks()
{
	int ox[] = { 2, 1,-1,-2,-2,-1, 1, 2};
	int oy[] = { 1, 2, 2, 1,-1,-2,-2,-1};
	for(int pos=0; pos<64; pos++)
	{
		uint64_t moves = 0;
		for(int d=0; d<8; d++)
		{
			int r = pos / 8;
			int c = pos % 8;
			c += ox[d];
			r += oy[d];
	
			if( r>=0 && r<8 && c>=0 && c<8 )
			{
				moves |= (0x1ull << (r*8+c));
			}
		}
		knightAttack_mask[pos] = moves;
	}
}

void InitializeKingAttacks()
{
	int ox[] = { 1, 1, 0,-1,-1,-1, 0, 1};
	int oy[] = { 0, 1, 1, 1, 0,-1,-1,-1};
	for(int pos=0; pos<64; pos++)
	{
		uint64_t moves = 0;
		for(int d=0; d<8; d++)
		{
			int r = pos / 8;
			int c = pos % 8;
			c += ox[d];
			r += oy[d];
	
			if( r>=0 && r<8 && c>=0 && c<8 )
			{
				moves |= (0x1ull << (r*8+c));
			}
		}
		kingAttack_mask[pos] = moves;
	}
}

void InitializeMasks()
{
	InitializePawnMoves();
	InitializeKnightAttacks();
	InitializeKingAttacks();
	slider_attacks.Initialize();
}

void InitZobrist()
{
    std::mt19937_64 generator(2765481ull);

	std::uniform_int_distribution<std::mt19937_64::result_type> dist;

	for(int sq=0; sq<64; sq++)
		for(int c=0; c<2; c++)
			for(int p=0; p<14; p++)
			{
				hash_piece[sq][c][p] = dist(generator);
			}

	for(int i=0; i<16; i++)
		hash_castling[i] = dist(generator);

	for(int i=0; i<8; i++)
		hash_enpassant[i] = dist(generator);
}

#endif
