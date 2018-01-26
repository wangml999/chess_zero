//
//  main.cpp
//  t1cpp
//
//  Created by Minglei Wang on 7/01/2018.
//  Copyright © 2018 Minglei Wang. All rights reserved.
//

#include "fast_go.h"
#include "tree.h"
#include <cassert>
#include <thread>

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
    Tree  t;
    
    assert(t.root->parent == nullptr);
    
    Board board;
    double value;
    
    for(int i=0; i<n; i++)
        t.search(board, value, 800);
}

void play(bool verbose=true)
{
    Tree t;
    Board board;
    
    //board.position = Position("1101111111011111011111111", NONE);
    //board.current = WHITE;
    while(board.status()==-1)
    {
        double value;

        auto t1 = std::chrono::high_resolution_clock::now();
        int n = t.search(board, value, 800);
        auto t2 = std::chrono::high_resolution_clock::now();
        auto diff = t2-t1;
        std::chrono::nanoseconds ns = std::chrono::duration_cast<std::chrono::nanoseconds>(diff);
        
        if(verbose)
        {
            std::cout << "step: " << board.steps << ", current: " << board.current << ", action: ";
        
            if(n==NN)
                std::cout << "pass";
            else
                std::cout << n;
        
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
        t.change_root(t.root->children[n]);
    }

    std::cout << "winner: " << board.status() << " score: " << board.score() << " steps: " << board.steps << std::endl;
    return;
}

int main(int argc, const char * argv[]) {
    //Position p = Position("0111111110111111111211120", NONE);
    //vector<int> s = p.possible_moves(WHITE);

    auto t1 = std::chrono::high_resolution_clock::now();

    //tree_test(50);
    //fast_go_test();
    play();

    auto t2 = std::chrono::high_resolution_clock::now();
    auto diff = t2-t1;
    std::chrono::nanoseconds ns = std::chrono::duration_cast<std::chrono::nanoseconds>(diff);
    std::cout << ns.count()*1.0/1000000000 << " seconds" << std::endl;
    
    return 0;
}
