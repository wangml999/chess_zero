#ifndef __CONFIG_H
#define __CONFIG_H

#define WN 5

#if WN==5
    #define TENSORFLOW_BATCH_SIZE 8
    #define KOMI 0.5
#elif WN==9
    #define TENSORFLOW_BATCH_SIZE 8
    #define KOMI 5.5
#elif WN==13
    #define TENSORFLOW_BATCH_SIZE 8
    #define KOMI 5.5
#elif WN==19
    #define TENSORFLOW_BATCH_SIZE 16
    #define KOMI 7.5
#endif

#define MCTS_REPS 800

#define CPUCT 0.9
#define VIRTUAL_LOSS  10

#define ADD_DIRICHLET_NOISE
#define ALPHA 0.3
//#define TENSORFLOW_BENCHMARK

#endif
