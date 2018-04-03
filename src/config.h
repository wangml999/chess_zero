#ifndef __CONFIG_H
#define __CONFIG_H

#define WN 5

#if WN==19
    #define KOMI 7.5
#elif WN==13
    #define KOMI 5.5
#elif WN==9
    #define KOMI 5.5
#else
    #define KOMI 0.0
#endif


#endif
