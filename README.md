# go_zero
This is a study project. The aim is to implement an AI go program according to DeepMind's AlphaGo Zero paper. I don't have that much computer power that google has so this program would start from 5x5 board. But the idea of the algorithm would be the same.

## what's included
* a player written in c++11. 
* a trainer written in python 3. 

## What's required
* compile tensorflow from source code. following the link to install tensorflow https://www.tensorflow.org/install/install_sources
* I am using r1.6. later version has not been tested. one addtional step after compile tensorflow as per the instruction is to compile the //tensorflow:libtensorflow.so which will be used in the c++ player. 
* The tensorflow include and lib paths in the makefile need to be updated accordingly. or update in xcode if compiling on mac os x.

## How to run it
* self play  
```shell
./go_zero -n 0 -s 
```
  the player will automatically update the player's network from [n]x[n]\models directory. n is the macro WN defined in the fast_go.h.
  the player plays with itself forever and every 100 games it generates data a file in [n]x[n]\data directory. 
  
* evaluation
```shell
./go_zero -m1 metagraph-0000xxxx -m2 metagraph-0000xxxx -n 400 -s
```
  this can be to evaluate two networks and plays 400 games without displaying board of each step. 
  
* play with human
```shell
./go_zero -m1 metagraph-0000xxxx -m2 human 
```
  this will play 1 game with human. 
 
* training

run train.py in python 3 will load the latest saved model and start looking for the self play data files. it will then randomly sample the moves and batch into mini batches to train the network. after x number of steps, it will compare with the latest model by running two passes evaluation to get the better network. the better network is saved as latest model or discarded. 
