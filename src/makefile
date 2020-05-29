CC=g++
tensorflow=/home/minglei/Downloads/tensorflow-1.14.0/
bazel-genfiles=$(tensorflow)bazel-genfiles/ 
package-include=/usr/local/lib/python3.5/dist-packages/tensorflow/include/ 
eigen=$(tensorflow)third_party/eigen3-c/ 
#eigen=/usr/local/lib/python3.5/dist-packages/tensorflow/include/tensorflow/include/external/eigen_archive/
nsync=$(tensorflow)bazel-tensorflow/external/nsync/public/ 
bazel-bin=$(tensorflow)bazel-bin/tensorflow/
tftrt-path=$(bazel-bin)compiler/tf2tensorrt/python/ops/
TFT=$(bazel-bin)compiler/tf2tensorrt/
CUDA=/usr/local/cuda-10.1/targets/x86_64-linux/include/
TT=/home/minglei/Downloads/workspace/tensorrt/samples/common
MAGIC_BITS=../lib/magic-bits/src

go_zero: go_zero
	$(CC) -Wall -O3 main.cpp -o ../bin/go_zero -std=c++11 -I$(tensorflow) -I$(bazel-genfiles) -I$(package-include) -I$(eigen) -I$(nsync) -I$(bazel-bin) -L$(bazel-bin) -ltensorflow_cc -ltensorflow_framework -pthread

go_zero_t: go_zero_t
	$(CC) -Wall -O3 main.cpp -o ../bin/go_zero_t -std=c++11 -I$(tensorflow) -I$(bazel-genfiles) -I$(package-include) -I$(eigen) -I$(nsync) -I$(bazel-bin) -L$(bazel-bin) -ltensorflow_cc -ltensorflow_framework -pthread

go_zero_d: go_zero_d
	$(CC) -g -Wall main.cpp -o ../bin/go_zero_d -std=c++11 -I$(tensorflow) -I$(bazel-genfiles) -I$(package-include) -I$(eigen) -I$(nsync) -I$(bazel-bin) -L$(bazel-bin) -ltensorflow_cc -ltensorflow_framework -pthread

go_zero_p: go_zero_p
	$(CC) -Wall -O3 -pg main.cpp -o ../bin/go_zero_p -std=c++11 -I$(tensorflow) -I$(bazel-genfiles) -I$(package-include) -I$(eigen) -I$(nsync) -I$(bazel-bin) -L$(bazel-bin) -ltensorflow_cc -ltensorflow_framework -pthread

test_r: test.cpp bit_board.h chess_board.h
	$(CC) -Wall -O3 test.cpp slider_attacks.cpp -o test_r -std=c++11

test_d: test.cpp bit_board.h chess_board.h
	$(CC) -Wall -g test.cpp slider_attacks.cpp -o test_d -std=c++11

test_p: test.cpp bit_board.h chess_board.h
	$(CC) -Wall -pg test.cpp slider_attacks.cpp -o test_p -std=c++11

chess: chess_d chess_r

chess_d: test.cpp bit_board.h chess_board.h tree.h network.h
	$(CC) -Wall -g test.cpp $(MAGIC_BITS)/slider_attacks.cpp -o ../bin/chess_d -std=c++11 -I$(MAGIC_BITS) -I$(tensorflow) -I$(bazel-genfiles) -I$(eigen) -I$(package-include) -I$(nsync) -I$(bazel-bin) -I$(TT) -L$(bazel-bin) -ltensorflow_cc -ltensorflow_framework -lnvinfer -lpthread -DDEBUG

chess_r: test.cpp bit_board.h chess_board.h tree.h network.h
	$(CC) -Wall -O3 test.cpp $(MAGIC_BITS)/slider_attacks.cpp -o ../bin/chess_r -std=c++11 -I$(MAGIC_BITS) -I$(tensorflow) -I$(bazel-genfiles) -I$(eigen) -I$(package-include) -I$(nsync) -I$(bazel-bin) -I$(TT) -L$(bazel-bin) -ltensorflow_cc -ltensorflow_framework -lnvinfer -lpthread
#	$(CC) -Wall -O3 test.cpp $(MAGIC_BITS)/slider_attacks.cpp -o ../bin/chess_r -std=c++11 -I$(MAGIC_BITS) -I$(tensorflow) -I$(bazel-genfiles) -I$(eigen) -I$(package-include) -I$(nsync) -I$(bazel-bin) -L$(bazel-bin) -ltensorflow_cc -ltensorflow_framework -pthread -L$(tftrt-path) -ltftrt -L$(TFT) -ltrt_op_kernels -ltrt_resources

chess_p: test.cpp bit_board.h chess_board.h tree.h network.h
	$(CC) -Wall -pg test.cpp $(MAGIC_BITS)/slider_attacks.cpp -o ../bin/chess_p -std=c++11 -I$(MAGIC_BITS) -I$(tensorflow) -I$(bazel-genfiles) -I$(package-include) -I$(eigen) -I$(nsync) -I$(bazel-bin) -L$(bazel-bin) -ltensorflow_cc -ltensorflow_framework -pthread

chess_t: test.cpp bit_board.h chess_board.h tree.h network.h
	$(CC) -Wall -O3 test.cpp $(MAGIC_BITS)/slider_attacks.cpp -o ../bin/chess_t -std=c++11 -I$(MAGIC_BITS) -I$(tensorflow) -I$(bazel-genfiles) -I$(eigen) -I$(package-include) -I$(nsync) -I$(bazel-bin) -I$(TT) -L$(bazel-bin) -ltensorflow_cc -ltensorflow_framework -lnvinfer -lpthread

