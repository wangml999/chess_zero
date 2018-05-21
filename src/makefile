CC=g++
tensorflow=/home/minglei/Downloads/tensorflow/
bazel-genfiles=$(tensorflow)bazel-genfiles/ 
package-include=/usr/local/lib/python3.5/dist-packages/tensorflow/include/ 
eigen=$(tensorflow)third_party/eigen3/ 
nsync=$(tensorflow)bazel-tensorflow/external/nsync/public/ 
bazel-bin=$(tensorflow)bazel-bin/tensorflow/

go_zero: go_zero
	$(CC) -Wall -O3 main.cpp -o ../bin/go_zero -std=c++11 -I$(tensorflow) -I$(bazel-genfiles) -I$(package-include) -I$(eigen) -I$(nsync) -I$(bazel-bin) -L$(bazel-bin) -ltensorflow_cc -ltensorflow_framework -pthread

go_zero_t: go_zero_t
	$(CC) -Wall -O3 main.cpp -o ../bin/go_zero_t -std=c++11 -I$(tensorflow) -I$(bazel-genfiles) -I$(package-include) -I$(eigen) -I$(nsync) -I$(bazel-bin) -L$(bazel-bin) -ltensorflow_cc -ltensorflow_framework -pthread

go_zero_d: go_zero_d
	$(CC) -g -Wall main.cpp -o ../bin/go_zero_d -std=c++11 -I$(tensorflow) -I$(bazel-genfiles) -I$(package-include) -I$(eigen) -I$(nsync) -I$(bazel-bin) -L$(bazel-bin) -ltensorflow_cc -ltensorflow_framework -pthread

go_zero_p: go_zero_p
	$(CC) -Wall -O3 -pg main.cpp -o ../bin/go_zero_p -std=c++11 -I$(tensorflow) -I$(bazel-genfiles) -I$(package-include) -I$(eigen) -I$(nsync) -I$(bazel-bin) -L$(bazel-bin) -ltensorflow_cc -ltensorflow_framework -pthread
