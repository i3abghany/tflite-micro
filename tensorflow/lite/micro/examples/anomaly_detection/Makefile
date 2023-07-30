CFLAGS=-std=c++17 -ggdb
all: example.cpp kws_input_data.cc kws_model_data.cc kws_model_settings.cc
# 	g++ $^ $(CFLAGS) ./libtensorflow-lite.a -I../tflite-micro/ -I../tflite_build/flatbuffers/include/ -I../tflite_build/gemmlowp 
	g++ $^ $(CFLAGS) ./libtensorflow-lite.a -I../tflite-micro/ -I../flatbuffers-2.0.6/include -I../gemmlowp/