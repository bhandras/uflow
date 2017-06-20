CXX := clang++
CXXFLAGS := -std=c++11 -Wall
TARGET := test

all:
	$(CXX) $(CXXFLAGS) *.cpp -o $(TARGET)

test: all
	./$(TARGET)

clean:
	rm -rf $(TARGET)

