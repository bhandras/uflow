CXX := clang++
CXXFLAGS := -std=c++1y -Wall
TARGET := uflow

all:
	$(CXX) $(CXXFLAGS) *.cpp -o $(TARGET)

test: all
	./$(TARGET)

clean:
	rm -rf $(TARGET)

