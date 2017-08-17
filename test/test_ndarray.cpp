#include "catch.hpp"
#include "../ndarray.h"
#include "../ndarray.cpp"

TEST_CASE("NDArray::add") {
  // empty + empty = empty
  REQUIRE(NDArray().add(NDArray()) == NDArray());
  // empty + something = something
  REQUIRE(NDArray().add(NDArray({1}, {1})) == NDArray({1}, {1}));
  // 1d add
  REQUIRE(NDArray({1}, {1}).add(NDArray({1}, {7})) == NDArray({1}, {8}));
  
  SECTION("Nd add") {
    GIVEN("Same shape NDArray's a and b") {
      NDArray a({2, 1, 3}, {1, 2, 3, 4, 5, 6});
      NDArray b({2, 1, 3}, {2, 3, 5, 7, 9, 11});
      
      NDArray c({2, 1, 3}, {3, 5, 8, 11, 14, 17});
      REQUIRE(a.add(b) == c);
      REQUIRE(a.add_(b) == c);
      REQUIRE(a == c);
    }

    GIVEN("Different shape NDArray's a and b") {
      NDArray a({2, 1, 3}, {1, 2, 3, 4, 5, 6});
      NDArray b({3}, {2, 3, 5});

      NDArray c({2, 1, 3}, {3, 5, 8, 6, 8, 11});
      REQUIRE(a.add(b) == c);
      
      b.reshape({1, 3});
      REQUIRE(a.add(b) == c);
      
      b.reshape({1, 1, 3});
      REQUIRE(a.add(b) == c);
      
      REQUIRE(a.add_(b) == c);
      REQUIRE(a == c);
    }
  }
}
