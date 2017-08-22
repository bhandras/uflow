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
      
      b.reshape({3, 1});
      NDArray d({2, 3, 3},
          {3, 4, 5, 4, 5, 6, 6, 7, 8, 6, 7, 8, 7, 8, 9, 9, 10, 11});
      REQUIRE(a.add(b) == d);
      
      REQUIRE(a.add_(b) == d);
      REQUIRE(a == d);
    }
  }
}

TEST_CASE("NDArray::dot") {
  REQUIRE(NDArray().dot(NDArray()) == NDArray());

  GIVEN("Incompatible shapes") {
    NDArray a({1, 3});

    CHECK_THROWS(a.dot(NDArray()));
    CHECK_THROWS(a.dot(NDArray({2})));
    CHECK_THROWS(a.dot(NDArray({3})));
    CHECK_THROWS(a.dot(NDArray({1, 2})));
    CHECK_THROWS(a.dot(NDArray({2, 3})));
    CHECK_THROWS(a.dot(NDArray({1, 1, 3})));
  }

  GIVEN("Compatible shapes") {
    NDArray a({3}, {1, 2, 3});
    NDArray b({3}, {4, 5, 6});
    NDArray c({1}, {32});

    REQUIRE(a.dot(b) == c);

    a.unsqueeze(0);
    b.unsqueeze(0);
    REQUIRE(a.dot(b) == c);

    a.unsqueeze(0);
    b.unsqueeze(0);
    REQUIRE(a.dot(b) == c);
  }
}

TEST_CASE("NDArray::mm") {
  GIVEN("Incompatible shapes") {
    CHECK_THROWS(NDArray().mm(NDArray()));
    CHECK_THROWS(NDArray({1}).mm(NDArray()));
    CHECK_THROWS(NDArray({1}).mm(NDArray({1})));
    
    CHECK_THROWS(NDArray({1}).mm(NDArray({1, 1})));
    CHECK_THROWS(NDArray({1, 1}).mm(NDArray({1})));

    CHECK_THROWS(NDArray({1, 2}).mm(NDArray({3, 4})));
    CHECK_THROWS(NDArray({1, 2, 3}).mm(NDArray({3, 3})));
    CHECK_THROWS(NDArray({1, 2, 3}).mm(NDArray({1, 3, 2})));
  }

  GIVEN("Compatible shapes") {
    REQUIRE(NDArray({1, 1}, {1}).mm(NDArray({1, 1}, {3}))
        == NDArray({1, 1}, {3}));

    REQUIRE(NDArray({1, 3}, {1, 2, 3}).mm(NDArray({3, 1}, {4, 5, 6}))
        == NDArray({1, 1}, {32}));

    REQUIRE(NDArray({2, 2}, {1, 2, 3, 4}).mm(NDArray({2, 2}, {5, 6, 7, 8}))
        == NDArray({2, 2}, {19, 22, 43, 50}));

    REQUIRE(NDArray({2, 2}, {1, 2, 3, 4}).mm(NDArray({2, 3}, {5, 6, 7, 8, 9, 10}))
        == NDArray({2, 3}, {21, 24, 27, 47, 54, 61}));

  }
}

TEST_CASE("NDArray::bmm") {
  /*
   * Valid combinations:
   * - (m, n) * (n, k) = (m, k)
   * - (m, n) * (b, n, k)  and  (b, m, n) * (n, k) = (b, m, k)
   * - (1, m, n) * (b, n, k)  and  (b, m, n) * (1, n, k) = (1, m, k)
   * - (b, m, n) * (b, n, k) = (b, m, k)
   */

  GIVEN("Incompatible shapes") {
    CHECK_THROWS(NDArray().bmm(NDArray()));
    CHECK_THROWS(NDArray({1}).bmm(NDArray()));
    CHECK_THROWS(NDArray({1}).bmm(NDArray({1})));

    CHECK_THROWS(NDArray({1}).bmm(NDArray({1, 1})));
    CHECK_THROWS(NDArray({1, 1}).bmm(NDArray({1})));

    CHECK_THROWS(NDArray({1, 2}).bmm(NDArray({2, 4})));
  }

  GIVEN("Compatible shapes") {
    REQUIRE(NDArray({1, 1, 1}, {1}).bmm(NDArray({1, 1}, {3}))
        == NDArray({1, 1, 1}, {3}));
    REQUIRE(NDArray({2, 1, 1}, {1, 2}).bmm(NDArray({1, 1}, {3}))
        == NDArray({2, 1, 1}, {3, 6}));
    
    REQUIRE(NDArray({1, 1}, {1}).bmm(NDArray({1, 1, 1}, {3}))
        == NDArray({1, 1, 1}, {3}));
    REQUIRE(NDArray({1, 1}, {3}).bmm(NDArray({2, 1, 1}, {3, 4}))
        == NDArray({2, 1, 1}, {9, 12}));

    REQUIRE(NDArray({2, 1, 1}, {1}).bmm(NDArray({1, 1, 1}, {3}))
        == NDArray({2, 1, 1}, {3}));
    REQUIRE(NDArray({1, 1, 1}, {1}).bmm(NDArray({2, 1, 1}, {3}))
        == NDArray({2, 1, 1}, {3}));


    REQUIRE(NDArray({1, 1, 3}, {1, 2, 3}).bmm(NDArray({3, 1}, {4, 5, 6}))
        == NDArray({1, 1, 1}, {32}));

    REQUIRE(NDArray({3, 2, 2}, {1, 2, 3, 4}).bmm(NDArray({2, 2}, {5, 6, 7, 8}))
        == NDArray({3, 2, 2}, {19, 22, 43, 50}));

    REQUIRE(NDArray({2, 2}, {1, 2, 3, 4}).bmm(NDArray({4, 2, 3}, {5, 6, 7, 8, 9, 10}))
        == NDArray({4, 2, 3}, {21, 24, 27, 47, 54, 61}));
  }
}

TEST_CASE("NDArray::reduce_max") {
}

TEST_CASE("NDArray::reduce_sum") {
}

TEST_CASE("NDArray::argmax") {
}

TEST_CASE("NDArray::squeeze, NDArray::unsqueeze") {
}

TEST_CASE("NDArray::get, NDArray::set") {
}

TEST_CASE("NDArray::max_filter") {
}

TEST_CASE("NDArray::minimum") {
}

