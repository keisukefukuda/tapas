
#include <utility>
#include <set>

#include <tapas/common.h>
#include <tapas/test.h>
#include <tapas/map.h>
#include <tapas/util.h>

SETUP_TEST;

template<class T> using V = std::vector<T>;

void Test_SortByKeys() {
  V<char> vals {'2', '3', '0', '4', '5', '1'};
  V<int>  keys { 2,   3,   0,   4,   5,   1};
  V<char> ans_vals  {'0', '1', '2', '3', '4', '5'};
  V<int>  ans_keys  { 0,   1,   2,   3,   4,   5 };

  tapas::SortByKeys(keys, vals);
  ASSERT_EQ(ans_vals, vals);
  ASSERT_EQ(ans_keys, keys);
}

void Test_IfSetIsSorted() {
  // Test if std::set (not unordered_set) iterates over the elements in a sorted fasion.
  srand(0);
  const int N = 100;
  std::set<std::pair<int, double>> S;

  for (int i = 0; i < N; i++) {
    S.insert(std::make_pair(rand(), drand48()));
  }

  std::vector<int> v;
  for (auto &&itr : S) {
    v.push_back(itr.first);
  }

  // check if v is sorted.
  for (size_t i = 0; i < v.size()-1; i++) {
    ASSERT_TRUE(v[i] <= v[i+1]);
  }
}

template<class SetType>
void Test_Diff() {
  SetType a = {1,2, 3, 4};
  SetType b = {2, 3};
  SetType c = tapas::util::SetDiff(a, b);
  SetType d = tapas::util::SetDiff(b, a);

  ASSERT_TRUE(c.size() == 2);
  ASSERT_TRUE(c.count(1) == 1);
  ASSERT_TRUE(c.count(2) == 0);
  ASSERT_TRUE(c.count(3) == 0);
  ASSERT_TRUE(c.count(4) == 1);

  ASSERT_TRUE(d.size() == 0);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  Test_IfSetIsSorted();
  Test_SortByKeys();
  Test_Diff<std::set<int>>();
  Test_Diff<std::unordered_set<int>>();

  TEST_REPORT_RESULT();
  
  MPI_Finalize();
  return (TEST_SUCCESS() ? 0 : 1);
}
