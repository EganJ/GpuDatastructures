#include "datastructures.h"
#include <iostream>

void testUnionFind()
{
  const int num_elements = 10;
  gpuds::unionfind::UnionFind uf(num_elements);

  std::vector<int> a = {1, 2, 3, 6, 8};
  std::vector<int> b = {2, 4, 4, 7, 10};

  std::cout << "Merging pairs:" << std::endl;

  uf.massMerge(a, b);

  std::vector<int> classes = uf.getClasses();
  for (int i = 1; i <= num_elements; ++i)
  {
    std::cout << "Element " << i << " is in class " << classes[i] << std::endl;
  }
}

int main()
{
  testUnionFind();
  return 0;
}
