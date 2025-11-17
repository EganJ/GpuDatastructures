#include <iostream>
#include <iomanip>

#include "datastructures.h"
#include "parser.h"

void testUnionFindSmall()
{
  const int num_elements = 10;
  gpuds::unionfind::UnionFind uf(num_elements);

  std::vector<int> a = {1, 2, 3, 6, 8};
  std::vector<int> b = {2, 4, 4, 7, 10};

  std::cout << "Before merging: " << std::endl;
  std::vector<int> preclasses = uf.getClasses();
  for (int i = 0; i < preclasses.size(); ++i)
  {
    std::cout << "Element " << i + 1 << " is in class " << preclasses[i] << std::endl;
  }

  std::cout << "Merging pairs:" << std::endl;

  uf.massMerge(a, b);

  std::vector<int> classes = uf.getClasses();
  for (int i = 0; i < classes.size(); ++i)
  {
    std::cout << "Element " << i + 1 << " is in class " << classes[i] << std::endl;
  }
}

void testUnionFindLarge()
{
  std::vector<int> a, b, expected;
  gpuds::unionfind::read_from_file("unionfindtestdata", a, b, expected);
  int num_elements = expected.size();
  std::cout << "Testing UnionFind with " << num_elements << " elements and " << a.size() << " merges." << std::endl;
  gpuds::unionfind::UnionFind uf(num_elements);
  uf.massMerge(a, b);
  std::cout << "Merges done, flattening classes..." << std::endl;
  uf.flatten();
  std::vector<int> classes = uf.getClasses();
  int num_errors = 0;
  for (int i = 0; i < num_elements; ++i)
  {
    if (classes[i] != expected[i])
    {
      if (num_errors < 100)
      {
        std::cout << "Error: Element " << i + 1 << " is in class " << classes[i] << " but expected " << expected[i] << std::endl;
      }
      num_errors++;
    }
  }
  if (num_errors == 0)
  {
    std::cout << "All classes match expected output!" << std::endl;
  }
  else
  {
    std::cout << num_errors << " total errors found." << std::endl;
    std::cout << "Printing head of expected vs actual classes:" << std::endl;
    std::cout << " Index Expected Actual" << std::endl;
    for (int i = 0; i < std::min(100, num_elements); ++i)
    {
      std::cout << std::setw(5) << i + 1 << ": " << std::setw(5) << expected[i] << std::setw(5) << classes[i] << std::endl;
    }

    std::cout << "..." << std::endl;
    for (int i = std::max(0, num_elements - 100); i < num_elements; ++i)
    {
      std::cout << std::setw(5) << i + 1 << ": " << std::setw(5) << expected[i] << std::setw(5) << classes[i] << std::endl;
    }

    std::cout << "Example merges:" << std::endl;
    for (int i = 0; i < std::min(100, (int)a.size()); ++i)
    {
      std::cout << "  Merge " << a[i] << " and " << b[i] << std::endl;
    }
  }
}

int main()
{
  // testUnionFindSmall();
  // testUnionFindLarge();

  std::ifstream rulefile("rules.txt");
 
  std::vector<FuncNode> nodes{};
  std::vector<Rule> rules{};
  parseRuleFile(rulefile, nodes, rules);
  std::cout << "Parsed " << rules.size() << " rules with " << nodes.size() << " function nodes." << std::endl;
  return 0;
}
