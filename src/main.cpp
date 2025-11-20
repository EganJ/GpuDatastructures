#include <iostream>
#include <iomanip>
#include <filesystem>

#include "datastructures.h"
#include "parser.h"
#include "eqsat.h"

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

  // Parse rules
  std::ifstream rulefile("tmp_rules.txt");

  std::vector<FuncNode> rule_nodes{};
  std::vector<Rule> rules{};
  parseRuleFile(rulefile, rule_nodes, rules);
  std::cout << "Parsed " << rules.size() << " rules with " << rule_nodes.size() << " uncompressed total nodes." << std::endl;
  // Compress rules node space
  {
    std::vector<FuncNode> compressed_rule_nodes;
    std::vector<Rule> compressed_rules;
    compress_nodespace(rule_nodes, rules, compressed_rule_nodes, compressed_rules);
    std::cout << "Compressed to " << compressed_rules.size() << " rules with " << compressed_rule_nodes.size() << " total nodes." << std::endl;
    rule_nodes = std::move(compressed_rule_nodes);
    rules = std::move(compressed_rules);
  }

  // Parse FPCore expressions
  std::string benchdir = "bench/";
  std::vector<uint32_t> expr_roots;
  std::vector<FuncNode> nodes;
  try
  {
    // Gaurantee that demo.fpcore is first
    {
      std::cout << "Parsing FPCore file: " << benchdir + "demo.fpcore" << std::endl;
      std::ifstream fpcorefile(benchdir + "demo.fpcore");
      parseFPCoreFile(fpcorefile, nodes, expr_roots);
    }
    for (const auto &entry : std::filesystem::recursive_directory_iterator(benchdir))
    {
      if (entry.path().extension() == ".fpcore" && entry.path().filename() != "demo.fpcore")
      {
        std::cout << "Parsing FPCore file: " << entry.path() << std::endl;
        std::ifstream fpcorefile(entry.path());
        parseFPCoreFile(fpcorefile, nodes, expr_roots);
      }
    }
  }
  catch (std::filesystem::filesystem_error &e)
  {
    std::cout << "Filesystem error: " << e.what() << std::endl;
  }

  // Construct datastructures on GPU and run matching kernels
  gpuds::eqsat::initialize_eqsat_memory();
  gpuds::eqsat::initialize_ruleset_on_device(rule_nodes, rules);

  int n_tests = 10;
  std::vector<int> root_subset = std::vector<int>(expr_roots.begin(), expr_roots.begin() + std::min(n_tests, (int)expr_roots.size()));
  std::cout << "Constructing E-graph for " << expr_roots.size() << " expressions:" << std::endl;
  for (int i = 0; i < root_subset.size(); ++i)
  {
    std::cout << " Expression " << i << " root node ID: " << root_subset[i] << " " << printExpression(nodes, root_subset[i]) << std::endl;
  }

  std::vector<int> adjusted_indices;
  gpuds::eqsat::EqSatSolver *solver = gpuds::eqsat::construct_eqsat_solver(nodes, root_subset, adjusted_indices);
  for (int i = 0; i < 3; i++)
  {
    gpuds::eqsat::launch_eqsat_match_rules(solver);
    gpuds::eqsat::launch_eqsat_apply_rules(solver);
    gpuds::eqsat::repair_egraph(solver);
  }
  return 0;
}
