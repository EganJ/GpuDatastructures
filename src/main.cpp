#include <iostream>
#include <iomanip>
#include <filesystem>
#include <cuda_runtime.h>

#include "datastructures.h"
#include "parser.h"
#include "eqsat.h"
#include "metrics.h"

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

int main(int argc, char **argv)
{
  int n_expressions;
  // Read from args.

  if (argc > 1) {
      n_expressions = std::stoi(argv[1]);
      printf("Using the first %d expressions found in the test set.\n", n_expressions);
  } else {
      printf("Please provide number of expressions to process as first argument.\n");
      return 1;
  }

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

  int n_tests = n_expressions;
  std::vector<int> root_subset = std::vector<int>(expr_roots.begin(), expr_roots.begin() + std::min(n_tests, (int)expr_roots.size()));
  std::cout << "Constructing E-graph for " << expr_roots.size() << " expressions:" << std::endl;
  for (int i = 0; i < root_subset.size(); ++i)
  {
    std::cout << " Expression " << i << " root node ID: " << root_subset[i] << " " << printExpression(nodes, root_subset[i]) << std::endl;
  }

  std::vector<int> adjusted_indices;
  gpuds::eqsat::EqSatSolver *solver = gpuds::eqsat::construct_eqsat_solver(nodes, root_subset, adjusted_indices);

  Metric data_metrics = Metric(); 
  data_metrics.n_rules = rules.size();
  data_metrics.n_input_exprs = root_subset.size();
  int N_ITERS = 5;
  cudaEvent_t match_start, match_stop, apply_stop, repair_stop;
  cudaEventCreate(&match_start);
  cudaEventCreate(&match_stop);
  cudaEventCreate(&apply_stop);
  cudaEventCreate(&repair_stop);

  int node_count;
  int class_count;
  for (int i = 0; i < N_ITERS; i++)
  {
    cudaEventRecord(match_start);
    gpuds::eqsat::launch_eqsat_match_rules(solver);
    cudaEventRecord(match_stop);
    gpuds::eqsat::launch_eqsat_apply_rules(solver);
    cudaEventRecord(apply_stop);
    // There is an implicit cudaDeviceSynchronize() inside of repair egraph.
    gpuds::eqsat::repair_egraph(solver, data_metrics);
    cudaEventRecord(repair_stop);
    std::cout << "Done with iteration " << i + 1 << std::endl;

    // Record times
    cudaEventSynchronize(repair_stop);
    float match_time;
    cudaEventElapsedTime(&match_time, match_start, match_stop);
    data_metrics.time_sec_match.push_back(match_time / 1000.0f);
    cudaEventSynchronize(apply_stop);
    float apply_time;
    cudaEventElapsedTime(&apply_time, match_stop, apply_stop);
    data_metrics.time_sec_apply.push_back(apply_time / 1000.0f);
    float full_time;
    cudaEventElapsedTime(&full_time, match_start, repair_stop);
    data_metrics.time_sec_full_iteration.push_back(full_time / 1000.0f);

    cudaMemcpy(&node_count,
            (char *)solver +
                offsetof(gpuds::eqsat::EqSatSolver, egraph) + offsetof(EGraph, num_nodes),
            sizeof(int), cudaMemcpyDeviceToHost);
  
    cudaMemcpy(&class_count,
          (char *)solver +
              offsetof(gpuds::eqsat::EqSatSolver, egraph) + offsetof(EGraph, num_classes),
          sizeof(int), cudaMemcpyDeviceToHost);

    data_metrics.node_count_over_time.push_back(node_count);
    data_metrics.class_count_over_time.push_back(class_count);
  }
  std::cout << "Final E-graph state:" << std::endl;
  gpuds::eqsat::print_eqsat_solver_state(solver);
  std::string metric_json = printMetricJson(data_metrics);
  std::ofstream metric_file("metrics_output.json");
  metric_file << metric_json;
  metric_file.close();
  std::cout << "Metrics written to metrics_output.json" << std::endl;

  cudaEventDestroy(match_start);
  cudaEventDestroy(match_stop);
  cudaEventDestroy(apply_stop);
  cudaEventDestroy(repair_stop);

  return 0;
}
