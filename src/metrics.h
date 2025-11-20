#ifndef METRICS_H
#define METRICS_H
#include <vector>
#include <string>

struct Metric
{
    int n_rules;
    int n_input_exprs;
    std::vector<float> time_sec_match;
    std::vector<float> time_sec_apply;
    std::vector<std::vector<float>> time_sec_repair_step1;
    std::vector<std::vector<float>> time_sec_repair_step2;
    std::vector<std::vector<float>> time_sec_repair_step4;
    std::vector<float> time_sec_full_iteration;
    std::vector<std::vector<int>> merges_per_repair_iteration;
    std::vector<int> node_count_over_time;
    std::vector<int> class_count_over_time;
};

std::string printMetricJson(const Metric &metric);

#endif