#include "metrics.h"

template <typename T>
std::string to_json_item(const T &item) {
    return std::to_string(item);
}

template <typename T2>
std::string to_json_item(const std::vector<T2> &vec)
{
    std::string json = "[";
    for (size_t i = 0; i < vec.size(); ++i)
    {
        json += to_json_item(vec[i]);
        if (i < vec.size() - 1)
        {
            json += ", ";
        }
    }
    json += "]";
    return json;
}

std::string printMetricJson(const Metric &metric)
{
    std::string json = "{\n";
    json += "  \"n_rules\": " + to_json_item(metric.n_rules) + ",\n";
    json += "  \"n_input_exprs\": " + to_json_item(metric.n_input_exprs) + ",\n";
    json += "  \"time_sec_match\": " + to_json_item(metric.time_sec_match) + ",\n";
    json += "  \"time_sec_apply\": " + to_json_item(metric.time_sec_apply) + ",\n";
    json += "  \"time_sec_repair_step1\": " + to_json_item(metric.time_sec_repair_step1) + ",\n";
    json += "  \"time_sec_repair_step2\": " + to_json_item(metric.time_sec_repair_step2) + ",\n";
    json += "  \"time_sec_repair_step4\": " + to_json_item(metric.time_sec_repair_step4) + ",\n";
    json += "  \"time_sec_full_iteration\": " + to_json_item(metric.time_sec_full_iteration) + ",\n";
    json += "  \"merges_per_repair_iteration\": " + to_json_item(metric.merges_per_repair_iteration) + ",\n";
    json += "  \"node_count_over_time\": " + to_json_item(metric.node_count_over_time) + ",\n";
    json += "  \"class_count_over_time\": " + to_json_item(metric.class_count_over_time) + "\n";
    json += "\n}";

    return json;
}