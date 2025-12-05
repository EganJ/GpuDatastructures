#include "rules.h"

unsigned char getOperandCount(FuncName n)
{
    return func_operand_count[(int)n];
}

void compress_nodespace(const std::vector<FuncNode> &nodes, const std::vector<int> &root_terms,
                        std::vector<FuncNode> &out_nodes, std::vector<int> &out_roots)
{
    std::map<int, int> old_to_new_id;

    out_nodes.clear();
    out_roots.clear();

    for (int i = 0; i < root_terms.size(); i++)
    {
        int new_root = add_node_deduplicated(nodes, out_nodes, old_to_new_id, root_terms[i]);
        out_roots.push_back(new_root);
    }
}

void compress_nodespace(const std::vector<FuncNode> &nodes, const std::vector<Rule> &in_rules,
                        std::vector<FuncNode> &out_nodes, std::vector<Rule> &out_rules)
{
    std::map<int, int> old_to_new_id;

    out_nodes.clear();
    out_rules.clear();

    for (int i = 0; i < in_rules.size(); i++)
    {
        int new_lhs_root = add_node_deduplicated(nodes, out_nodes, old_to_new_id, in_rules[i].lhs);
        int new_rhs_root = add_node_deduplicated(nodes, out_nodes, old_to_new_id, in_rules[i].rhs);
        Rule new_rule;
        new_rule.id = in_rules[i].id;
        new_rule.lhs = new_lhs_root;
        new_rule.rhs = new_rhs_root;
        out_rules.push_back(new_rule);
    }
}

int add_node_deduplicated(const std::vector<FuncNode> &original_memspace,
                          std::vector<FuncNode> &new_namespace,
                          std::map<int, int> &id_mappings, int original_id)
{
    if (id_mappings.find(original_id) != id_mappings.end())
    {
        return id_mappings[original_id]; // already added
    }

    FuncNode node = original_memspace[original_id];
    if (node.name != FuncName::Var && node.name != FuncName::Const)
    {
        int argc = (int)getOperandCount(node.name);
        for (int i = 0; i < argc; i++)
        {
            int child_original_id = node.args[i];
            int child_new_id = add_node_deduplicated(original_memspace,
                                                     new_namespace, id_mappings, child_original_id);
            node.args[i] = child_new_id;
        }
    }

    // Need to add this node only if its not already present!
    bool present = false;
    for (int i = 0; i < new_namespace.size(); i++)
    {
        const FuncNode &existing_node = new_namespace[i];
        if (existing_node.name == node.name)
        {
            bool all_args_match = true;
            int argc = getOperandCount(node.name);
            for (int j = 0; j < argc; j++)
            {
                if (existing_node.args[j] != node.args[j])
                {
                    all_args_match = false;
                    break;
                }
            }
            if (all_args_match)
            {
                present = true;
                id_mappings[original_id] = i;
                break;
            }
        }
    }

    if (!present)
    {
        int new_id = new_namespace.size();
        new_namespace.push_back(node);
        id_mappings[original_id] = new_id;
        return new_id;
    }
    else
    {
        return id_mappings[original_id];
    }
}