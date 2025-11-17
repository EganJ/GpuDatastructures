#ifndef EGGPU_PARSER_H
#define EGGPU_PARSER_H

#include <fstream>
#include <vector>
#include <string>

#include "rules.h"


void parseRuleFile(std::ifstream &file, std::vector<FuncNode> &nodes, std::vector<Rule> &rules);

std::string printExpression(const std::vector<FuncNode> &nodes, uint32_t nodeId);

#endif // EGGPU_PARSER_H