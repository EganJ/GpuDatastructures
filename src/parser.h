#ifndef EGGPU_PARSER_H
#define EGGPU_PARSER_H

#include <fstream>
#include <vector>
#include <string>

#include "rules.h"


void parseRuleFile(std::ifstream &file, std::vector<FuncNode> &nodes, std::vector<Rule> &rules);

#endif // EGGPU_PARSER_H