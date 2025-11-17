#include <vector>
#include <string>
#include <map>
#include <stdio.h>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <bit>

#include "parser.h"

#define PI 3.1415926535897932384626433832795
#define E 2.718281828459045235360287471352

const int n_skipchars = 3;
const char skipchars[] = {' ', '\t', '\n'};
int n_word_endchars = 4;
const char word_endchars[] = {'(', ')', '[', ']'}; // plus any skipchars

void debugPrint(const std::string &msg)
{
    std::cout << msg << std::endl;
}

const std::unordered_map<std::string, FuncName> map = {
    {"+", Plus},
    {"-", Minus},
    {"*", Multiply},
    {"/", Divide},
    {"neg", Neg},
    {"pow", Pow},
    {"sqrt", Sqrt},
    {"fabs", FAbs},
    {"cbrt", Cbrt},
    {"log", Log},
    {"exp", Exp},
    {"sin", Sin},
    {"cos", Cos},
    {"tan", Tan},
    {"asin", ASin},
    {"acos", ACos},
    {"atan", ATan},
    {"sinh", Sinh},
    {"cosh", Cosh},
    {"tanh", Tanh},
    {"asinh", ASinh},
    {"acosh", ACosh},
    {"atanh", ATanh},
    {"<", LT},
    {"<=", LE},
    {">", GT},
    {">=", GE},
    {"not", Not},
    {"if", If},
    {"and", And},
    {"or", Or}};

bool isSkipChar(char c)
{
    for (int i = 0; i < n_skipchars; ++i)
    {
        if (c == skipchars[i])
            return true;
    }
    return false;
}

bool isWordEndChar(char c)
{
    if (isSkipChar(c))
        return true;
    for (int i = 0; i < n_word_endchars; ++i)
    {
        if (c == word_endchars[i])
            return true;
    }
    return false;
}

std::string nextToken(std::string in, int &index)
{
    while (isSkipChar(in[index]))
        index++;
    int start = index;
    while (!isWordEndChar(in[index]) && index < in.size())
        index++;
    int end = index;

    return in.substr(start, end - start);
}

/**
 * Returns the next term, either an atom or a parenthesized expression.
 */
std::string nextTerm(std::string in, int &index, char open = '(', char close = ')')
{
    while (isSkipChar(in[index]))
        index++;
    int start = index;
    if (in[index] != open)
    {
        return nextToken(in, index);
    }
    else
    {
        // find matching closing paren
        int parens = 1;
        index++; // skip open paren
        while (parens > 0 && index < in.size())
        {
            if (in[index] == open)
                parens++;
            else if (in[index] == close)
                parens--;
            index++;
        }
        int end = index; // index is after close paren
        return in.substr(start, end - start);
    }
}

int encode_float(float f)
{
    return std::bit_cast<int>(f);
}

/**
 * Returns the FuncNode populated only with the opcode (and var/const value, if applicable).
 */
FuncNode strToFunc(std::string in, std::map<std::string, int> &varNums)
{
    FuncNode f;

    if (map.find(in) == map.end())
    {
        if (std::string("-+0123456789").find(in[0]) != std::string::npos)
        {
            f.name = Const;
            f.args[0] = encode_float(std::stof(in));
        }
        else if (in == "1/2")
        {
            f.name = Const;
            f.args[0] = encode_float(0.5);
        }
        else if (in == "1/3")
        {
            f.name = Const;
            f.args[0] = encode_float(1.0 / 3.0);
        }
        else if (in == "PI")
        {
            f.name = Const;
            f.args[0] = encode_float(PI);
        }
        else if (in == "E")
        {
            f.name = Const;
            f.args[0] = encode_float(E);
        }
        else
        {
            debugPrint("Variable: " + in);
            if (varNums.find(in) == varNums.end()) {
                debugPrint("Assigning var number " + std::to_string(varNums.size()) + " to variable " + in);
                varNums[in] = varNums.size();
            }

            f.name = Var;
            f.args[0] = varNums[in];
        }
    }
    else
    {
        f.name = map.at(in);
    }

    return f;
}

/**
 * Builds a tree from the input string, returning the index of the root node in the nodes vector.
 * Expects "in" in the form "(op arg1 arg2 ...)" or "atom".
 */
unsigned buildTree(std::string in, std::vector<FuncNode> &nodes, std::map<std::string, int> &varNums)
{
    if (in[0] != '(')
    {
        FuncNode terminal = strToFunc(in, varNums);
        nodes.push_back(terminal);
        return nodes.size() - 1;
    }
    else
    {
        int start = 1; // skip opening paren
        std::string op = nextToken(in, start);
        FuncNode f = strToFunc(op, varNums);
        int argc = getOperandCount(f.name);

        for (int i = 0; i < getOperandCount(f.name); ++i)
        {
            std::string term = nextTerm(in, start);
            unsigned child_id = buildTree(term, nodes, varNums);
            f.args[i] = child_id;
        }

        nodes.push_back(f);
        return nodes.size() - 1;
    }
}

/**
 * Parses a rule from a string "[name lhs rhs]"
 */
void parseRule(std::vector<FuncNode> &nodes, std::vector<Rule> &rules, std::string &rule)
{
    debugPrint("Parsing rule: " + rule);
    assert(rule[0] == '[');
    int pos = 1;                                  // skip opening bracket
    std::string rule_name = nextToken(rule, pos); // throw this away for now

    std::map<std::string, int> varNums;
    std::string lhs_str = nextTerm(rule, pos);
    unsigned lhs_root = buildTree(lhs_str, nodes, varNums);

    std::string rhs_str = nextTerm(rule, pos);
    unsigned rhs_root = buildTree(rhs_str, nodes, varNums);

    Rule r;
    r.id = rules.size();
    r.lhs = lhs_root;
    r.rhs = rhs_root;
    rules.push_back(r);
}

/**
 * Requires that all rules fit in a single line.
 */
void parseRuleFile(std::ifstream &file, std::vector<FuncNode> &nodes, std::vector<Rule> &rules)
{
    std::string line = "";
    int line_num = 0;
    std::vector<std::string> lines;
    while (std::getline(file, line))
    {
        int start = 0;
        while (start < line.size() && isSkipChar(line[start]))
            start++;
        if (start < line.size())
        {

            if (line[start] == '[')
            {
                std::string rule = nextTerm(line, start, '[', ']');
                parseRule(nodes, rules, rule);
            }
        }
    }
}

std::string opToName(FuncName fn)
{
    for (const auto &pair : map)
    {
        if (pair.second == fn)
        {
            return pair.first;
        }
    }
    return "unknown_op";
}

std::string printExpression(const std::vector<FuncNode> &nodes, uint32_t nodeId)
{
   std::string result = "";
    const FuncNode &node = nodes[nodeId];
   if (node.name == Var)
   {
       result += "v" + std::to_string(nodes[nodeId].args[0]);
   }
   else if (node.name == Const)
   {
       float val = std::bit_cast<float>(nodes[nodeId].args[0]);
       result += std::to_string(val);
   }
   else
   {
       result += "(" + opToName(node.name);
       for (int i = 0; i < getOperandCount(node.name); ++i)
       {
           result += " " + printExpression(nodes, node.args[i]);
       }
       result += ")";
   }
   return result;
}