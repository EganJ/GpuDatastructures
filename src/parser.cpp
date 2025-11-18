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
    // std::cout << msg << std::endl;
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
    {"or", Or},
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
    {"or", Or},
    {"floor", UnknownUnary},
    {"fmax", UnknownBinary},
    {"log2", UnknownUnary}, // TODO could possibly desugar these
    {"hypot", UnknownBinary}, // TODO could possibly desugar these
    {"atan2", UnknownBinary}, // TODO can this be desugared?
    {"fmod", UnknownBinary},
    {"==", UnknownBinary}, // TODO for bools, can be desugared
    {"fma", UnknownTernary}, // TODO could possibly desugar this, or add to op set.
    {"re_sqr", UnknownBinary}, // Actually user-defined function in terms of our ops (possible through FPCore)
    {"im_sqr", UnknownBinary}, // Actually user-defined function in terms of our ops (possible through FPCore)
    {"modulus_sqr", UnknownBinary}, // Actually user-defined function in terms of our ops (possible through FPCore)
    {"modulus", UnknownBinary}, // Actually user-defined function in terms of our ops (possible through FPCore)
    {"copysign", UnknownBinary}, // Actually user-defined function in terms of our ops (possible through FPCore)
    {"log1p", UnknownUnary}, // Actually user-defined function in terms of our ops (possible through FPCore)
    };

std::string opToName(FuncName fn)
{
    switch (fn) {
        case UnknownUnary:
            return "unknown_unary";
        case UnknownBinary:
            return "unknown_binary";
        case UnknownTernary:
            return "unknown_ternary";
        default:
            break;
    }

    for (const auto &pair : map)
    {
        if (pair.second == fn)
        {
            return pair.first;
        }
    }
    return "unknown_op";
}

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

void skipAll(std::string in, int &index)
{
    while (index < in.size() && isSkipChar(in[index]))
        index++;
}

std::string nextToken(std::string in, int &index)
{
    skipAll(in, index);
    if (index >= in.size())
    {
        return "";
    }
    int start = index;
    while (index < in.size() && (!isWordEndChar(in[index]) || index == start))
        index++;
    int end = index;

    return in.substr(start, end - start);
}

/**
 * Returns the next term, either an atom or a parenthesized expression.
 */
std::string nextTerm(std::string in, int &index, char open = '(', char close = ')')
{
    skipAll(in, index);
    if (index >= in.size())
    {
        return "";
    }
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

/**
 * Returns the next term with the given delimiters, skipping any leading items in the string
 * until an open
 */
std::string nextTermInSequence(std::string in, int &index, char open = '(', char close = ')')
{
    while (index < in.size() && in[index] != open)
        index++;
    if (index >= in.size())
        return "";

    return nextTerm(in, index, open, close);
}

/**
 * May be [var expr] or (var expr) bindings
 */
std::string nextLetBinding(std::string in, int &index)
{
    while(index < in.size() && (
        in[index] != '[' && in[index] != '(' && in[index] != ')' && in[index] != ']'
    )) {
        index++;
    }
    int start = index;
    if (index >= in.size() || (in[index] != '[' && in[index] != '('))
    {
        return "";
    }
    char open = in[index];
    char close = (open == '[') ? ']' : ')';
    // find matching closing
    return nextTerm(in, index, open, close);
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
            if (varNums.find(in) == varNums.end())
            {
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
 *
 * May have let bindings.
 */
unsigned buildTree(std::string in, std::vector<FuncNode> &nodes, std::map<std::string, int> &varNums, std::map<std::string, unsigned> &letBindings)
{
    if (in[0] != '(')
    {
        // Check if it's a let binding
        if (letBindings.find(in) != letBindings.end())
        {
            return letBindings[in];
        }
        else
        {
            FuncNode terminal = strToFunc(in, varNums);
            nodes.push_back(terminal);
            return nodes.size() - 1;
        }
    }
    else
    {
        int start = 1; // skip opening paren
        std::string op = nextToken(in, start);

        if (op == "let" || op == "let*")
        {
            bool sequential = (op == "let*");
            std::map<std::string, unsigned> localBindings; // copy parent bindings
            // If sequential this will write to letBindings for each term to read in sequence,
            // otherwise write to localBindings and merge at end.
            auto &write_bindings = sequential ? letBindings : localBindings;
            std::string bindings_term = nextTerm(in, start);
            std::string expr = nextTerm(in, start);

            // Parse bindings
            int bind_index = 1; // skip open paren
            std::string binding;
            while ((binding = nextLetBinding(bindings_term, bind_index)) != "")
            {
                int local_bind_index = 1;
                std::string bind_name = nextToken(binding, local_bind_index);
                std::string bind_expr = nextTerm(binding, local_bind_index);
                unsigned var_node = buildTree(bind_expr, nodes, varNums, letBindings);
                write_bindings[bind_name] = var_node;
            }

            // Merge local bindings if not sequential
            if (!sequential)
            {
                for (const auto &pair : localBindings)
                {
                    letBindings[pair.first] = pair.second;
                }
            }

            // Build and return main expr
            return buildTree(expr, nodes, varNums, letBindings);
        }
        else
        {

            FuncNode f = strToFunc(op, varNums);
            int argc = getOperandCount(f.name);

            for (int i = 0; i < getOperandCount(f.name); ++i)
            {
                std::string term = nextTerm(in, start);
                // Only one ambiguity here: unary vs binary minus.
                if (f.name == Minus && term == ")")
                {
                    // It's a unary minus
                    f.name = Neg;
                    argc = 1;
                    break;
                }

                unsigned child_id = buildTree(term, nodes, varNums, letBindings);
                f.args[i] = child_id;
            }

            nodes.push_back(f);
            return nodes.size() - 1;
        }
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
    std::map<std::string, unsigned> letBindings;
    std::string lhs_str = nextTerm(rule, pos);
    unsigned lhs_root = buildTree(lhs_str, nodes, varNums, letBindings);

    std::string rhs_str = nextTerm(rule, pos);
    unsigned rhs_root = buildTree(rhs_str, nodes, varNums, letBindings);
    assert(letBindings.size() == 0 && "Let bindings should not be used in rule parsing");

    Rule r;
    r.id = rules.size();
    r.lhs = lhs_root;
    r.rhs = rhs_root;
    rules.push_back(r);
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

void stripFPCoreComments(std::string &in)
{
   // Find all ; and replace to end of line with spaces
   size_t pos = 0;
   while ((pos = in.find(';', pos)) != std::string::npos){
         size_t end_pos = in.find('\n', pos);
         if (end_pos == std::string::npos) {
              end_pos = in.size();
         }
         for (size_t i = pos; i < end_pos; ++i) {
              in[i] = ' ';
         }
         pos = end_pos;
   }
}

/**
 * Does not require that all expressions fit in a single line.
 * Will find an FPCore expression, parse its vars, and parse the last term (the expression body).
 * Will ignore any other content in the expression.
 */
void parseFPCoreFile(std::ifstream &file, std::vector<FuncNode> &nodes, std::vector<uint32_t> &expr_roots)
{
    // Read file to string in entirety
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    stripFPCoreComments(content);

    int index = 0;
    for (std::string found = nextTermInSequence(content, index); found != ""; found = nextTermInSequence(content, index))
    {
        debugPrint("Found term: " + found);
        int sub_index = 1; // skip open paren
        std::string token = nextToken(found, sub_index);
        assert(token == "FPCore" && "Parse error: expected FPCore expression");

        std::string vars = nextTermInSequence(found, sub_index);
        debugPrint("Vars: " + vars);

        std::map<std::string, int> varNums;
        std::map<std::string, unsigned> letBindings;

        // Spin up a new var. Since we know them ahead of time, we can bind them directly
        // into the letBindings. We then should expect varNums to remain empty.
        int var_str_idx = 1; // skip open paren
        std::string var_name;
        while ((var_name = nextToken(vars, var_str_idx)) != ")")
        {
            unsigned var_node = nodes.size();
            FuncNode varFunc;
            varFunc.name = Var;
            varFunc.args[0] = letBindings.size();
            nodes.push_back(varFunc);
            letBindings[var_name] = var_node;
        }

        // Find last term in FPCore expression.
        std::string found_rev = found;
        std::reverse(found_rev.begin(), found_rev.end());
        int rev_index = 1; // skip close paren
        std::string last_term = nextTerm(found_rev, rev_index, ')', '(');
        std::reverse(last_term.begin(), last_term.end());
        debugPrint("Last term: " + last_term);

        // Parse last term with let bindings. Check that no unbound vars are encountered.
        unsigned expr_root = buildTree(last_term, nodes, varNums, letBindings);
        assert(varNums.size() == 0 && "All vars should be bound in let bindings");
        expr_roots.push_back(expr_root);

        debugPrint("Parsed expression: " + printExpression(nodes, expr_root));
    }
}