#ifndef EGGPU_RULES_H
#define EGGPU_RULES_H

#include <cstdint>

enum FuncName {
    Unset, // default unitialized value
    Var,
    Const,
    Plus,
    Minus,
    Multiply,
    Divide,
    Neg,
    Pow,
    Sqrt,
    FAbs,
    Cbrt,
    Log,
    Exp,
    Sin,
    Cos,
    Tan,
    ASin,
    ACos,
    ATan,
    Sinh,
    Cosh,
    Tanh,
    ASinh,
    ACosh,
    ATanh,
    LT,
    LE,
    GT,
    GE,
    Not,
    If,
    And,
    Or,
    UnknownUnary,
    UnknownBinary,
    UnknownTernary
};

const unsigned char func_operand_count[] = {
    0,    // unset,
    1,    // var,
    1,    // const
    2,    // {"+", Plus},
    2,    // {"-", Minus},
    2,    // {"*", Multiply},
    2,    // {"/", Divide},
    1,    // {"neg", Neg},
    2,    // {"pow", Pow},
    1,    // {"sqrt", Sqrt},
    1,    // {"fabs", FAbs},
    1,    // {"cbrt", Cbrt},
    1,    // {"log", Log},
    1,    // {"exp", Exp},
    1,    // {"sin", Sin},
    1,    // {"cos", Cos},
    1,    // {"tan", Tan},
    1,    // {"asin", ASin},
    1,    // {"acos", ACos},
    1,    // {"atan", ATan},
    1,    // {"sinh", Sinh},
    1,    // {"cosh", Cosh},
    1,    // {"tanh", Tanh},
    1,    // {"asinh", ASinh},
    1,    // {"acosh", ACosh},
    1,    // {"atanh", ATanh},
    2,    // {"<", LT},
    2,    // {"<=", LE},
    2,    // {">", GT},
    2,    // {">=", GE},
    1,    // {"not", Not},
    3,    // {"if", If},
    2,    // {"and", And},
    1,    // {"or", Or},
    1,    // UnknownUnary,
    2,    // UnknownBinary,
    3     // UnknownTernary
};


const static unsigned char MAX_FUNC_ARGS = 3;
struct FuncNode {
    FuncName name;
    unsigned char numArgs;
    /**
     * In case of most funcs, gives ID of child nodes.
     * In case of Var, gives ID of variable.
     * In case of Const, gives value (as casted float)
     */
    uint32_t args[MAX_FUNC_ARGS]; 
};
// all rules match to 4 variables or less (but possibly greater nodes)
const static unsigned char MAX_RULE_VARS = 4; 
struct Rule {
    uint32_t id;
    uint32_t lhs; // Id of the lhs FuncNode root
    uint32_t rhs; // Id of the rhs FuncNode root
};


unsigned char getOperandCount(FuncName n);


#endif //EGGPU_RULES_H