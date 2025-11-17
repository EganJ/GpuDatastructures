#include "rules.h"

unsigned char getOperandCount(FuncName n)
{
    return func_operand_count[(int)n];
}