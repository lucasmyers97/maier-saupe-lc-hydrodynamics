
#include "Function.hpp"

namespace factory {

void Function::declare(const std::string& full_name, FunctionMaker* maker) {
  FunctionFactory::get().declare(full_name, maker);
}

}
