
#include "FunctionFactory/Function.hpp"
#include "FunctionFactory/Factory.hpp"

namespace functionfactory {

void Function::declare(const std::string& full_name, FunctionMaker* maker) {
  Factory::get().declare(full_name, maker);
}

}
