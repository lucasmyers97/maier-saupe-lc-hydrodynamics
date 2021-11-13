
#include "Function.hpp"
#include "Factory.hpp"

namespace functionfactory {

void Function::declare(const std::string& full_name, FunctionMaker* maker) {
  Factory::get().declare(full_name, maker);
}

}
