
#include "Factory.hpp"

namespace functionfactory {

Factory& Factory::get() {
  static Factory the_factory_;
  return the_factory_;
}

void Factory::declare(const std::string& full_name, FunctionMaker* maker) {
  auto lib_it{library_.find(full_name)};
  if (lib_it != library_.end()) {
    // probably should raise an exception of some kind
  }
  library_[full_name] = maker;
}

std::unique_ptr<Function> Factory::make(const std::string& full_name) {
  auto lib_it{library_.find(full_name)};
  if (lib_it == library_.end()) {
    // probably should raise exception rather than just return nullptr
    return nullptr;
  }
  // found a maker -> make it
  return lib_it->second();
}

}
