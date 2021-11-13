#ifndef FUNCTIONFACTORY_FACTORY_H_
#define FUNCTIONFACTORY_FACTORY_H_

#include "FunctionFactory/Function.hpp"

namespace functionfactory {

/**
 * Factory to create functions
 */
class Factory {
 public:
  // get the factory
  static Factory& get();

  // register a new function to be constructible
  void declare(const std::string& full_name, FunctionMaker* maker);

  // make a new function by name
  std::unique_ptr<Function> make(const std::string& full_name);

 private:
  /// library of possible functions to create
  std::unordered_map<std::string,FunctionMaker*> library_;
};

}

#endif
