
#ifndef FUNCTIONFACTORY_FUNCTION_H_
#define FUNCTIONFACTORY_FUNCTION_H_

#include <memory>

namespace factory {

/**
 * We need to define the type of function that can create
 * one of our functions.
 */
class Function;
typedef std::unique_ptr<Function> FunctionMaker();

/**
 * The base class that all dynamically created
 * functions must inherit from. This class does nothing
 * except register functions with the factory.
 */
class Function {
 public:
  // virtual destructor so the derived class's destructor will be called
  virtual ~Function() = default;
  // declare a new derived Function
  static void declare(const std::string& full_name, FunctionMaker* maker);
};

}  // functionfactory

/**
 * macro for declaring a new function that can be dynamically created.
 *
 * This does two tasks for us.
 * 1. It defines a function of type FunctionMaker to create an instance
 *    of the input class.
 * 2. It registers the input class with the factory using the Function::declare
 *    method. This registration is done early 
 */
#define DECLARE_FUNCTION(NS,CLASS) \
  std::unique_ptr<factory::Function> CLASS##Maker() { \
    return std::make_unique<NS::CLASS>(); \
  } \
  __attribute((constructor(1000))) static void CLASS##Declare() { \
    factory::Function::declare( \
        std::string(#NS)+"::"+std::string(#CLASS), &CLASS##Maker); \
  }

#endif
