
#ifndef FUNCTIONFACTORY_FUNCTION_H_
#define FUNCTIONFACTORY_FUNCTION_H_

namespace functionfactory {

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
 * macro for declaring a new function that can be dynamically
 * created.
 */
#define DECLARE_FUNCTION(NS,CLASS) \
  std::unique_ptr<Function> CLASS##Maker() { \
    return std::make_unique<NS::CLASS>(); \
  } \
  __attribute((constructor(1000)) static void CLASS##Declare() { \
    functionfactory::Function::declare( \
        std::string(#NS)+"::"+std::string(#CLASS), &CLASS##Maker); \
  }

#endif
