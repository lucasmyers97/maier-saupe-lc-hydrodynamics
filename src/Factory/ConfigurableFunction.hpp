
#ifndef FACTORY_CONFIGURABLEFUNCTION_H_
#define FACTORY_CONFIGURABLEFUNCTION_H_

#include <memory>
#include "Factory.hpp"

// namespace with functions in it
namespace functions {
// functions that are configurable
namespace configurable {

/**
 * The base class that all dynamically created
 * functions must inherit from. This class does nothing
 * except register functions with the factory.
 */
class Base {
 public:
  // virtual destructor so the derived class's destructor will be called
  virtual ~Base() = default;
  // the maker pases the args to the constructor
  Base(int,double) {}
};

/**
 * Define the type of factory to create the derived classes
 * from this base. 
 *
 * Since this type of function we want to be creating
 * has arguments for its constructor, we need to provide
 * more template parameters to the factory::Factory class.
 */
typedef factory::Factory<Base,
        std::unique_ptr<Base>,
        int,double> Factory;

}  // namespace configurable
}  // namespace functions

/**
 * macro for declaring a new function that can be dynamically created.
 *
 * This does two tasks for us.
 * 1. It defines a function to dynamically create an instance of the input class.
 *    This function matches with constructor for this category.
 * 2. It registers the input class with the factory
 */
#define DECLARE_CONFIGURABLE_FUNCTION(NS,CLASS) \
  std::unique_ptr<functions::configurable::Base> CLASS##Maker(int i,double d) { \
    return std::make_unique<NS::CLASS>(i,d); \
  } \
  __attribute((constructor(1000))) static void CLASS##Declare() { \
    functions::configurable::Factory::get().declare( \
        std::string(#NS)+"::"+std::string(#CLASS), &CLASS##Maker); \
  }

#endif
