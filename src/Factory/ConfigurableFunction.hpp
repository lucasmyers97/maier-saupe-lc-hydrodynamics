#ifndef FACTORY_CONFIGURABLEFUNCTION_HPP
#define FACTORY_CONFIGURABLEFUNCTION_HPP

// make sure we can define our factory
#include "Factory.hpp"

// namespace with functions in it
namespace functions {

/**
 * The base class that all dynamically created
 * functions must inherit from. This class does nothing
 * except register functions with the factory.
 */
class ConfigurablePrototype {
 public:
  // virtual destructor so the derived class's destructor will be called
  virtual ~ConfigurablePrototype() = default;
  // the maker pases the args to the constructor
  ConfigurablePrototype(int, double) {}

  /**
   * Pure virtual function to have derived classes use
   */
  virtual double evaluate() = 0;

  /**
   * Define the type of factory to create the derived classes
   * from this base.
   *
   * Since this type of function we want to be creating
   * has arguments for its constructor, we need to provide
   * more template parameters to the factory::Factory class.
   */
  using Factory =
      factory::Factory<ConfigurablePrototype,
                       std::unique_ptr<ConfigurablePrototype>, int, double>;
};  // ConfigurablePrototype

}  // namespace functions

/**
 * macro for declaring a new function that can be dynamically created.
 *
 * This does two tasks for us.
 * 1. It defines a function to dynamically create an instance of the input
 * class. This function matches with constructor for this category.
 * 2. It registers the input class with the factory
 */
#define DECLARE_CONFIGURABLE_FUNCTION(NS, CLASS)                               \
  namespace NS {                                                               \
  std::unique_ptr<::functions::ConfigurablePrototype> CLASS##Maker(int i,      \
                                                                   double d) { \
    return std::make_unique<CLASS>(i, d);                                      \
  }                                                                            \
  __attribute__((constructor)) static void CLASS##Declare() {                  \
    ::functions::ConfigurablePrototype::Factory::get().declare(                \
        std::string(#NS) + "::" + std::string(#CLASS), &CLASS##Maker);         \
  }                                                                            \
  }

#endif  // FACTORY_CONFIGURABLEFUNCTION_HPP
