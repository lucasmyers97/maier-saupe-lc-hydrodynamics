
#ifndef FUNCTIONFACTORY_CONFIGURABLEFUNCTION_H_
#define FUNCTIONFACTORY_CONFIGURABLEFUNCTION_H_

#include <memory>
#include "Factory.hpp"

// namespace with functions in it
namespace configurablefunctions {

/**
 * The base class that all dynamically created
 * functions must inherit from. This class does nothing
 * except register functions with the factory.
 */
class ConfigurableFunction {
 public:
  // define the type of function we will use to make this object
  typedef std::unique_ptr<ConfigurableFunction> Maker(int,double);
  // virtual destructor so the derived class's destructor will be called
  virtual ~ConfigurableFunction() = default;
  // the maker pases the args to the constructor
  ConfigurableFunction(int,double) {}
};

/**
 * Define the type of factory to create the derived classes
 * from this base. 
 *
 * I think this is helpful because
 * 1. If you are using namespaces more liberally,
 *    then this save some namespace typing.
 * 2. It avoid confusion with the factory design.
 *    With this typedef, getting an object would be
 *      FunctionFactory::get().make("foo::Bar");
 *    While without, you would
 *      Factory<Function>::get().make("foo::Bar");
 *    even though one would be reasonable to assume to try
 *      Factory<foo::Bar>::get().make("foo::Bar");
 *    which **would not work**.
 *
 * Moreover, notice that since we are in a different
 * namespace that where Factory is defined, we can redefine
 * the type 'Factory' to be specifically for Functions in
 * the functions namespace. This leads to the very eye-pleasing
 * format
 *  functions::Factory::get().make("foo::Bar");
 */
typedef factory::Factory<ConfigurableFunction,
        std::unique_ptr<ConfigurableFunction>,
        int,double> Factory;

}  // configurablefunctions

/**
 * macro for declaring a new function that can be dynamically created.
 *
 * This does two tasks for us.
 * 1. It defines a function of type FunctionMaker to create an instance
 *    of the input class.
 * 2. It registers the input class with the factory using the Function::declare
 *    method. This registration is done early 
 */
#define DECLARE_CONFIGURABLE_FUNCTION(NS,CLASS) \
  std::unique_ptr<configurablefunctions::ConfigurableFunction> CLASS##Maker(int i,double d) { \
    return std::make_unique<NS::CLASS>(i,d); \
  } \
  __attribute((constructor(1000))) static void CLASS##Declare() { \
    configurablefunctions::Factory::get().declare( \
        std::string(#NS)+"::"+std::string(#CLASS), &CLASS##Maker); \
  }

#endif
