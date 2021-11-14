#ifndef FACTORY_FUNCTION_H_
#define FACTORY_FUNCTION_H_

#include <memory>
#include "Factory.hpp"

// namespace with functions in it
namespace functions {

/**
 * The base class that all dynamically created
 * functions must inherit from. This class does nothing
 * except register functions with the factory.
 */
class Base {
 public:
  // virtual destructor so the derived class's destructor will be called
  virtual ~Base() = default;
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
 *      functions::Factory::get().make("foo::Bar");
 *    While without, you would
 *      factory::Factory<Base>::get().make("foo::Bar");
 *    even though one would be reasonable to assume to try
 *      Factory<foo::Bar>::get().make("foo::Bar");
 *    which **would not work**.
 *
 * Moreover, notice that since we are in a different
 * namespace that where Factory is defined, we can redefine
 * the type 'Factory' to be specifically for Bases in
 * the functions namespace. This leads to the very eye-pleasing
 * format
 *  functions::Factory::get().make("foo::Bar");
 */
typedef factory::Factory<Base> Factory;

}  // functions

/**
 * macro for declaring a new function that can be dynamically created.
 *
 * This does two tasks for us.
 * 1. It defines a function of to dynamically create an instance of the input class.
 * 2. It registers the input class with the factory.
 *    This registration is done early in the program's execution procedure
 *    because of the __attribute__ attached to it.
 *
 * A good short answer to your questions about how __attribute__'s work is
 *  https://stackoverflow.com/questions/2053029/how-exactly-does-attribute-constructor-work
 * NOTE: __attribute__ is GCC-specific syntax that is not portable to clang.
 *  https://gcc.gnu.org/onlinedocs/gcc/Function-Attributes.html
 */
#define DECLARE_FUNCTION(NS,CLASS) \
  std::unique_ptr<functions::Base> CLASS##Maker() { \
    return std::make_unique<NS::CLASS>(); \
  } \
  __attribute__((constructor)) static void CLASS##Declare() { \
    functions::Factory::get().declare( \
        std::string(#NS)+"::"+std::string(#CLASS), &CLASS##Maker); \
  }

#endif
