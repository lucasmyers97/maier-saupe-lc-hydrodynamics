#ifndef FACTORY_FUNCTION_HPP
#define FACTORY_FUNCTION_HPP

// make sure we can define our factory
#include "Factory.hpp"

// namespace with functions in it
namespace functions {

/**
 * The base class that all dynamically created
 * functions must inherit from.
 *
 * This class does nothing except be the handle
 * for all other types of objects in this 'category'.
 */
class FunctionPrototype {
 public:
  // virtual destructor so the derived class's destructor will be called
  virtual ~FunctionPrototype() = default;

  /**
   * Also define other virtual functions that you want the
   * child classes to implement.
   *
   * Here I've put in a pure virtual function that
   * will have the child return some integer.
   *
   * The '= 0' declares this function as "pure virtual".
   * Which means two things:
   * 1. The child classes are _required_ to implement it.
   * 2. The parent class ('FunctionPrototype') cannot be instantiated.
   */
  virtual int evaluate() = 0;

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
   *      factory::Factory<FunctionPrototype>::get().make("foo::Bar");
   *    even though one would be reasonable to assume to try
   *      Factory<foo::Bar>::get().make("foo::Bar");
   *    which **would not work**.
   *
   * Moreover, notice that since we are in a different
   * namespace that where Factory is defined, we can redefine
   * the type 'Factory' to be specifically for FunctionPrototypes in
   * the functions namespace. This leads to the very eye-pleasing
   * format
   *  functions::Factory::get().make("foo::Bar");
   */
  using Factory = factory::Factory<FunctionPrototype>;
};

}  // namespace functions

/**
 * macro for declaring a new function that can be dynamically created.
 *
 * This does two tasks for us.
 * 1. It defines a function to dynamically create an instance of the input
 * class.
 * 2. It registers the input class with the factory.
 *    This registration is done early in the program's execution procedure
 *    because of the __attribute__ attached to it.
 *
 * This macro should be called outside of all namespaces to avoid any clashing.
 *
 * A good short answer to your questions about how __attribute__'s work is
 *  https://stackoverflow.com/questions/2053029/how-exactly-does-attribute-constructor-work
 * NOTE: __attribute__ is a function decorator supported by BOTH gcc and clang
 *  https://gcc.gnu.org/onlinedocs/gcc/Function-Attributes.html
 *  https://blog.timac.org/2016/0716-constructor-and-destructor-attributes/
 * There are methods to do similar procedures that is portable to other
 * compilers; however, their syntax is much more complicated.
 */
#define DECLARE_FUNCTION(NS, CLASS)                                    \
  namespace NS {                                                       \
  std::unique_ptr<::functions::FunctionPrototype> CLASS##Maker() {     \
    return std::make_unique<CLASS>();                                  \
  }                                                                    \
  __attribute__((constructor)) static void CLASS##Declare() {          \
    ::functions::FunctionPrototype::Factory::get().declare(            \
        std::string(#NS) + "::" + std::string(#CLASS), &CLASS##Maker); \
  }                                                                    \
  }

#endif  // FACTORY_FUNCTION_HPP
