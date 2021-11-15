#ifndef FACTORY_FACTORY_HPP
#define FACTORY_FACTORY_HPP

#include <memory> // for the unique_ptr default
#include <string> // for the keys in the library map
#include <unordered_map> // for the library
#include <exception> // to throw not found exceptions

/**
 * factory namespace
 *
 * This namespace is used to isolate the templated Factory
 * from where other Factories are defined. There should be
 * nothing else in this namespace in order to avoid potential
 * name conflicts.
 */
namespace factory {

/**
 * Factory to dynamically create objects derived from a specific prototype class.
 *
 * This factory is a singleton class meaning it cannot be created by the user.
 *
 * The factory has three template parameters in order of complexity.
 * 1. Prototype - REQUIRED - the type of object that this factory creates.
 *    This should be the base class that all types in this factory derive from.
 * 2. PrototypePtr - optional - the type of pointer to object
 *    By default, we use std::unique_ptr for good memory management.
 * 3. PrototypeMakerArgs - optional - type of objects passed into the object maker
 *    i.e. same as arguments to the constructor used by the object maker
 *
 * In order to save code repetition, it is suggested to alias
 * your specific factory in your own namespace. This allows you to control
 * all the template inputs for your factory in one location.
 *
 *  using MyPrototypeFactory = factory::Factory<MyPrototype>;
 *
 * Or, if you are in some other namespace, you can shorten it even more.
 *
 *  namespace foo {
 *    using Factory = factory::Factory<MyPrototype>;
 *  }
 */
template<
  typename Prototype,
  typename PrototypePtr = std::unique_ptr<Prototype>,
  typename ... PrototypeMakerArgs
  >
class Factory {
 public:
  /**
   * the signature of a function that can be used by this factory
   * to dynamically create a new object.
   *
   * This is merely here to make the definition of the Factory simpler.
   */
  using PrototypeMaker = PrototypePtr (*)(PrototypeMakerArgs...);

 public:
  /**
   * get the factory
   *
   * Using a static function variable gaurantees that the factory
   * is created as soon as it is needed and that it is deleted
   * before the program completes.
   */
  static Factory& get() {
    static Factory the_factory;
    return the_factory;
  }

  /**
   * register a new object to be constructible
   *
   * We insert the new object into the library after
   * checking that it hasn't been defined before.
   *
   * We throw a runtime_error exception if the object has been declared before.
   * This exception can easily be avoided by making sure the declaration
   * macro for a prototype links the name of the PrototypeMaker function to
   * the name of the derived class. This means the user would have a compile-time
   * error rather than a runtime exception.
   *
   * full_name - name to use as a reference for the declared object
   * maker - a pointer to a function that can dynamically create an instance
   */
  void declare(const std::string& full_name, PrototypeMaker maker) {
    auto lib_it{library_.find(full_name)};
    if (lib_it != library_.end()) {
      throw std::runtime_error("An object named "+full_name+" has already been declared.");
    }
    library_[full_name] = maker;
  }

  /**
   * make a new object by name
   *
   * We look through the library to find the requested object.
   * If found, we create one and return a pointer to the newly
   * created object. If not found, we raise an exception.
   *
   * The arguments to the maker are determined at compiletime
   * using the template parameters of Factory.
   *
   * full_name - name of object to create, same name as passed to declare
   * maker_args - parameter pack of arguments to pass on to maker
   *
   * Returns a pointer to the parent class that the objects derive from.
   */
  PrototypePtr make(const std::string& full_name, PrototypeMakerArgs... maker_args) {
    auto lib_it{library_.find(full_name)};
    if (lib_it == library_.end()) {
      throw std::runtime_error("An object named "+full_name+" has not been declared.");
    }
    return lib_it->second(maker_args...);
  }

  /// delete the copy constructor
  Factory(Factory const&) = delete;

  /// delete the assignment operator
  void operator=(Factory const&) = delete;

 private:
  /// private constructor to prevent creation
  Factory() = default;

  /// library of possible objects to create
  std::unordered_map<std::string,PrototypeMaker> library_;
};  // Factory

}  // namespace factory

#endif  // FACTORY_FACTORY_HPP
