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
 * from other other Factories are defined. There should be
 * nothing else in this namespace in order to avoid potential
 * name conflicts.
 */
namespace factory {

/**
 * Factory to create objects
 *
 * This factory is a singleton class meaning it cannot be
 * created by the user.
 *
 * The factory has three template parameters in order of complexity.
 * 1. Object - REQUIRED - the type of object that this factory creates.
 *    This should be the base class that all types in this factory derive from.
 * 2. ObjectPtr - optional - the type of pointer to object
 *    By default, we use std::unique_ptr for good memory management.
 * 3. ObjectMakerArgs - optional - type of objects passed into the object maker
 *    i.e. same as arguments to the constructor used by the object maker
 *
 * In order to save code repetition, it is suggested to typedef
 * your specific factory in your own namespace. This allows you to control
 * all the template inputs for your factory in one location.
 *
 *  typedef factory::Factory<MyObject> MyObjectFactory;
 *
 * Or, if you are in some other namespace, you can shorten it even more.
 *
 *  namespace foo {
 *    typedef factory::Factory<MyObject> Factory;
 *  }
 */
template<
  typename Object,
  typename ObjectPtr = std::unique_ptr<Object>,
  typename ... ObjectMakerArgs
  >
class Factory {
 public:
  /// the signature of a function that can be used by this factory
  typedef ObjectPtr (*ObjectMaker)(ObjectMakerArgs...);

 public:
  /**
   * get the factory
   *
   * Using a static function variable gaurantees that the factory
   * is created as soon as it is need and that it is deleted
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
   */
  void declare(const std::string& full_name, ObjectMaker maker) {
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
   * The arguments to the maker are determined at compiletime.
   */
  ObjectPtr make(const std::string& full_name, ObjectMakerArgs... maker_args) {
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
  std::unordered_map<std::string,ObjectMaker> library_;
};  // Factory

}  // namespace factory

#endif  // FACTORY_FACTORY_HPP
