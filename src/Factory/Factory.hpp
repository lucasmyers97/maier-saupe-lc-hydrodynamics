#ifndef FUNCTIONFACTORY_FACTORY_H_
#define FUNCTIONFACTORY_FACTORY_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <exception>

namespace factory {

/**
 * Factory to create objects
 *
 * This factory is a singleton class meaning it cannot be
 * created by the user.
 */
template<
  typename Object,
  typename ObjectPtr = std::unique_ptr<Object>,
  typename ObjectMaker = ObjectPtr (*)()
  >
class Factory {
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
   */
  ObjectPtr make(const std::string& full_name) {
    auto lib_it{library_.find(full_name)};
    if (lib_it == library_.end()) {
      throw std::runtime_error("An object named "+full_name+" has not been declared.");
    }
    return lib_it->second();
  }

  /// delete the copy constructor
  Factory(Factory const&) = delete;

  /// delete the assignment operator
  void operator=(Factory const&) = delete;

 private:
  /// private constructor to prevent creation
  Factory() = default;

  /// library of possible functions to create
  std::unordered_map<std::string,ObjectMaker> library_;
};

}

#endif
