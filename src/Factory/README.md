# Factory

A factory for dynamically creating classes derived from a "prototype" by name.

## Terminology

- Factory: A object that has a look-up table between class names and pointers to functions that can create them
- Maker: A function that can create a specific class
- Prototype: A abstract base class from which derived classes can be used

## Design

The factory itself works on two steps.

First, all of the different derived classes "register" or "declare" themselves so that the factory knows how to create them. 
This registration is done by providing a function that can construct them in association with the name of the derived class.

Second, the factory creates any of the registered classes and return a pointer to it in the form of a prototype-class pointer.

## Usage

Using the factory effectively can be done in situations where many classes all follow the same design structure but have
different implementations for specific steps. In order to reflect this "same design structure", we define an abstract
base class (a class with at least one pure virtual function) for all of our derived classes to inherit from.
This abstract base class is our "prototype".

```cpp
// LibraryEntry.hpp
#ifndef LIBRARYENTRY_HPP
#define LIBRARYENTRY_HPP
// we need the factory template
#include "Factory/Factory.hpp"

// this class is our prototype
class LibraryEntry {
 public:
  // virtual destructor so we can dynamically create derived classes
  virtual ~LibraryEntry() = default;
  // pure virtual function that our derived classes will implement
  virtual std::string name() = 0;
  // the factory type that we will use here
  using Factory = ::factory::Factory<LibraryEntry>;
};  // LibraryEntry

// a macro to help with registering our library entries with our factory
#define DECLARE_LIBRARYENTRY(NS, CLASS)                                \
  namespace NS {                                                       \
  std::unique_ptr<::LibraryEntry> CLASS##Maker() {                     \
    return std::make_unique<CLASS>();                                  \
  }                                                                    \
  __attribute__((constructor)) static void CLASS##Declare() {          \
    ::LibraryEntry::Factory.get().declare(                             \
        std::string(#NS) + "::" + std::string(#CLASS), &CLASS##Maker); \
  }                                                                    \
  }
#endif // LIBRARYENTRY_HPP
```

This `LibraryEntry` prototype satisfies our requirements. Now, we can define several other library entries in other source files.

```cpp
// Book.cpp
#include "LibraryEntry.hpp"
namespace library {
class Book : public LibraryEntry {
 public :
  virtual std::string name() final override {
    return "Where the Red Fern Grows";
  }
};
}

DECLARE_LIBRARYENTRY(library,Book)
```

```cpp
// Podcast.cpp
#include "LibraryEntry.hpp"
namespace library {
namespace audio {
class Podcast : public LibraryEntry {
 public :
  virtual std::string name() final override {
    return "538 Politics Podcast";
  }
};
}
}

DECLARE_LIBRARYENTRY(library::audio,Podcast)
```

```cpp
// Album.cpp
#include "LibraryEntry.hpp"
namespace library {
namespace audio {
class Album : public LibraryEntry {
 public :
  virtual std::string name() final override {
    return "Kind of Blue";
  }
};
}
}

DECLARE_LIBRARYENTRY(library::audio,Album)
```

Since the `DECLARE_LIBRARYENTRY` macro defines a function that is decorated with an attribute causing
it to be run at library-load time, all of these classes will be registered with the `LibraryEntry::Factory`
before `main` is begun. Therefore, we can define a `main` that can get these classes by name.

```
#include "LibraryEntry.hpp"

int main(int argc, char* argv[]) {
  std::string full_cpp_name{argv[1]}; 
  auto entry_ptr{LibraryEntry::Factory::get().make(full_cpp_name)};
  std::cout << entry_ptr->name() << std::endl;
}
```

Compiling this main into the `favorite-things` executable would then lead to the behavior.
```
$ favorite-things library::Book
Where the Red Fern Grows
$ favorite-things library::audio::Podcast
538 Politics Podcast
$ favorite-things library::audio::Album
Kind of Blue
```
