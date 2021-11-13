
// FunctionFactory
#include "Factory.hpp"
#include "Function.hpp"
#include <iostream>

namespace foo {

class Bar : public functionfactory::Function {
 public:
  Bar() {
    std::cout << "I am a Bar" << std::endl;
  }
};

}

DECLARE_FUNCTION(foo,Bar);

int main() {
  // prints out "I am a Bar"
  std::unique_ptr<functionfactory::Function> baz = functionfactory::Factory::get().make("foo::Bar"); 
}
