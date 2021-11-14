
// FunctionFactory
#include "Function.hpp"
#include <iostream>

namespace foo {

class Bar : public factory::Function {
 public:
  Bar() {
    std::cout << "I am a Bar" << std::endl;
  }
};

}

DECLARE_FUNCTION(foo,Bar);

int main() {
  // prints out "I am a Bar"
  std::unique_ptr<factory::Function> baz = factory::FunctionFactory::get().make("foo::Bar"); 
}
