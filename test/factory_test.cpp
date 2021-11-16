#include <boost/test/tools/interface.hpp>
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "Factory/Function.hpp"
#include "Factory/ConfigurableFunction.hpp"

namespace foo {

/**
 * One child class of the functions::Base
 *
 * I have to define the pure virtual function of the parent.
 * Here, I've also attached the 'final override' keywords
 * to signify that (1) there should be no child classes of this class
 * and (2) this function should be overriding a parent class function.
 *
 * This helps prevent future bugs at compile time.
 */
class NoTemplate : public functions::FunctionPrototype {
  int evaluate() final override {
   return 1;
  } 
}; // NoTemplate

/**
 * Same thing as above, but I'm going to return
 * the template parameter in evaluate.
 */
template <int N>
class Template : public functions::FunctionPrototype {
  int evaluate() final override {
   return N;
  } 
}; // Template

typedef Template<2> Template2;

class TestConfig : public functions::ConfigurablePrototype {
  int i_;
  double d_;
 public:
  TestConfig(int i, double d) 
    : functions::ConfigurablePrototype(i,d),i_{i},d_{d} {}
 
  double evaluate() final override {
    return i_ + d_;
  }
}; // NoTemplate

} // namespace foo

DECLARE_FUNCTION(foo,NoTemplate);
DECLARE_FUNCTION(foo,Template2);
DECLARE_CONFIGURABLE_FUNCTION(foo,TestConfig);

BOOST_AUTO_TEST_CASE(factory_test) {
  std::unique_ptr<functions::FunctionPrototype> ptr;
  BOOST_CHECK_NO_THROW(ptr = functions::FunctionPrototype::Factory::get().make("foo::NoTemplate"));
  BOOST_CHECK(ptr->evaluate() == 1);
  BOOST_CHECK_NO_THROW(ptr = functions::FunctionPrototype::Factory::get().make("foo::Template2"));
  BOOST_CHECK(ptr->evaluate() == 2);
  BOOST_CHECK_THROW(ptr = functions::FunctionPrototype::Factory::get().make("DNE"), std::runtime_error);

  std::unique_ptr<functions::ConfigurablePrototype> cptr;
  BOOST_CHECK_NO_THROW(cptr = functions::ConfigurablePrototype::Factory::get().make("foo::TestConfig",1,3.0));
  BOOST_CHECK(cptr->evaluate() == 4.0);
}
