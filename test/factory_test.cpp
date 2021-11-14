#include <boost/test/tools/interface.hpp>
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "Factory/Function.hpp"
#include "Factory/ConfigurableFunction.hpp"

namespace utf = boost::unit_test;

namespace functions {

class NoTemplate : public Function {
 
}; // NoTemplate

template <int N>
class Template : public Function {

}; // Template

typedef Template<2> Template2;

}

DECLARE_FUNCTION(functions,NoTemplate);
DECLARE_FUNCTION(functions,Template2);

namespace foo {

class TestConfig : public configurablefunctions::ConfigurableFunction {
 public:
  TestConfig(int i, double d) : ConfigurableFunction(i,d),i_{i},d_{d}{}
  int i_;
  double d_;
 
}; // NoTemplate

}

DECLARE_CONFIGURABLE_FUNCTION(foo,TestConfig);

BOOST_AUTO_TEST_CASE(factory_test) {
  std::unique_ptr<functions::Function> ptr;
  BOOST_CHECK_NO_THROW(ptr = functions::Factory::get().make("functions::NoTemplate"));
  BOOST_CHECK_NO_THROW(ptr = functions::Factory::get().make("functions::Template2"));
  BOOST_CHECK_THROW(ptr = functions::Factory::get().make("DNE"), std::runtime_error);

  std::unique_ptr<configurablefunctions::ConfigurableFunction> cptr;
  BOOST_CHECK_NO_THROW(cptr = configurablefunctions::Factory::get().make("foo::TestConfig",1,3.0));
  BOOST_CHECK(dynamic_cast<foo::TestConfig&>(*cptr).i_ == 1);
  BOOST_CHECK(dynamic_cast<foo::TestConfig&>(*cptr).d_ == 3.0);
}
