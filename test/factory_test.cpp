#include <boost/test/tools/interface.hpp>
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "Factory/Function.hpp"
#include "Factory/ConfigurableFunction.hpp"

namespace foo {

class NoTemplate : public functions::Base {
 
}; // NoTemplate

template <int N>
class Template : public functions::Base {

}; // Template

typedef Template<2> Template2;

class TestConfig : public functions::configurable::Base {
 public:
  TestConfig(int i, double d) 
    : functions::configurable::Base(i,d),i_{i},d_{d} {}
  int i_;
  double d_;
 
}; // NoTemplate

} // namespace foo


DECLARE_FUNCTION(foo,NoTemplate);
DECLARE_FUNCTION(foo,Template2);
DECLARE_CONFIGURABLE_FUNCTION(foo,TestConfig);

BOOST_AUTO_TEST_CASE(factory_test) {
  std::unique_ptr<functions::Base> ptr;
  BOOST_CHECK_NO_THROW(ptr = functions::Factory::get().make("foo::NoTemplate"));
  BOOST_CHECK_NO_THROW(ptr = functions::Factory::get().make("foo::Template2"));
  BOOST_CHECK_THROW(ptr = functions::Factory::get().make("DNE"), std::runtime_error);

  std::unique_ptr<functions::configurable::Base> cptr;
  BOOST_CHECK_NO_THROW(cptr = functions::configurable::Factory::get().make("foo::TestConfig",1,3.0));
  BOOST_CHECK(dynamic_cast<foo::TestConfig&>(*cptr).i_ == 1);
  BOOST_CHECK(dynamic_cast<foo::TestConfig&>(*cptr).d_ == 3.0);
}
