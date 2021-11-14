#include <boost/test/tools/interface.hpp>
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "Factory/Function.hpp"

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

BOOST_AUTO_TEST_CASE(factory_test) {
  std::unique_ptr<functions::Function> ptr;
  BOOST_CHECK_NO_THROW(ptr = functions::Factory::get().make("functions::NoTemplate"));
  BOOST_CHECK_NO_THROW(ptr = functions::Factory::get().make("functions::Template2"));
  BOOST_CHECK_THROW(ptr = functions::Factory::get().make("DNE"), std::runtime_error);
}
