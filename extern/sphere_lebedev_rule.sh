#! /bin/bash
#
cp sphere_lebedev_rule.hpp /$HOME/include
#
g++ -c -Wall -I /$HOME/include sphere_lebedev_rule.cpp
if [ $? -ne 0 ]; then
  echo "Compile error."
  exit
fi
#
mv sphere_lebedev_rule.o ~/libcpp/sphere_lebedev_rule.o
#
echo "Normal end of execution."
