# Factory

A factory for dynamically creating classes derived from a "prototype" by name.

## Terminology

- Factory: A object that has a look-up table between class names and pointers to functions that can create them
- Maker: A function that can create a specific class
- Prototype: A abstract base class from which derived classes can be used

## Design

The factory itself works on two levels.
First, all of the different derived classes "register" or "declare" themselves so that the factory
knows how to create them. This registration is done by providing a maker in association with the name of the derived class.
Second, the factory can create any of the registered classes and return a pointer to it in the form of a prototype pointer.
