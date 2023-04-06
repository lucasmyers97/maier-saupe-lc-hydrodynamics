# Notes on restructuring

Sometime soon I will need to restructure this entire repository so that it's actually manageable for one person.
To this end, here are some notes on deal.II's step-13 which deals with this explicitly.

## Notes on Step-13

### Evaluation class

* Define `Evaluation` abstract class which is just any postprocessing that is done on a solution object.
* Abstractly defines `()` operator which eats a `DoFHandler` and a `solution` object.
* Keeps everything const.
* Evaluation also takes up its own namespace.
* Create concrete class which evaluates solution at a point.
* Also define `SolutionOutput` class which just outputs solution with `DataOutBase`

### Solver class

* Here they make a very general solver class which does the following:
    - Solve a problem
    - Postprocess the solution with a list of evaluation objects
    - Refine the grid
* The thing only holds a pointer to a `Triangulation` object.
* The intantiation of a derived class gets passed:
    - A reference to a `Triangulation`
    - A const reference to a `FiniteElement`
    - A const reference to a `Quadrature` object
    - A const reference to a `Function` which functions as the boundary values.
* The instantiation owns:
    - `DofHandler`
    - `Vector`
    - Pointers to all the objects passed in
    - Linear system struct, defined in the class which holds:
        - `AffineConstraints`
        - `SparsityPattern`
        - `SparseMatrix`
        - `Vector` (for rhs)
    - Structs for parallel assembly
* To implement the right-hand-side construction, they derive another class from the `Solver` class which:
    - Basically calls the construction of the parent class.
    - And then has a class to construct the rhs.
* There are also two refinement classes, one which does Kelly, one which does global. Both are derived from the `Solver` class.
    - I feel like this is a lot of indirection.

## What `NematicSystemMPI` and `NematicSystemMPIDriver` actually do

### `NematicSystemMPI`

* `setup_dofs`
    - distributes fe
    - hanging node constraints
    - Dirichlet (if the simulation is running Dirichlet)
    - Internal fixed nodes (if freezing defects)
    - Sparsity pattern
    - Initialize matrix + rhs
    - Extra debug stuff that I need to get rid of.
* `initialize_fe_field`
    - configuration_constraints for inital value boundary values + hanging nodes
    - interpolate solution, also set past solution to that value
    - two versions: one interpolates solution with boundary-values, one takes in solution from outside
* `assemble_system`
    - have a whole slew of these which mix and match LdG vs MS and also time-stepping algorithms
* `solve_and_update`
    - solves system for Newton step and updates current solution
* `set_past_solution_to_current`
    - exactly what it says, mostly for data incapsulation
* `find_defects`
    - calls external function to find defects
    - stores in internal defect points repository
    - creates object to return which has new defect points
    - converts from vector of deal.II points to vector of vectors (should fix this)
* `calc_energy`
    - calculates each of the energy terms from the configuration
* `output_defect_positions`
    - outputs to hdf5, some hinky stuff with distributed vector
* `output_configuration_energies`
    - same as above, but for energies
* `output_results` and `output_Q_components`
    - use deal.II dataout objects to do output on field configuration
* `return_x`
    - lots of functions which return references to some private variable
* `return_defect_positions_at_time`
    - searches defect positions for ones at specific times

### `NematicSystemMPIDriver`

* `make_grid`
    - makes grid based on input type, globally refines it.
    - perhaps I should do this with `generate_from_name_and_arguments`.
* `refine_around_defects`
    - does initial further refines around defects
* `sort_defect_points`
    - figures out which past defect points the current defect points are closest to
    - essentially links defects up over time
* `recenter_defect_alignment`
    - coarsens and refines grid appropriately to follow defects
    - turns out to be a nontrivial operation
* `iterate_convex_splitting`
    - iterates in time using convex splitting method
* `iterate_forward_euler`
    - iterates in time using forward euler
* `iterate_semi_implicit`
    - iterates in time using semi implicit
* `iterate_timestep`
    - basically just chooses between each of the iteration schemes
* `run`
    - runs the whole simulation
    - declares and reads parameters
    - gets initial defect locations
    - makes grid
    - refines around defects
    - if defects are frozen, has to mark different areas of the domain
    - sets up dofs and initializes fe field
    - iterates timesteps a bunch of times, checking whether refines are necessary
    - outputs things that need to be output at the appropriate intervals
* 

## Handling parameters for these simulations

### Parameter structs

* Each object which has user-set parameters contains an internal struct which deals with those parameters.
* Any direct parameter of the class is stored as a member of that struct.
* Parameters which can be enumerated as options should be defined as `enum class` structures
* If the object owns some other object which also has parameters, the parameter struct will have an instance of the owned object's parameter struct.
* The parameter struct will have static functions `declare_parameters` and `get_parameters`. 
    - The former takes a mutable reference to a ParameterHandler object, and is const.
    - It recursively declares the parameters for all object parameters.
    - The latter takes a const reference to a ParameterHandler object, is const, and returns an instance of the given parameter struct.
    - It recursively gets all relevant values from a parameter file.
* For each enum class, the struct will also need to define a helper function which will parse a string input and give back a named enum.
* Note that deal.II offers some kind of "Convert" class which does these parameter conversions -- probably worth looking into which ones I can coopt for my own usage, and also perhaps extending this class via inheritance to cover my own use-cases.
