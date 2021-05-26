# C++ Protoype of Stable Multiscale Finite Element Complexes (MsFEC)


This C++ code implements the *MsFEC* to demonstrate the method. *MsFEC* is a framework to 
construct stable pairings of multiscale finite elements throughout the entire L2-de Rham
complex when rough data is involved, i.e., we solve a scalar or vector valued modified
Laplace problem (hence we seek a Hodge decomposition). 
For further information build the documentation of the project.

---
**HINT**

MsFEC is **MPI parallel** and can be used on clusters to compute fairly
large problems - depending on your machine(s) up to **a few 100 milltion
unknowns** in 3D. 
---

Note that there is still room to optimize the implementation (e.g., faster linear solvers).

| **Documentation** |
|:-----------------:|
| [![][docs-latest-img]][docs-latest-url] |

---
**NOTE**

*MsFEC* requires:

* A Linux distribution (we used Debian and Ubuntu for the development)
* **cmake** v2.8.12 or higher	
* **doxygen**, **mathjax** and **GraphViz** (for the documentation)
* A working installation of **[deal.ii](www.dealii.org)** v9.1.1 or higher 
with **MPI**, **p4est** and all **Trilinos** dependencies must be installed. This
can easily be done through the **[spack](https://spack.readthedocs.io/en/latest/)** 
package manager
* **[Paraview](www.paraview.org)** or **[Visit](https://wci.llnl.gov/simulation/computer-codes/visit/)** 
for the visualization
* If you wish to modify the code **clang-format-6.0** (recommented to indent the code, 
available in most linux distributions) and a working **debugger** (we use gdb) is usually a good idea

---


### Building the Library and Linking Executables

To build the project together with Eclipse project files you must first clone the repository:

```
git clone https://github.com/konsim83/MPI-MSFEC.git MPI_MSFEC
```
We want an out-of-source-build with build files in a folder parallel to the code:

```
mkdir MPI_MSFEC_build
cd MPI_MSFEC_build
```
Then create the build files with `cmake`:

```
cmake -DDEAL_II_DIR=/path/to/dealii -DCMAKE_ECLIPSE_MAKE_ARGUMENTS=-j4 -G"Eclipse CDT4 - Unix Makefiles" ../MPI_MSFEC
```
You can now import an existing project in Eclipse. To generate the executable in debug mode type

```
make debug
make -jN
```
If you want to produce a faster reslease version type

```
make release
make -jN
```
To run the executable with an example parameter file run

```
mpirun -n N source/MsFEC_Ned_RT -p ../MSFEC/example_parameters/parameter_ned_rt.in
```
where N is now the number of MPI processes. This will run the code for the multiscale (modified) 
Nedelec-Raviart-Thomas pairing.

You should also be able to run this on clusters.

To run all tests type

```
ctest -V -R
```



### Building the Documentation

If you want to build the documentation locally 
you will need `doxygen`, `mathjax` and some other
packages such as `GraphViz` installed.

To build the documentation with `doxygen` enter the code folder

```
cd MPI_MSFEC/doc
```
and type

```
doxygen Doxyfile
```
This will generate a html documentation of classes in the `MPI_MSFEC/documentation/html` directory.
To open it open the `index.html` in a web browser.

[docs-latest-img]: https://img.shields.io/badge/Documentation-current-blue
[docs-latest-url]: https://konsim83.github.io/MPI-MSFEC/
