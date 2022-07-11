# pmns_phi52

This repository contains the source code for the RNS systems with moduli of size 52-bits, with both versions, sequential C and SIMD using AVX512 instruction set extension.

In order to perform the tests, it is necessary to follow the this procedure, in a shell:

- execute the `./rdpmc.sh` script
- go into the directory of the desired version
- type `make -B NB_COEFF=`
where `NB_COEFF` has to be either 8, 16, 24, 32, 40, 48, 56 or 64. If you dont specify `NB_COEFF`, the default is 8.

- then you can :
	- test the computations by typing `./fulltest512`
	- test the performances by typing `./timing512`
	
- you also may launch the test for all sizes : `make tests`
- or launch the performance measurments for all sizes : `make bench`

The Makefile specifies `gcc-10`, which is the gcc version we used in our tests. Previous versions may not be able to compile the source code as it is.
