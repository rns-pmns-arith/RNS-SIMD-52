#define _GNU_SOURCE

#include <unistd.h>

#include <stdio.h>

#include <sys/syscall.h>

#define NFUNS 1		 // nombre de fois qu'on appelle la fonction à tester entre 2 mesures
#define NTEST 500	//5000 // nombre de fois ou on repete le meme jeu de donnees
#define NSAMPLES 10 //100 // nombre differents de jeu de donnees

/**** Measurements procedures according to INTEL white paper

 "How to benchmark code execution times on INTEL IA-32 and IA-64" 
 
 *****/

// ATTENTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// Ne pas oublier de desactiver le turbo boost
// /bin/sh -c "/usr/bin/echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo"
// pour fiabiliser la mesure

inline static uint64_t cpucyclesStart(void)
{

	unsigned hi, lo;
	__asm__ __volatile__("CPUID\n\t"
						 "RDTSC\n\t"
						 "mov %%edx, %0\n\t"
						 "mov %%eax, %1\n\t"
						 : "=r"(hi), "=r"(lo)
						 :
						 : "%rax", "%rbx", "%rcx", "%rdx");

	return ((uint64_t)lo) ^ (((uint64_t)hi) << 32);
}

inline static uint64_t cpucyclesStop(void)
{

	unsigned hi, lo;
	__asm__ __volatile__("RDTSCP\n\t"
						 "mov %%edx, %0\n\t"
						 "mov %%eax, %1\n\t"
						 "CPUID\n\t"
						 : "=r"(hi), "=r"(lo)
						 :
						 : "%rax", "%rbx", "%rcx", "%rdx");

	return ((uint64_t)lo) ^ (((uint64_t)hi) << 32);
}

// rdpmc_instructions uses a "fixed-function" performance counter to return the count of retired instructions on
//       the current core in the low-order 48 bits of an unsigned 64-bit integer.
inline static unsigned long rdpmc_instructions(void)
{
	unsigned a, d, c;

	c = (1 << 30);
	__asm__ __volatile__("rdpmc"
						 : "=a"(a), "=d"(d)
						 : "c"(c));

	return ((unsigned long)a) | (((unsigned long)d) << 32);
	;
}

// rdpmc_actual_cycles uses a "fixed-function" performance counter to return the count of actual CPU core cycles
//       executed by the current core.  Core cycles are not accumulated while the processor is in the "HALT" state,
//       which is used when the operating system has no task(s) to run on a processor core.
unsigned long rdpmc_actual_cycles()
{
	unsigned a, d, c;

	c = (1 << 30) + 1;
	__asm__ volatile("rdpmc"
					 : "=a"(a), "=d"(d)
					 : "c"(c));

	return ((unsigned long)a) | (((unsigned long)d) << 32);
	;
}

// rdpmc_reference_cycles uses a "fixed-function" performance counter to return the count of "reference" (or "nominal")
//       CPU core cycles executed by the current core.  This counts at the same rate as the TSC, but does not count
//       when the core is in the "HALT" state.  If a timed section of code shows a larger change in TSC than in
//       rdpmc_reference_cycles, the processor probably spent some time in a HALT state.
unsigned long rdpmc_reference_cycles()
{
	unsigned a, d, c;

	c = (1 << 30) + 2;
	__asm__ volatile("rdpmc"
					 : "=a"(a), "=d"(d)
					 : "c"(c));

	return ((unsigned long)a) | (((unsigned long)d) << 32);
	;
}