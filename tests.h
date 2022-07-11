#include "structs_data.h"

#define _GNU_SOURCE

#include <unistd.h>

#include <stdio.h>

#include <sys/syscall.h>

#define NFUNS 1     // nombre de fois qu'on appelle la fonction à tester entre 2 mesures
#define NTEST 1000  // nombre de fois ou on repete le meme jeu de donnees
#define NSAMPLES 50 // nombre differents de jeu de donnees

inline uint64_t cpucyclesStart(void);
inline uint64_t cpucyclesStop(void);
inline unsigned long rdpmc_instructions(void);