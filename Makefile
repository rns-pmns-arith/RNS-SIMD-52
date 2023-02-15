# Makefile

OPTIONS= -O3 -g -march=native -funroll-all-loops -Wno-overflow -lgmp

ifdef NB_COEFF
        MACROS= -DNB_COEFF=$(NB_COEFF)
else      
        MACROS= -DNB_COEFF=8  
endif

OPTIONS+=$(MACROS)


all: timing512 fulltest512

fulltest512: fulltest512.c rns.o rnsv_AVX512.o
	gcc-10 -o fulltest512 fulltest512.c rns.o rnsv_AVX512.o $(OPTIONS)

timing512: timing512.c rns.o rnsv_AVX512.o
	gcc-10 -o timing512 timing512.c rns.o rnsv_AVX512.o $(OPTIONS)

rns.o: rns.c rns.h structs_data.h 
	gcc-10 -c rns.c $(OPTIONS)
	
rnsv_AVX512.o: rnsv_AVX512.c rns.h structs_data.h 
	gcc-10 -c rnsv_AVX512.c $(OPTIONS)
	
tests:
	for size in 8 16 24 32 40 48 56 64 ; do \
		make -B fulltest512 NB_COEFF=$$size ; \
		./fulltest512 ; \
	done

bench:
	for size in 8 16 24 32 40 48 56 64 ; do \
		make -B timing512 NB_COEFF=$$size ; \
		./timing512 ; \
	done


clean:
	rm *.o timing512 fulltest512

