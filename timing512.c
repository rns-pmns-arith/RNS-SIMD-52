#include <stdlib.h>

#include <stdio.h>

#include <gmp.h>

#include <math.h>

#include "rns.h"

#include "tests.c"

#include "rnsv_AVX512.h"

int main(void)
{

	FILE *fpt, *fr;

	fpt = fopen("results/Results.json", "w+");
	fr = fopen("results/Resultes.csv", "w");


	fprintf(fpt, "{\n");

	printf("\nVectorized RNS timing :\n");

	//printf("\n1. Generating data :\n");

	// Initializing random
	gmp_randstate_t state;
	gmp_randinit_default(state);
	// Timers
	unsigned long long timer, t1, t2;
	// Variables
	int64_t op1[NB_COEFF];
	int64_t op2[NB_COEFF];
	int64_t resa[NB_COEFF];
	int64_t resb[NB_COEFF];
	__m512i avx512_op1[NB_COEFF / 8];
	__m512i avx512_op2[NB_COEFF / 8];
	__m512i avx512_resa[NB_COEFF / 8];
	__m512i avx512_resb[NB_COEFF / 8];
	
		mpz_t A, B, C;
	mpz_inits(A, B, C, NULL);


	#include "base64x52.h"

    // Base
    struct rns_base_t rns_a;
    rns_a.size = NB_COEFF;

    rns_a.m = m1;

    rns_a.k = k1;

    init_rns(&rns_a);
    avx512_init_rns(&rns_a);
 
    int64_t tmp_k[NB_COEFF];

    for (int j = 0; j < NB_COEFF; j++)
    {
        tmp_k[j] = (int64_t)k1[j];
    }

    __m512i avx512_k1[NB_COEFF / 8];
    from_int64_t_to_m512i_rns(avx512_k1, &rns_a, tmp_k);
    rns_a.avx512_k = avx512_k1;

    // Second Base
    struct rns_base_t rns_b;
    rns_b.size = NB_COEFF;

    rns_b.m = m2;
        
    rns_b.k = k2;

    init_rns(&rns_b);

    for (int j = 0; j < NB_COEFF; j++)
    {
        tmp_k[j] = (int64_t)k2[j];
    }

 
	init_rns(&rns_b);
    avx512_init_rns(&rns_b);



    __m512i avx512_k2[NB_COEFF / 8];
    from_int64_t_to_m512i_rns(avx512_k1, &rns_a, tmp_k);
    rns_b.avx512_k = avx512_k2;

	mpz_t M;
	mpz_inits(M, NULL);
	mpz_set(M, rns_a.M); // Get M from the base
	unsigned long long int timing = ULLONG_MAX;
	unsigned long before_cycles, after_cycles, cycles = ULONG_MAX;
	unsigned long before_instructions, after_instructions, instructions = ULONG_MAX;
	unsigned long before_ref, after_ref, ref = ULONG_MAX;

	printf("\n\tBase :\n");
	for (int i = 0; i < NB_COEFF; i++)
	{
		printf("\t\t %ld\n", m1[i]);
	}

	printf("\n1. Multiplication :\n");

	fprintf(fpt, "\"multiplication\" :\n\t{\n");
	fprintf(fpt, "\t\"sequential\" :\n\t\t[\n");

	printf("\n\tHeating caches... ");
	mpz_urandomm(A, state, M); // Randomly generates A < M
	mpz_urandomm(B, state, M); // Randomly generated B < M
	from_int_to_rns(op1, &rns_a, A);
	from_int_to_rns(op2, &rns_a, B);

	for (int i = 0; i < NTEST; i++)
	{
		mul_rns_cr(resa, &rns_a, op1, op2);
	}

	printf("Done.\n");

	printf("\tTesting... ");

	for (int i = 0; i < NSAMPLES; i++)
	{
		mpz_urandomm(A, state, M); //Randomly generates A < M
		mpz_urandomm(B, state, M); //Randomly generates B < M
		from_int_to_rns(op1, &rns_a, A);
		from_int_to_rns(op2, &rns_a, B);
		for (int j = 0; j < NTEST; j++)
		{

			// RDTSC
			t1 = cpucyclesStart();

			mul_rns_cr(resa, &rns_a, op1, op2);

			t2 = cpucyclesStop();

			if (timing > (t2 - t1) / NFUNS)
				timing = (t2 - t1) / NFUNS;

			// Instructions
			before_instructions = rdpmc_instructions();

			mul_rns_cr(resa, &rns_a, op1, op2);

			after_instructions = rdpmc_instructions();

			if (instructions > (after_instructions - before_instructions) / NFUNS)
				instructions = (after_instructions - before_instructions) / NFUNS;

			// actual cycles
			before_cycles = rdpmc_actual_cycles();

			mul_rns_cr(resa, &rns_a, op1, op2);

			after_cycles = rdpmc_actual_cycles();

			if (cycles > (after_cycles - before_cycles) / NFUNS)
				cycles = (after_cycles - before_cycles) / NFUNS;

			// reference cycles
			before_ref = rdpmc_reference_cycles();

			mul_rns_cr(resa, &rns_a, op1, op2);

			after_ref = rdpmc_reference_cycles();

			if (ref > (after_ref - before_ref) / NFUNS)
				ref = (after_ref - before_ref) / NFUNS;
		}
		fprintf(fpt, "\t\t\t{\n");
		fprintf(fpt, "\t\t\t\t\"TSC Cycles\" : %lld,\n\t\t\t\t\"Instructions\" : %ld,\n\t\t\t\t\"Actual cycles\" : %ld,\n\t\t\t\t\"Reference cycles\" : %ld\n\t\t\t}", timing, instructions, cycles, ref);
		if (i < NSAMPLES - 1)
			fprintf(fpt, ",");
		fprintf(fpt, "\n");
	}
	fprintf(fpt, "\t\t],\n");

	printf("Done.\n");
	printf("\tRNS sequential multiplication : %lld CPU cycles.\n", timing);
	printf("\tRNS sequential multiplication : %ld instructions.\n", instructions);
	printf("\tRNS sequential multiplication : %ld actual CPU cycles.\n", cycles);
	printf("\tRNS sequential multiplication : %ld reference CPU cycles.\n", ref);

fprintf(fr, "Multiplication; Séquentiel; %d; %d; %ld; %lld; %ld; %ld\n", NB_COEFF, TAILLE_MODULE, instructions, timing, cycles, ref);


	timing = ULLONG_MAX;
	cycles = ULONG_MAX;
	instructions = ULONG_MAX;
	ref = ULONG_MAX;

	//printf("\n\tHeating caches... ");
	mpz_urandomm(A, state, M); //Randomly generates A < M
	mpz_urandomm(B, state, M); //Randomly generates B < M

	from_int_to_rns(op1, &rns_a, A);
	from_int_to_rns(op2, &rns_a, B);
	from_int64_t_to_m512i_rns(avx512_op1, &rns_a, op1);
	from_int64_t_to_m512i_rns(avx512_op2, &rns_a, op2);

	for (int i = 0; i < NTEST; i++)
	{

		avx512_mul_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);
	}

	//printf("Done.\n");

	printf("\tTesting... ");

	// timing
	for (int i = 0; i < NSAMPLES; i++)
	{
		mpz_urandomm(A, state, M); //Randomly generates A < M
		mpz_urandomm(B, state, M); //Randomly generates B < M
		from_int_to_rns(op1, &rns_a, A);
		from_int_to_rns(op2, &rns_a, B);
		from_int64_t_to_m512i_rns(avx512_op1, &rns_a, op1);
		from_int64_t_to_m512i_rns(avx512_op2, &rns_a, op2);
		for (int j = 0; j < NTEST; j++)
		{

			// RDTSC
			t1 = cpucyclesStart();

			avx512_mul_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);
			avx512_mul_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);
			avx512_mul_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);
			avx512_mul_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);
			avx512_mul_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);
			avx512_mul_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);
			avx512_mul_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);
			avx512_mul_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);
			avx512_mul_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);
			avx512_mul_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);
			t2 = cpucyclesStop();

			if (timing > (t2 - t1) / NFUNS)
				timing = (t2 - t1) / NFUNS;

			// Instructions
			before_instructions = rdpmc_instructions();

			avx512_mul_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);
			avx512_mul_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);
			avx512_mul_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);
			avx512_mul_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);
			avx512_mul_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);
			avx512_mul_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);
			avx512_mul_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);
			avx512_mul_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);
			avx512_mul_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);
			avx512_mul_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);

			after_instructions = rdpmc_instructions();

			if (instructions > (after_instructions - before_instructions) / NFUNS)
				instructions = (after_instructions - before_instructions) / NFUNS;

			// actual cycles
			before_cycles = rdpmc_actual_cycles();

			avx512_mul_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);
			avx512_mul_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);
			avx512_mul_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);
			avx512_mul_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);
			avx512_mul_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);
			avx512_mul_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);
			avx512_mul_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);
			avx512_mul_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);
			avx512_mul_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);
			avx512_mul_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);

			after_cycles = rdpmc_actual_cycles();

			if (cycles > (after_cycles - before_cycles) / NFUNS)
				cycles = (after_cycles - before_cycles) / NFUNS;

			// reference cycles
			before_ref = rdpmc_reference_cycles();

			avx512_mul_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);
			avx512_mul_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);
			avx512_mul_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);
			avx512_mul_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);
			avx512_mul_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);
			avx512_mul_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);
			avx512_mul_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);
			avx512_mul_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);
			avx512_mul_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);
			avx512_mul_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);

			after_ref = rdpmc_reference_cycles();

			if (ref > (after_ref - before_ref) / NFUNS)
				ref = (after_ref - before_ref) / NFUNS;
		}

		fprintf(fpt, "\t\t\t{\n");
		fprintf(fpt, "\t\t\t\t\"TSC Cycles\" : %lld,\n\t\t\t\t\"Instructions\" : %ld,\n\t\t\t\t\"Actual cycles\" : %ld,\n\t\t\t\t\"Reference cycles\" : %ld\n\t\t\t}", timing, instructions, cycles, ref);
		if (i < NSAMPLES - 1)
			fprintf(fpt, ",");
		fprintf(fpt, "\n");
	}

	fprintf(fpt, "\t\t]\n\t},\n");

	printf("Done AVX512.\n");
	printf("\tRNS vectorized 512 multiplication : %lld CPU cycles.\n", timing/10);
	printf("\tRNS vectorized 512 multiplication : %ld instructions.\n", instructions/10);
	printf("\tRNS vectorized 512 multiplication : %ld actual CPU cycles.\n", cycles/10);
	printf("\tRNS vectorized 512 multiplication : %ld reference CPU cycles.\n", ref/10);

fprintf(fr, "Multiplication; AVX-512; %d; %d; %ld; %lld; %ld; %ld\n", NB_COEFF, TAILLE_MODULE, instructions, timing, cycles, ref);

///////////////////////////////////////////////////////////////////////////////

	printf("\n\n2. Addition :\n");
	fprintf(fpt, "\"addition\" :\n\t{\n");
	fprintf(fpt, "\t\"sequential\" :\n\t\t[\n");

	timing = ULLONG_MAX;
	cycles = ULONG_MAX;
	instructions = ULONG_MAX;
	ref = ULONG_MAX;

	//printf("\n\tHeating caches... ");
	mpz_urandomm(A, state, M); // Randomly generates A < M
	mpz_urandomm(B, state, M); // Randomly generated B < M
	from_int_to_rns(op1, &rns_a, A);
	from_int_to_rns(op2, &rns_a, B);

	for (int i = 0; i < NTEST; i++)
	{
		add_rns_cr(resa, &rns_a, op1, op2);
	}
	//printf("Done.\n");

	//printf("\tTesting... ");

	for (int i = 0; i < NSAMPLES; i++)
	{
		mpz_urandomm(A, state, M); //Randomly generates A < M
		mpz_urandomm(B, state, M); //Randomly generates B < M
		from_int_to_rns(op1, &rns_a, A);
		from_int_to_rns(op2, &rns_a, B);
		for (int j = 0; j < NTEST; j++)
		{

			// RDTSC
			t1 = cpucyclesStart();

			add_rns_cr(resa, &rns_a, op1, op2);

			t2 = cpucyclesStop();

			if (timing > (t2 - t1) / NFUNS)
				timing = (t2 - t1) / NFUNS;

			// Instructions
			before_instructions = rdpmc_instructions();

			add_rns_cr(resa, &rns_a, op1, op2);

			after_instructions = rdpmc_instructions();

			if (instructions > (after_instructions - before_instructions) / NFUNS)
				instructions = (after_instructions - before_instructions) / NFUNS;

			// actual cycles
			before_cycles = rdpmc_actual_cycles();

			add_rns_cr(resa, &rns_a, op1, op2);

			after_cycles = rdpmc_actual_cycles();

			if (cycles > (after_cycles - before_cycles) / NFUNS)
				cycles = (after_cycles - before_cycles) / NFUNS;

			// reference cycles
			before_ref = rdpmc_reference_cycles();

			add_rns_cr(resa, &rns_a, op1, op2);

			after_ref = rdpmc_reference_cycles();

			if (ref > (after_ref - before_ref) / NFUNS)
				ref = (after_ref - before_ref) / NFUNS;
		}

		fprintf(fpt, "\t\t\t{\n");
		fprintf(fpt, "\t\t\t\t\"TSC Cycles\" : %lld,\n\t\t\t\t\"Instructions\" : %ld,\n\t\t\t\t\"Actual cycles\" : %ld,\n\t\t\t\t\"Reference cycles\" : %ld\n\t\t\t}", timing, instructions, cycles, ref);
		if (i < NSAMPLES - 1)
			fprintf(fpt, ",");
		fprintf(fpt, "\n");
	}

	fprintf(fpt, "\t\t],\n");


fprintf(fr, "Addition; Séquentiel; %d; %d; %ld; %lld; %ld; %ld\n", NB_COEFF, TAILLE_MODULE, instructions, timing, cycles, ref);





	printf("Done.\n");
	printf("\tRNS sequential addition : %lld CPU cycles.\n", timing);
	printf("\tRNS sequential addition : %ld instructions.\n", instructions);
	printf("\tRNS sequential addition : %ld actual CPU cycles.\n", cycles);
	printf("\tRNS sequential addition : %ld reference CPU cycles.\n", ref);


	fprintf(fpt, "\t\"parallel\" :\n\t\t[\n");

	timing = ULLONG_MAX;
	cycles = ULONG_MAX;
	instructions = ULONG_MAX;
	ref = ULONG_MAX;

//	printf("\n\tHeating caches... ");
	mpz_urandomm(A, state, M); //Randomly generates A < M
	mpz_urandomm(B, state, M); //Randomly generates B < M

	from_int_to_rns(op1, &rns_a, A);
	from_int_to_rns(op2, &rns_a, B);
	from_int64_t_to_m512i_rns(avx512_op1, &rns_a, op1);
	from_int64_t_to_m512i_rns(avx512_op2, &rns_a, op2);

	for (int i = 0; i < NTEST; i++)
	{

		avx512_add_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);
	}

//	printf("Done.\n");

	printf("\tTesting... ");

	// timing
	for (int i = 0; i < NSAMPLES; i++)
	{
		mpz_urandomm(A, state, M); //Randomly generates A < M
		mpz_urandomm(B, state, M); //Randomly generates B < M
		from_int_to_rns(op1, &rns_a, A);
		from_int_to_rns(op2, &rns_a, B);
		from_int64_t_to_m512i_rns(avx512_op1, &rns_a, op1);
		from_int64_t_to_m512i_rns(avx512_op2, &rns_a, op2);
		for (int j = 0; j < NTEST; j++)
		{

			// RDTSC
			t1 = cpucyclesStart();

			avx512_add_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);

			t2 = cpucyclesStop();

			if (timing > (t2 - t1) / NFUNS)
				timing = (t2 - t1) / NFUNS;

			// Instructions
			before_instructions = rdpmc_instructions();

			avx512_add_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);

			after_instructions = rdpmc_instructions();

			if (instructions > (after_instructions - before_instructions) / NFUNS)
				instructions = (after_instructions - before_instructions) / NFUNS;

			// actual cycles
			before_cycles = rdpmc_actual_cycles();

			avx512_add_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);

			after_cycles = rdpmc_actual_cycles();

			if (cycles > (after_cycles - before_cycles) / NFUNS)
				cycles = (after_cycles - before_cycles) / NFUNS;

			// reference cycles
			before_ref = rdpmc_reference_cycles();

			avx512_add_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);

			after_ref = rdpmc_reference_cycles();
			//printf("%ld\n", after_ref-before_ref);

			if (ref > (after_ref - before_ref) / NFUNS)
				ref = (after_ref - before_ref) / NFUNS;
		}

		fprintf(fpt, "\t\t\t{\n");
		fprintf(fpt, "\t\t\t\t\"TSC Cycles\" : %lld,\n\t\t\t\t\"Instructions\" : %ld,\n\t\t\t\t\"Actual cycles\" : %ld,\n\t\t\t\t\"Reference cycles\" : %ld\n\t\t\t}", timing, instructions, cycles, ref);
		if (i < NSAMPLES - 1)
			fprintf(fpt, ",");
		fprintf(fpt, "\n");
	}

	fprintf(fpt, "\t\t]\n\t},\n");

	printf("Done.\n");
	printf("\tRNS vectorized 512 addition : %lld CPU cycles.\n", timing);
	printf("\tRNS vectorized 512 addition : %ld instructions.\n", instructions);
	printf("\tRNS vectorized 512 addition : %ld actual CPU cycles.\n", cycles);
	printf("\tRNS vectorized 512 addition : %ld reference CPU cycles.\n", ref);

fprintf(fr, "Addition; AVX-512; %d; %d; %ld; %lld; %ld; %ld\n", NB_COEFF, TAILLE_MODULE, instructions, timing, cycles, ref);


//////////////////////////////////////////////////////


	printf("\n\n3. Substraction :\n");

	fprintf(fpt, "\"substraction\" :\n\t{\n");
	fprintf(fpt, "\t\"sequential\" :\n\t\t[\n");

	timing = ULLONG_MAX;
	cycles = ULONG_MAX;
	instructions = ULONG_MAX;
	ref = ULONG_MAX;

	printf("\n\tHeating caches... ");
	mpz_urandomm(A, state, M); // Randomly generates A < M
	mpz_urandomm(B, state, M); // Randomly generated B < M
	from_int_to_rns(op1, &rns_a, A);
	from_int_to_rns(op2, &rns_a, B);

	for (int i = 0; i < NTEST; i++)
	{
		sub_rns_cr(resa, &rns_a, op1, op2);
	}
	printf("Done.\n");

	printf("\tTesting... ");

	for (int i = 0; i < NSAMPLES; i++)
	{
		mpz_urandomm(A, state, M); //Randomly generates A < M
		mpz_urandomm(B, state, M); //Randomly generates B < M
		from_int_to_rns(op1, &rns_a, A);
		from_int_to_rns(op2, &rns_a, B);
		for (int j = 0; j < NTEST; j++)
		{

			// RDTSC
			t1 = cpucyclesStart();

			sub_rns_cr(resa, &rns_a, op1, op2);

			t2 = cpucyclesStop();

			if (timing > (t2 - t1) / NFUNS)
				timing = (t2 - t1) / NFUNS;

			// Instructions
			before_instructions = rdpmc_instructions();

			sub_rns_cr(resa, &rns_a, op1, op2);

			after_instructions = rdpmc_instructions();

			if (instructions > (after_instructions - before_instructions) / NFUNS)
				instructions = (after_instructions - before_instructions) / NFUNS;

			// actual cycles
			before_cycles = rdpmc_actual_cycles();

			sub_rns_cr(resa, &rns_a, op1, op2);

			after_cycles = rdpmc_actual_cycles();

			if (cycles > (after_cycles - before_cycles) / NFUNS)
				cycles = (after_cycles - before_cycles) / NFUNS;

			// reference cycles
			before_ref = rdpmc_reference_cycles();

			sub_rns_cr(resa, &rns_a, op1, op2);

			after_ref = rdpmc_reference_cycles();

			if (ref > (after_ref - before_ref) / NFUNS)
				ref = (after_ref - before_ref) / NFUNS;
		}

		fprintf(fpt, "\t\t\t{\n");
		fprintf(fpt, "\t\t\t\t\"TSC Cycles\" : %lld,\n\t\t\t\t\"Instructions\" : %ld,\n\t\t\t\t\"Actual cycles\" : %ld,\n\t\t\t\t\"Reference cycles\" : %ld\n\t\t\t}", timing, instructions, cycles, ref);
		if (i < NSAMPLES - 1)
			fprintf(fpt, ",");
		fprintf(fpt, "\n");
	}

	fprintf(fpt, "\t\t],\n");

	printf("Done.\n");
	printf("\tRNS sequential substraction : %lld CPU cycles.\n", timing);
	printf("\tRNS sequential substraction : %ld instructions.\n", instructions);
	printf("\tRNS sequential substraction : %ld actual CPU cycles.\n", cycles);
	printf("\tRNS sequential substraction : %ld reference CPU cycles.\n", ref);

fprintf(fr, "Soustraction; Séquentiel; %d; %d; %ld; %lld; %ld; %ld\n", NB_COEFF, TAILLE_MODULE, instructions, timing, cycles, ref);



	fprintf(fpt, "\t\"parallel\" :\n\t\t[\n");

	timing = ULLONG_MAX;
	cycles = ULONG_MAX;
	instructions = ULONG_MAX;
	ref = ULONG_MAX;

//	printf("\n\tHeating caches... ");
	mpz_urandomm(A, state, M); //Randomly generates A < M
	mpz_urandomm(B, state, M); //Randomly generates B < M

	from_int_to_rns(op1, &rns_a, A);
	from_int_to_rns(op2, &rns_a, B);
	from_int64_t_to_m512i_rns(avx512_op1, &rns_a, op1);
	from_int64_t_to_m512i_rns(avx512_op2, &rns_a, op2);

	for (int i = 0; i < NTEST; i++)
	{

		avx512_sub_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);
	}

//	printf("Done.\n");

	printf("\tTesting... ");

	// timing
	for (int i = 0; i < NSAMPLES; i++)
	{
		mpz_urandomm(A, state, M); //Randomly generates A < M
		mpz_urandomm(B, state, M); //Randomly generates B < M
		from_int_to_rns(op1, &rns_a, A);
		from_int_to_rns(op2, &rns_a, B);
		from_int64_t_to_m512i_rns(avx512_op1, &rns_a, op1);
		from_int64_t_to_m512i_rns(avx512_op2, &rns_a, op2);
		for (int j = 0; j < NTEST; j++)
		{

			// RDTSC
			t1 = cpucyclesStart();

			avx512_sub_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);

			t2 = cpucyclesStop();

			if (timing > (t2 - t1) / NFUNS)
				timing = (t2 - t1) / NFUNS;

			// Instructions
			before_instructions = rdpmc_instructions();

			avx512_sub_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);

			after_instructions = rdpmc_instructions();

			if (instructions > (after_instructions - before_instructions) / NFUNS)
				instructions = (after_instructions - before_instructions) / NFUNS;

			// actual cycles
			before_cycles = rdpmc_actual_cycles();

			avx512_sub_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);

			after_cycles = rdpmc_actual_cycles();

			if (cycles > (after_cycles - before_cycles) / NFUNS)
				cycles = (after_cycles - before_cycles) / NFUNS;

			// reference cycles
			before_ref = rdpmc_reference_cycles();

			avx512_sub_rns_cr(avx512_resa, &rns_a, avx512_op1, avx512_op2);

			after_ref = rdpmc_reference_cycles();

			if (ref > (after_ref - before_ref) / NFUNS)
				ref = (after_ref - before_ref) / NFUNS;
		}

		fprintf(fpt, "\t\t\t{\n");
		fprintf(fpt, "\t\t\t\t\"TSC Cycles\" : %lld,\n\t\t\t\t\"Instructions\" : %ld,\n\t\t\t\t\"Actual cycles\" : %ld,\n\t\t\t\t\"Reference cycles\" : %ld\n\t\t\t}", timing, instructions, cycles, ref);
		if (i < NSAMPLES - 1)
			fprintf(fpt, ",");
		fprintf(fpt, "\n");
	}

	fprintf(fpt, "\t\t]\n\t},\n");

	printf("Done.\n");
	printf("\tRNS vectorized 512 substraction : %lld CPU cycles.\n", timing);
	printf("\tRNS vectorized 512 substraction : %ld instructions.\n", instructions);
	printf("\tRNS vectorized 512 substraction : %ld actual CPU cycles.\n", cycles);
	printf("\tRNS vectorized 512 substraction : %ld reference CPU cycles.\n", ref);

fprintf(fr, "Soustraction; AVX-512; %d; %d; %ld; %lld; %ld; %ld\n", NB_COEFF, TAILLE_MODULE, instructions, timing, cycles, ref);


	// Base conversion Base 1
	struct conv_base_t conv_atob;
	conv_atob.rns_a = &rns_a;
	conv_atob.rns_b = &rns_b;
	initialize_inverses_base_conversion(&conv_atob);

	int64_t ttt[NB_COEFF];

	timing = ULLONG_MAX;
	cycles = ULONG_MAX;
	instructions = ULONG_MAX;
	ref = ULONG_MAX;

	printf("\n\n4. Base conversion :\n");

	fprintf(fpt, "\"base_conversion\" :\n\t{\n");
	fprintf(fpt, "\t\"sequential\" :\n\t\t[\n");

	printf("\n\tHeating caches... ");
	mpz_urandomm(A, state, M); // Randomly generates A < M
	mpz_urandomm(B, state, M); // Randomly generated B < M
	from_int_to_rns(op1, &rns_a, A);
	from_int_to_rns(op2, &rns_a, B);

	for (int i = 0; i < NTEST; i++)
	{
		base_conversion_cr(op2, &conv_atob, op1, ttt);
	}
	printf("Done.\n");

	printf("\tTesting... ");

	for (int i = 0; i < NSAMPLES; i++)
	{
		mpz_urandomm(A, state, M); //Randomly generates A < M
		mpz_urandomm(B, state, M); //Randomly generates B < M
		from_int_to_rns(op1, &rns_a, A);
		for (int j = 0; j < NTEST; j++)
		{

			// RDTSC
			t1 = cpucyclesStart();

			base_conversion_cr(op2, &conv_atob, op1, ttt);

			t2 = cpucyclesStop();

			if (timing > (t2 - t1) / NFUNS)
				timing = (t2 - t1) / NFUNS;

			// Instructions
			before_instructions = rdpmc_instructions();

			base_conversion_cr(op2, &conv_atob, op1, ttt);

			after_instructions = rdpmc_instructions();

			if (instructions > (after_instructions - before_instructions) / NFUNS)
				instructions = (after_instructions - before_instructions) / NFUNS;

			// actual cycles
			before_cycles = rdpmc_actual_cycles();

			base_conversion_cr(op2, &conv_atob, op1, ttt);

			after_cycles = rdpmc_actual_cycles();

			if (cycles > (after_cycles - before_cycles) / NFUNS)
				cycles = (after_cycles - before_cycles) / NFUNS;

			// reference cycles
			before_ref = rdpmc_reference_cycles();

			base_conversion_cr(op2, &conv_atob, op1, ttt);

			after_ref = rdpmc_reference_cycles();

			if (ref > (after_ref - before_ref) / NFUNS)
				ref = (after_ref - before_ref) / NFUNS;
		}

		fprintf(fpt, "\t\t\t{\n");
		fprintf(fpt, "\t\t\t\t\"TSC Cycles\" : %lld,\n\t\t\t\t\"Instructions\" : %ld,\n\t\t\t\t\"Actual cycles\" : %ld,\n\t\t\t\t\"Reference cycles\" : %ld\n\t\t\t}", timing, instructions, cycles, ref);
		if (i < NSAMPLES - 1)
			fprintf(fpt, ",");
		fprintf(fpt, "\n");
	}

	fprintf(fpt, "\t\t],\n");

	printf("Done.\n");
	printf("\tRNS sequential base conversion : %lld CPU cycles.\n", timing);
	printf("\tRNS sequential base conversion : %ld instructions.\n", instructions);
	printf("\tRNS sequential base conversion : %ld actual CPU cycles.\n", cycles);
	printf("\tRNS sequential base conversion : %ld reference CPU cycles.\n", ref);

fprintf(fr, "Base conversion; Séquentiel; %d; %d; %ld; %lld; %ld; %ld\n", NB_COEFF, TAILLE_MODULE, instructions, timing, cycles, ref);

	fprintf(fpt, "\t\"parallel\" :\n\t\t[\n");

	timing = ULLONG_MAX;
	cycles = ULONG_MAX;
	instructions = ULONG_MAX;
	ref = ULONG_MAX;

//	printf("\n\tHeating caches... ");
	mpz_urandomm(A, state, M); //Randomly generates A < M

//printf("%p \n", conv_atob.avx512_mrsa_to_b);

	avx512_init_mrs(&conv_atob);
	avx512_initialize_inverses_base_conversion(&conv_atob);
//printf("%p \n", conv_atob.avx512_mrsa_to_b);

	from_int_to_rns(op1, &rns_a, A);
	from_int64_t_to_m512i_rns(avx512_op1, &rns_a, op1);
//printf("%p \n", conv_atob.avx512_mrsa_to_b);

	for (int i = 0; i < NTEST; i++)
	{
		avx512_base_conversion_cr(avx512_op2, &conv_atob, avx512_op1, op1);
		
	}

//	printf("Done.\n");

	printf("\tTesting... ");

	// timing
	for (int i = 0; i < NSAMPLES; i++)
	{
		mpz_urandomm(A, state, M); //Randomly generates A < M
		from_int_to_rns(op1, &rns_a, A);
		from_int64_t_to_m512i_rns(avx512_op1, &rns_a, op1);
		for (int j = 0; j < NTEST; j++)
		{
			// RDTSC
			t1 = cpucyclesStart();

			avx512_base_conversion_cr(avx512_op2, &conv_atob, avx512_op1, op1);

			t2 = cpucyclesStop();

			if (timing > (t2 - t1) / NFUNS)
				timing = (t2 - t1) / NFUNS;

			// Instructions
			before_instructions = rdpmc_instructions();

			avx512_base_conversion_cr(avx512_op2, &conv_atob, avx512_op1, op1);

			after_instructions = rdpmc_instructions();

			if (instructions > (after_instructions - before_instructions) / NFUNS)
				instructions = (after_instructions - before_instructions) / NFUNS;

			// actual cycles
			before_cycles = rdpmc_actual_cycles();

			avx512_base_conversion_cr(avx512_op2, &conv_atob, avx512_op1, op1);

			after_cycles = rdpmc_actual_cycles();

			if (cycles > (after_cycles - before_cycles) / NFUNS)
				cycles = (after_cycles - before_cycles) / NFUNS;

			// reference cycles
			before_ref = rdpmc_reference_cycles();

			avx512_base_conversion_cr(avx512_op2, &conv_atob, avx512_op1, op1);

			after_ref = rdpmc_reference_cycles();

			if (ref > (after_ref - before_ref) / NFUNS)
				ref = (after_ref - before_ref) / NFUNS;
		}

		fprintf(fpt, "\t\t\t{\n");
		fprintf(fpt, "\t\t\t\t\"TSC Cycles\" : %lld,\n\t\t\t\t\"Instructions\" : %ld,\n\t\t\t\t\"Actual cycles\" : %ld,\n\t\t\t\t\"Reference cycles\" : %ld\n\t\t\t}", timing, instructions, cycles, ref);
		if (i < NSAMPLES - 1)
			fprintf(fpt, ",");
		fprintf(fpt, "\n");
	}

	fprintf(fpt, "\t\t]\n\t},\n");

	printf("Done.\n");
	printf("\tRNS vectorized 512 base conversion : %lld CPU cycles.\n", timing);
	printf("\tRNS vectorized 512 base conversion : %ld instructions.\n", instructions);
	printf("\tRNS vectorized 512 base conversion : %ld actual CPU cycles.\n", cycles);
	printf("\tRNS vectorized 512 base conversion : %ld reference CPU cycles.\n", ref);

fprintf(fr, "Base conversion; AVX-512; %d; %d; %ld; %lld; %ld; %ld\n", NB_COEFF, TAILLE_MODULE, instructions, timing, cycles, ref);












	timing = ULLONG_MAX;
	cycles = ULONG_MAX;
	instructions = ULONG_MAX;
	ref = ULONG_MAX;

	// MODULAR MULTIPLICATION
	printf("\n\n5. Modular multiplication (Mixed Radix) :\n");

	fprintf(fpt, "\"modular_multiplication\" :\n\t{\n");
	fprintf(fpt, "\t\"sequential\" :\n\t\t[\n");

	mpz_t inv_p_modM, inv_M_modMp, modul_p;
	mpz_inits(inv_p_modM, inv_M_modMp, modul_p, NULL);

	int64_t pa[NB_COEFF];
	int64_t pb[NB_COEFF];
	int64_t pab[NB_COEFF];
	int64_t pbb[NB_COEFF];
	int64_t pca[NB_COEFF];
	int64_t pcb[NB_COEFF];
	int64_t pp1[NB_COEFF];
	int64_t pp2[NB_COEFF];
	int64_t pp3[NB_COEFF];

	// Set custom values
	mpz_set_str(modul_p, "3653232203086218934955973655030869070722999178105346302138244960631343306387218512293583627872011030526507750192262510093", 10);
	mpz_set_str(inv_p_modM, "-7210642370083763919688086698199040857322895088554003933210287226647459666846134833419938084604981461493089686639677942359747717700454441525223348684285", 10);
	mpz_set_str(inv_M_modMp, "2926906825829426928727294150364906856635623568440932569450673109926460590684432927230290255276608760237299661987870702836538185953568700154975953006659", 10);

	// Initialization
	base_conversion_cr(pb, &conv_atob, pa, ttt);

	//Modular multiplication

	// Base conversion Base 2
	struct conv_base_t conv_btoa;
	conv_btoa.rns_a = &rns_b;
	conv_btoa.rns_b = &rns_a;
	initialize_inverses_base_conversion(&conv_btoa);

	struct mod_mul_t mult;
	mpz_t tmp_gcd, t, tmp_inv;

	mpz_init(tmp_gcd);
	mpz_init(t);
	mpz_init(tmp_inv);
	from_int_to_rns(pp2, &rns_b, modul_p); // P mod Mb

	mpz_sub(tmp_inv, rns_a.M, modul_p);
	mpz_gcdext(tmp_gcd, inv_p_modM, t, tmp_inv, rns_a.M);
	from_int_to_rns(pp1, &rns_a, inv_p_modM); //(-P)^-1 mod Ma

	mpz_gcdext(tmp_gcd, inv_M_modMp, t, rns_a.M, rns_b.M);
	from_int_to_rns(pp3, &rns_b, inv_M_modMp); // Ma^{-1} mod Mb

	mult.inv_p_modMa = pp1;
	mult.p_modMb = pp2;
	mult.inv_Ma_modMb = pp3;
	mult.conv_atob = &conv_atob;
	
	mult.conv_btoa = &conv_btoa;
	
	

	int64_t *tmp[4]; // RNS modular multiplication intermediate results
	// One more for the base convertion
	tmp[0] = (int64_t *)malloc(NB_COEFF * sizeof(int64_t));
	tmp[1] = (int64_t *)malloc(NB_COEFF * sizeof(int64_t));
	tmp[2] = (int64_t *)malloc(NB_COEFF * sizeof(int64_t));
	tmp[3] = (int64_t *)malloc(NB_COEFF * sizeof(int64_t));

	mpz_urandomm(A, state, modul_p); //Randomly generates A < P
	mpz_urandomm(B, state, modul_p); //Randomly generates A < P
	from_int_to_rns(pa, &rns_a, A);
	from_int_to_rns(pb, &rns_a, B);
	from_int_to_rns(pab, &rns_b, A);
	from_int_to_rns(pbb, &rns_b, B);

	// Heating caches
	printf("\n\tHeating caches... ");
	for (int i = 0; i < NTEST; i++)
	{

		mult_mod_rns_cr(pca, pcb, pa, pab, pb, pbb, &mult, tmp);
	}
	printf("Done.\n");

	// Testing
	printf("\tTesting... ");

	for (int i = 0; i < NSAMPLES; i++)
	{
		mpz_urandomm(A, state, modul_p); //Randomly generates A < M
		mpz_urandomm(B, state, modul_p); //Randomly generates B < M
		from_int_to_rns(pa, &rns_a, A);
		from_int_to_rns(pb, &rns_a, B);
		from_int_to_rns(pab, &rns_b, A);
		from_int_to_rns(pbb, &rns_b, B);

		for (int j = 0; j < NTEST; j++)
		{

			// RDTSC
			t1 = cpucyclesStart();

			mult_mod_rns_cr(pca, pcb, pa, pab, pb, pbb, &mult, tmp);

			t2 = cpucyclesStop();

			if (timing > (t2 - t1) / NFUNS)
				timing = (t2 - t1) / NFUNS;

			// Instructions
			before_instructions = rdpmc_instructions();

			mult_mod_rns_cr(pca, pcb, pa, pab, pb, pbb, &mult, tmp);

			after_instructions = rdpmc_instructions();

			if (instructions > (after_instructions - before_instructions) / NFUNS)
				instructions = (after_instructions - before_instructions) / NFUNS;

			// actual cycles
			before_cycles = rdpmc_actual_cycles();

			mult_mod_rns_cr(pca, pcb, pa, pab, pb, pbb, &mult, tmp);

			after_cycles = rdpmc_actual_cycles();

			if (cycles > (after_cycles - before_cycles) / NFUNS)
				cycles = (after_cycles - before_cycles) / NFUNS;

			// reference cycles
			before_ref = rdpmc_reference_cycles();

			mult_mod_rns_cr(pca, pcb, pa, pab, pb, pbb, &mult, tmp);

			after_ref = rdpmc_reference_cycles();

			if (ref > (after_ref - before_ref) / NFUNS)
				ref = (after_ref - before_ref) / NFUNS;
		}
		fprintf(fpt, "\t\t\t{\n");
		fprintf(fpt, "\t\t\t\t\"TSC Cycles\" : %lld,\n\t\t\t\t\"Instructions\" : %ld,\n\t\t\t\t\"Actual cycles\" : %ld,\n\t\t\t\t\"Reference cycles\" : %ld\n\t\t\t}", timing, instructions, cycles, ref);
		if (i < NSAMPLES - 1)
			fprintf(fpt, ",");
		fprintf(fpt, "\n");
	}

	fprintf(fpt, "\t\t],\n");

	printf("Done.\n");
	printf("\tRNS sequential modular multiplication : %lld CPU cycles.\n", timing);
	printf("\tRNS sequential modular multiplication : %ld instructions.\n", instructions);
	printf("\tRNS sequential modular multiplication : %ld actual CPU cycles.\n", cycles);
	printf("\tRNS sequential modular multiplication : %ld reference CPU cycles.\n", ref);





	timing = ULLONG_MAX;
	cycles = ULONG_MAX;
	instructions = ULONG_MAX;
	ref = ULONG_MAX;

	int64_t a[NB_COEFF];
	__m512i avx512_pa[NB_COEFF / 8];
	__m512i avx512_pb[NB_COEFF / 8];
	__m512i avx512_pab[NB_COEFF / 8];
	__m512i avx512_pbb[NB_COEFF / 8];

	__m512i avx512_pp1[NB_COEFF / 8];
	__m512i avx512_pp2[NB_COEFF / 8];
	__m512i avx512_pp3[NB_COEFF / 8];

	from_int64_t_to_m512i_rns(avx512_pp1, &rns_a, pp1);
	from_int64_t_to_m512i_rns(avx512_pp2, &rns_b, pp2);
	from_int64_t_to_m512i_rns(avx512_pp3, &rns_b, pp3);

	avx512_init_mrs(&conv_btoa);
	avx512_initialize_inverses_base_conversion(&conv_btoa);
	mult.avx512_inv_p_modMa = avx512_pp1;
	mult.avx512_p_modMb = avx512_pp2;
	mult.avx512_inv_Ma_modMb = avx512_pp3;

	__m512i tmp512_0[NB_COEFF / 8];
	__m512i tmp512_1[NB_COEFF / 8];
	__m512i tmp512_2[NB_COEFF / 8];
	// Using an array is less efficient

	mpz_urandomm(A, state, modul_p); //Randomly generates A < P
	mpz_urandomm(B, state, modul_p); //Randomly generates A < P
	from_int_to_rns(pa, &rns_a, A);
	from_int_to_rns(pb, &rns_a, B);
	from_int_to_rns(pab, &rns_b, A);
	from_int_to_rns(pbb, &rns_b, B);

	from_int64_t_to_m512i_rns(avx512_pa, &rns_a, pa);
	from_int64_t_to_m512i_rns(avx512_pb, &rns_a, pb);
	from_int64_t_to_m512i_rns(avx512_pab, &rns_b, pab);
	from_int64_t_to_m512i_rns(avx512_pbb, &rns_b, pbb);

	// Heating caches
	printf("\n\tHeating caches... ");
	for (int i = 0; i < NTEST; i++)
	{
		//printf("i = %d\n",i);
		avx512_mult_mod_rns_cr(avx512_resa, avx512_resb, avx512_pa, avx512_pab, avx512_pb, avx512_pbb, &mult, tmp512_0, tmp512_1, tmp512_2, a);
	}
	printf("Done.\n");

	// Testing
	printf("\tTesting... ");

	for (int i = 0; i < NSAMPLES; i++)
	{
		mpz_urandomm(A, state, modul_p); //Randomly generates A < P
		mpz_urandomm(B, state, modul_p); //Randomly generates A < P

		from_int_to_rns(pa, &rns_a, A);
		from_int_to_rns(pb, &rns_a, B);
		from_int_to_rns(pab, &rns_b, A);
		from_int_to_rns(pbb, &rns_b, B);

		from_int64_t_to_m512i_rns(avx512_pa, &rns_a, pa);
		from_int64_t_to_m512i_rns(avx512_pb, &rns_a, pb);
		from_int64_t_to_m512i_rns(avx512_pab, &rns_b, pab);
		from_int64_t_to_m512i_rns(avx512_pbb, &rns_b, pbb);

		for (int j = 0; j < NTEST; j++)
		{

			// RDTSC
			t1 = cpucyclesStart();

			avx512_mult_mod_rns_cr(avx512_resa, avx512_resb, avx512_pa, avx512_pab, avx512_pb, avx512_pbb, &mult, tmp512_0, tmp512_1, tmp512_2, a);

			t2 = cpucyclesStop();

			if (timing > (t2 - t1) / NFUNS)
				timing = (t2 - t1) / NFUNS;

			// Instructions
			before_instructions = rdpmc_instructions();

			avx512_mult_mod_rns_cr(avx512_resa, avx512_resb, avx512_pa, avx512_pab, avx512_pb, avx512_pbb, &mult, tmp512_0, tmp512_1, tmp512_2, a);

			after_instructions = rdpmc_instructions();

			if (instructions > (after_instructions - before_instructions) / NFUNS)
				instructions = (after_instructions - before_instructions) / NFUNS;

			// actual cycles
			before_cycles = rdpmc_actual_cycles();

			avx512_mult_mod_rns_cr(avx512_resa, avx512_resb, avx512_pa, avx512_pab, avx512_pb, avx512_pbb, &mult, tmp512_0, tmp512_1, tmp512_2, a);

			after_cycles = rdpmc_actual_cycles();

			if (cycles > (after_cycles - before_cycles) / NFUNS)
				cycles = (after_cycles - before_cycles) / NFUNS;

			// reference cycles
			before_ref = rdpmc_reference_cycles();

			avx512_mult_mod_rns_cr(avx512_resa, avx512_resb, avx512_pa, avx512_pab, avx512_pb, avx512_pbb, &mult, tmp512_0, tmp512_1, tmp512_2, a);

			after_ref = rdpmc_reference_cycles();

			if (ref > (after_ref - before_ref) / NFUNS)
				ref = (after_ref - before_ref) / NFUNS;
		}
		fprintf(fpt, "\t\t\t{\n");
		fprintf(fpt, "\t\t\t\t\"TSC Cycles\" : %lld,\n\t\t\t\t\"Instructions\" : %ld,\n\t\t\t\t\"Actual cycles\" : %ld,\n\t\t\t\t\"Reference cycles\" : %ld\n\t\t\t}", timing, instructions, cycles, ref);
		if (i < NSAMPLES - 1)
			fprintf(fpt, ",");
		fprintf(fpt, "\n");
	}

	fprintf(fpt, "\t\t]\n\t},\n");

	printf("Done.\n");
	printf("\tRNS parallel 512 modular multiplication : %lld CPU cycles.\n", timing);
	printf("\tRNS parallel 512 modular multiplication : %ld instructions.\n", instructions);
	printf("\tRNS parallel 512 modular multiplication : %ld actual CPU cycles.\n", cycles);
	printf("\tRNS parallel 512 modular multiplication : %ld reference CPU cycles.\n", ref);

	timing = ULLONG_MAX;
	cycles = ULONG_MAX;
	instructions = ULONG_MAX;
	ref = ULONG_MAX;























	//free(a);
	//goto fin;
//suite:


	// Cox conversion
	printf("\n\n6. Cox base conversion\n");

	fprintf(fpt, "\"cox_base_conv\" :\n\t{\n");
	fprintf(fpt, "\t\"sequential\" :\n\t\t[\n");

	printf("\n\tHeating caches... ");
	mpz_urandomm(A, state, M); // Randomly generates A < M
	mpz_urandomm(B, state, M); // Randomly generated B < M
	from_int_to_rns(op1, &rns_a, A);
	from_int_to_rns(op2, &rns_a, B);

	for (int i = 0; i < NTEST; i++)
	{
		base_conversion_cox(op2, &conv_atob, op1, 0, 0, 0);
	}
	printf("Done.\n");

	printf("\tTesting... ");

	for (int i = 0; i < NSAMPLES; i++)
	{
		mpz_urandomm(A, state, M); //Randomly generates A < M
		mpz_urandomm(B, state, M); //Randomly generates B < M
		from_int_to_rns(op1, &rns_a, A);
		for (int j = 0; j < NTEST; j++)
		{

			// RDTSC
			t1 = cpucyclesStart();

			base_conversion_cox(op2, &conv_atob, op1, 0, 0, 0);

			t2 = cpucyclesStop();

			if (timing > (t2 - t1) / NFUNS)
				timing = (t2 - t1) / NFUNS;

			// Instructions
			before_instructions = rdpmc_instructions();

			base_conversion_cox(op2, &conv_atob, op1, 0, 0, 0);

			after_instructions = rdpmc_instructions();

			if (instructions > (after_instructions - before_instructions) / NFUNS)
				instructions = (after_instructions - before_instructions) / NFUNS;

			// actual cycles
			before_cycles = rdpmc_actual_cycles();

			base_conversion_cox(op2, &conv_atob, op1, 0, 0, 0);

			after_cycles = rdpmc_actual_cycles();

			if (cycles > (after_cycles - before_cycles) / NFUNS)
				cycles = (after_cycles - before_cycles) / NFUNS;

			// reference cycles
			before_ref = rdpmc_reference_cycles();

			base_conversion_cox(op2, &conv_atob, op1, 0, 0, 0);

			after_ref = rdpmc_reference_cycles();

			if (ref > (after_ref - before_ref) / NFUNS)
				ref = (after_ref - before_ref) / NFUNS;
		}

		fprintf(fpt, "\t\t\t{\n");
		fprintf(fpt, "\t\t\t\t\"TSC Cycles\" : %lld,\n\t\t\t\t\"Instructions\" : %ld,\n\t\t\t\t\"Actual cycles\" : %ld,\n\t\t\t\t\"Reference cycles\" : %ld\n\t\t\t}", timing, instructions, cycles, ref);
		if (i < NSAMPLES - 1)
			fprintf(fpt, ",");
		fprintf(fpt, "\n");
	}

	fprintf(fpt, "\t\t],\n");

	printf("Done.\n");
	printf("\tRNS sequential cox base conversion : %lld CPU cycles.\n", timing);
	printf("\tRNS sequential cox base conversion : %ld instructions.\n", instructions);
	printf("\tRNS sequential cox base conversion : %ld actual CPU cycles.\n", cycles);
	printf("\tRNS sequential cox base conversion : %ld reference CPU cycles.\n", ref);

	
	timing = ULLONG_MAX;
	cycles = ULONG_MAX;
	instructions = ULONG_MAX;
	ref = ULONG_MAX;

	mpz_urandomm(A, state, M); //Randomly generates A < M


	from_int_to_rns(op1, &rns_a, A);
	from_int64_t_to_m512i_rns(avx512_op1, &rns_a, op1);

	for (int i = 0; i < NTEST; i++)
	{
		avx512_base_conversion_cox(avx512_op2, &conv_atob, avx512_op1, op1);

		
	}

//	printf("Done.\n");

	printf("\tTesting... ");

	// timing
	for (int i = 0; i < NSAMPLES; i++)
	{
		mpz_urandomm(A, state, M); //Randomly generates A < M
		from_int_to_rns(op1, &rns_a, A);
		from_int64_t_to_m512i_rns(avx512_op1, &rns_a, op1);
		for (int j = 0; j < NTEST; j++)
		{
			// RDTSC
			t1 = cpucyclesStart();

			avx512_base_conversion_cox(avx512_op2, &conv_atob, avx512_op1, op1);

			t2 = cpucyclesStop();

			if (timing > (t2 - t1) / NFUNS)
				timing = (t2 - t1) / NFUNS;

			// Instructions
			before_instructions = rdpmc_instructions();

			avx512_base_conversion_cox(avx512_op2, &conv_atob, avx512_op1, op1);

			after_instructions = rdpmc_instructions();

			if (instructions > (after_instructions - before_instructions) / NFUNS)
				instructions = (after_instructions - before_instructions) / NFUNS;

			// actual cycles
			before_cycles = rdpmc_actual_cycles();

			avx512_base_conversion_cox(avx512_op2, &conv_atob, avx512_op1, op1);

			after_cycles = rdpmc_actual_cycles();

			if (cycles > (after_cycles - before_cycles) / NFUNS)
				cycles = (after_cycles - before_cycles) / NFUNS;

			// reference cycles
			before_ref = rdpmc_reference_cycles();

			avx512_base_conversion_cox(avx512_op2, &conv_atob, avx512_op1, op1);

			after_ref = rdpmc_reference_cycles();

			if (ref > (after_ref - before_ref) / NFUNS)
				ref = (after_ref - before_ref) / NFUNS;
		}

		fprintf(fpt, "\t\t\t{\n");
		fprintf(fpt, "\t\t\t\t\"TSC Cycles\" : %lld,\n\t\t\t\t\"Instructions\" : %ld,\n\t\t\t\t\"Actual cycles\" : %ld,\n\t\t\t\t\"Reference cycles\" : %ld\n\t\t\t}", timing, instructions, cycles, ref);
		if (i < NSAMPLES - 1)
			fprintf(fpt, ",");
		fprintf(fpt, "\n");
	}

	fprintf(fpt, "\t\t]\n\t},\n");

	printf("Done.\n");
	printf("\tRNS vectorized 512 cox base conversion : %lld CPU cycles.\n", timing);
	printf("\tRNS vectorized 512 cox base conversion : %ld instructions.\n", instructions);
	printf("\tRNS vectorized 512 cox base conversion : %ld actual CPU cycles.\n", cycles);
	printf("\tRNS vectorized 512 cox base conversion : %ld reference CPU cycles.\n", ref);

fprintf(fr, "Base conversion; AVX-512; %d; %d; %ld; %lld; %ld; %ld\n", NB_COEFF, TAILLE_MODULE, instructions, timing, cycles, ref);



	timing = ULLONG_MAX;
	cycles = ULONG_MAX;
	instructions = ULONG_MAX;
	ref = ULONG_MAX;

	// Cox modular multiplication
	printf("\n\n7. Cox modular multiplication (first base extension Bajard-Imbert, second Kawamura):\n");

	fprintf(fpt, "\"cox_mod_mul\" :\n\t{\n");
	fprintf(fpt, "\t\"sequential\" :\n\t\t[\n");

	mpz_urandomm(A, state, modul_p); //Randomly generates A < P
	mpz_urandomm(B, state, modul_p); //Randomly generates A < P
	from_int_to_rns(pa, &rns_a, A);
	from_int_to_rns(pb, &rns_a, B);
	from_int_to_rns(pab, &rns_b, A);
	from_int_to_rns(pbb, &rns_b, B);

	// Heating caches
	printf("\n\tHeating caches... ");

	mpz_urandomm(A, state, M);
	from_int_to_rns(op1, &rns_a, A);
	for (int i = 0; i < NTEST; i++)
	{

		mult_mod_rns_cr_cox(resa, resb, pa, pab, pb, pbb, &mult, tmp);
	}
	printf("Done.\n");

	// Testing
	printf("\tTesting... ");

	for (int i = 0; i < NSAMPLES; i++)
	{

		mpz_urandomm(A, state, modul_p); //Randomly generates A < P
		mpz_urandomm(B, state, modul_p); //Randomly generates A < P
		from_int_to_rns(pa, &rns_a, A);
		from_int_to_rns(pb, &rns_a, B);
		from_int_to_rns(pab, &rns_b, A);
		from_int_to_rns(pbb, &rns_b, B);

		for (int j = 0; j < NTEST; j++)
		{

			// RDTSC
			t1 = cpucyclesStart();

			mult_mod_rns_cr_cox(resa, resb, pa, pab, pb, pbb, &mult, tmp);

			t2 = cpucyclesStop();

			if (timing > (t2 - t1) / NFUNS)
				timing = (t2 - t1) / NFUNS;

			// Instructions
			before_instructions = rdpmc_instructions();

			mult_mod_rns_cr_cox(resa, resb, pa, pab, pb, pbb, &mult, tmp);

			after_instructions = rdpmc_instructions();

			if (instructions > (after_instructions - before_instructions) / NFUNS)
				instructions = (after_instructions - before_instructions) / NFUNS;

			// actual cycles
			before_cycles = rdpmc_actual_cycles();

			mult_mod_rns_cr_cox(resa, resb, pa, pab, pb, pbb, &mult, tmp);

			after_cycles = rdpmc_actual_cycles();

			if (cycles > (after_cycles - before_cycles) / NFUNS)
				cycles = (after_cycles - before_cycles) / NFUNS;

			// reference cycles
			before_ref = rdpmc_reference_cycles();

			mult_mod_rns_cr_cox(resa, resb, pa, pab, pb, pbb, &mult, tmp);

			after_ref = rdpmc_reference_cycles();

			if (ref > (after_ref - before_ref) / NFUNS)
				ref = (after_ref - before_ref) / NFUNS;
		}
		fprintf(fpt, "\t\t\t{\n");
		fprintf(fpt, "\t\t\t\t\"TSC Cycles\" : %lld,\n\t\t\t\t\"Instructions\" : %ld,\n\t\t\t\t\"Actual cycles\" : %ld,\n\t\t\t\t\"Reference cycles\" : %ld\n\t\t\t}", timing, instructions, cycles, ref);
		if (i < NSAMPLES - 1)
			fprintf(fpt, ",");
		fprintf(fpt, "\n");
	}

	fprintf(fpt, "\t\t],\n");

















	printf("Done.\n");
	printf("\tRNS sequential cox modular multiplication : %lld CPU cycles.\n", timing);
	printf("\tRNS sequential cox modular multiplication : %ld instructions.\n", instructions);
	printf("\tRNS sequential cox modular multiplication : %ld actual CPU cycles.\n", cycles);
	printf("\tRNS sequential cox modular multiplication : %ld reference CPU cycles.\n", ref);



	timing = ULLONG_MAX;
	cycles = ULONG_MAX;
	instructions = ULONG_MAX;
	ref = ULONG_MAX;

	from_int64_t_to_m512i_rns(avx512_pp1, &rns_a, pp1);
	from_int64_t_to_m512i_rns(avx512_pp2, &rns_b, pp2);
	from_int64_t_to_m512i_rns(avx512_pp3, &rns_b, pp3);

	mult.avx512_inv_p_modMa = avx512_pp1;
	mult.avx512_p_modMb = avx512_pp2;
	mult.avx512_inv_Ma_modMb = avx512_pp3;

	// Using an array is less efficient

	mpz_urandomm(A, state, modul_p); //Randomly generates A < P
	mpz_urandomm(B, state, modul_p); //Randomly generates A < P
	from_int_to_rns(pa, &rns_a, A);
	from_int_to_rns(pb, &rns_a, B);
	from_int_to_rns(pab, &rns_b, A);
	from_int_to_rns(pbb, &rns_b, B);

	from_int64_t_to_m512i_rns(avx512_pa, &rns_a, pa);
	from_int64_t_to_m512i_rns(avx512_pb, &rns_a, pb);
	from_int64_t_to_m512i_rns(avx512_pab, &rns_b, pab);
	from_int64_t_to_m512i_rns(avx512_pbb, &rns_b, pbb);

	// Heating caches
	printf("\n\tHeating caches... ");
	for (int i = 0; i < NTEST; i++)
	{
		//printf("i = %d\n",i);
		avx512_mult_mod_rns_cr_cox(avx512_resa, avx512_resb, avx512_pa, avx512_pab, avx512_pb, avx512_pbb, &mult, tmp512_0, tmp512_1, tmp512_2, a);
	}
	printf("Done.\n");

	// Testing
	printf("\tTesting... ");

	for (int i = 0; i < NSAMPLES; i++)
	{
		mpz_urandomm(A, state, modul_p); //Randomly generates A < P
		mpz_urandomm(B, state, modul_p); //Randomly generates A < P

		from_int_to_rns(pa, &rns_a, A);
		from_int_to_rns(pb, &rns_a, B);
		from_int_to_rns(pab, &rns_b, A);
		from_int_to_rns(pbb, &rns_b, B);

		from_int64_t_to_m512i_rns(avx512_pa, &rns_a, pa);
		from_int64_t_to_m512i_rns(avx512_pb, &rns_a, pb);
		from_int64_t_to_m512i_rns(avx512_pab, &rns_b, pab);
		from_int64_t_to_m512i_rns(avx512_pbb, &rns_b, pbb);

		for (int j = 0; j < NTEST; j++)
		{

			// RDTSC
			t1 = cpucyclesStart();

			avx512_mult_mod_rns_cr_cox(avx512_resa, avx512_resb, avx512_pa, avx512_pab, avx512_pb, avx512_pbb, &mult, tmp512_0, tmp512_1, tmp512_2, a);

			t2 = cpucyclesStop();

			if (timing > (t2 - t1) / NFUNS)
				timing = (t2 - t1) / NFUNS;

			// Instructions
			before_instructions = rdpmc_instructions();

			avx512_mult_mod_rns_cr_cox(avx512_resa, avx512_resb, avx512_pa, avx512_pab, avx512_pb, avx512_pbb, &mult, tmp512_0, tmp512_1, tmp512_2, a);

			after_instructions = rdpmc_instructions();

			if (instructions > (after_instructions - before_instructions) / NFUNS)
				instructions = (after_instructions - before_instructions) / NFUNS;

			// actual cycles
			before_cycles = rdpmc_actual_cycles();

			avx512_mult_mod_rns_cr_cox(avx512_resa, avx512_resb, avx512_pa, avx512_pab, avx512_pb, avx512_pbb, &mult, tmp512_0, tmp512_1, tmp512_2, a);

			after_cycles = rdpmc_actual_cycles();

			if (cycles > (after_cycles - before_cycles) / NFUNS)
				cycles = (after_cycles - before_cycles) / NFUNS;

			// reference cycles
			before_ref = rdpmc_reference_cycles();

			avx512_mult_mod_rns_cr_cox(avx512_resa, avx512_resb, avx512_pa, avx512_pab, avx512_pb, avx512_pbb, &mult, tmp512_0, tmp512_1, tmp512_2, a);

			after_ref = rdpmc_reference_cycles();

			if (ref > (after_ref - before_ref) / NFUNS)
				ref = (after_ref - before_ref) / NFUNS;
		}
		fprintf(fpt, "\t\t\t{\n");
		fprintf(fpt, "\t\t\t\t\"TSC Cycles\" : %lld,\n\t\t\t\t\"Instructions\" : %ld,\n\t\t\t\t\"Actual cycles\" : %ld,\n\t\t\t\t\"Reference cycles\" : %ld\n\t\t\t}", timing, instructions, cycles, ref);
		if (i < NSAMPLES - 1)
			fprintf(fpt, ",");
		fprintf(fpt, "\n");
	}

	fprintf(fpt, "\t\t]\n\t},\n");

	printf("Done.\n");
	printf("\tRNS parallel 512 cox modular multiplication : %lld CPU cycles.\n", timing);
	printf("\tRNS parallel 512 cox modular multiplication : %ld instructions.\n", instructions);
	printf("\tRNS parallel 512 cox modular multiplication : %ld actual CPU cycles.\n", cycles);
	printf("\tRNS parallel 512 cox modular multiplication : %ld reference CPU cycles.\n", ref);

	timing = ULLONG_MAX;
	cycles = ULONG_MAX;
	instructions = ULONG_MAX;
	ref = ULONG_MAX;












fin:

	fclose(fpt);

	fclose(fr);
	
	return 0;
}
