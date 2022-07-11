#include "rnsv_AVX512.h"

#include <stdlib.h>

#include <stdio.h>

#include <unistd.h>

#include <stdint.h>

#include <string.h>

#include <time.h>

#include <gmp.h>

#include <immintrin.h>

#include <time.h>

#include "rns.h"


//////////////////////////////////////////////
/// NEW
/////////////////////////////////////////////
inline void from_m512i_to_int64_t_rns(int64_t *rop, struct rns_base_t *base, __m512i *op)
{
	for (int i = 0; i < base->size / 8; i++)
	{
		_mm512_storeu_si512((__m512i *)&rop[8 * i], op[i]);
	}
}

inline void from_int64_t_to_m512i_rns(__m512i *rop, struct rns_base_t *base, int64_t *op)
{
	int j;
	for (j = 0; j < (base->size) / 8; j += 1)
	{
		rop[j] = _mm512_set_epi64(op[8 * j + 7], op[8 * j + 6], op[8 * j + 5], op[8 * j + 4], 
			                      op[8 * j + 3], op[8 * j + 2], op[8 * j + 1], op[8 * j]);
	}
}
/////////////////////////////////////////////


// ----------------------------------------------------------------------------------------------------------
// Prints
// ------


//////////////////////////////////////////////
/// NEW  
/////////////////////////////////////////////

void affiche512i(__m512i x, char* s){

	uint64_t x64[8];
	_mm512_store_si512(x64,x);
	
	printf("%s := ",s);
	for(int i=0;i<8;i++) printf("%16.16lX ",x64[i]);
	printf("\n");
}

void affiche512i_32(__m512i x, char* s){

	uint32_t x32[16];
	_mm512_store_si512((uint64_t*)x32,x);
	
	printf("%s := ",s);
	for(int i=0;i<16;i++) printf("%8.8X ",x32[i]);
	printf("\n");
}

void affiche104(__m512i l, __m512i h,char * s){

	uint64_t l64[8], h64[8];
	_mm512_store_si512(l64,l);
	_mm512_store_si512(h64,h);
		
	printf("%s := ",s);
	for(int i=0;i<8;i++) printf("%16.16lX%13.13lX ",h64[i],l64[i]&0xfffffffffffffUL);
	printf("\n");
}

//////////////////////////////////////////////



//////////////////////////////////////////////
/// NEW  
/////////////////////////////////////////////
inline void avx512_init_rns(struct rns_base_t *base)
{

	int n = base->size;

	__m512i *avx512_inv_Mi = (__m512i *)_mm_malloc(n * sizeof(__m512i) / 8, 64);

	for (int i = 0; i < n / 8; i++)
	{
		avx512_inv_Mi[i] = _mm512_set_epi64(base->int_inv_Mi[8 * i + 7], base->int_inv_Mi[8 * i + 6], base->int_inv_Mi[8 * i + 5], base->int_inv_Mi[8 * i + 4], 
			                                base->int_inv_Mi[8 * i + 3], base->int_inv_Mi[8 * i + 2], base->int_inv_Mi[8 * i + 1], base->int_inv_Mi[8 * i]);
	}

	base->avx512_inv_Mi = avx512_inv_Mi;

	__m512i *tmp = (__m512i *)_mm_malloc(n * sizeof(__m512i) / 8, 64);

	for (int i = 0; i < n / 8; i++)
	{
		tmp[i] = _mm512_set_epi64(base->m[8 * i + 7], 
			base->m[8 * i + 6], 
			base->m[8 * i + 5], 
			base->m[8 * i + 4], 
			                      base->m[8 * i + 3], 
			                      base->m[8 * i + 2], 
			                      base->m[8 * i + 1], 
			                      base->m[8 * i]);
	}

	base->avx512_m = tmp;
}

inline void avx512_init_mrs(struct conv_base_t *conv_base)
{
	int i;
	int size = conv_base->rns_a->size;
	__m512i tmp[NB_COEFF / 8];
	conv_base->avx512_mrsa_to_b = (__m512i **)malloc(size * sizeof(__m512i *));

	for (i = 0; i < size; i++)
	{
		conv_base->avx512_mrsa_to_b[i] = (__m512i *)_mm_malloc(size * sizeof(__m512i) / 8, 64);
	}

	for (i = 0; i < size; i++)
	{
		from_int64_t_to_m512i_rns(conv_base->avx512_mrsa_to_b[i], conv_base->rns_a, conv_base->mrsa_to_b[i]);
	}
}

inline void avx512_initialize_inverses_base_conversion(struct conv_base_t *conv_base)
{

	int size = conv_base->rns_a->size;

	__m512i **tmp_Arr;

	tmp_Arr = (__m512i **)_mm_malloc(size * sizeof(__m512i *), 64);

	for (int i = 0; i < size; i++)
	{
		tmp_Arr[i] = (__m512i *)_mm_malloc(size * sizeof(__m512i) / 8, 64); 
	}

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size / 8; j++)
		{
			tmp_Arr[i][j] = _mm512_set_epi64(conv_base->Mi_modPi[i][8 * j + 7], conv_base->Mi_modPi[i][8 * j + 6], conv_base->Mi_modPi[i][8 * j + 5], conv_base->Mi_modPi[i][8 * j + 4], 
				                             conv_base->Mi_modPi[i][8 * j + 3], conv_base->Mi_modPi[i][8 * j + 2], conv_base->Mi_modPi[i][8 * j + 1], conv_base->Mi_modPi[i][8 * j]);
		}
	}
	conv_base->avx512_Mi_modPi = tmp_Arr;

	conv_base->avx512_invM_modPi = (__m512i *)_mm_malloc(size * sizeof(__m512i) / 8, 64);

	for (int i = 0; i < size / 8; i++)
	{
		conv_base->avx512_invM_modPi[i] = _mm512_set_epi64(conv_base->invM_modPi[8 * i + 7], conv_base->invM_modPi[8 * i + 6], conv_base->invM_modPi[8 * i + 5], conv_base->invM_modPi[8 * i + 4], 
			                                               conv_base->invM_modPi[8 * i + 3], conv_base->invM_modPi[8 * i + 2], conv_base->invM_modPi[8 * i + 1], conv_base->invM_modPi[8 * i]);
	}
}

/////////////////////////////////////////////

// ----------------------------------------------------------------------------------------------------------
// Addition
// --------

/* _m256i addition with Crandall moduli.

BEFORE :
	- a first __m256i operand
	- b second __m256i operand
	- k Crandall moduli

AFTER :
	- rop contains (a + b) mod k

NEEDS :
	- rop allocated

ENSURES :
	- a UNCHANGED
	- b UNCHANGED
	- k UNCHANGED
*/

//////////////////////////////////////////////
/// NEW  
/////////////////////////////////////////////

inline __m512i avx512_add_mod_cr(__m512i a, __m512i b, __m512i k)
{

	__m512i tmp_mask = _mm512_slli_epi64(_mm512_set1_epi64(1), TAILLE_MODULE); //A METTRE EN DUR
	__m512i mask = _mm512_sub_epi64(tmp_mask, _mm512_set1_epi64(1));//A METTRE EN DUR
	__m512i tmp = _mm512_add_epi64(a, b);
	__m512i up = _mm512_srli_epi64(tmp, TAILLE_MODULE); // La retenue sortante
	__m512i lo = _mm512_and_si512(tmp, mask); // La partie basse de la somme
	__m512i tmp_res = _mm512_madd_epi16(up, k);
	__m512i res = _mm512_add_epi64(lo, tmp_res);
	return res;
}

inline void avx512_add_rns_cr(__m512i *rop, struct rns_base_t *base, __m512i *pa, __m512i *pb)
{
	int j;

	for (j = 0; j < (base->size) / 8; j += 1)
	{
		rop[j] = avx512_add_mod_cr(pa[j], pb[j], base->avx512_k[j]);
	}
}
/////////////////////////////////////////////

// ----------------------------------------------------------------------------------------------------------
// Substraction
// ------------

/* _m256i substraction with Crandall moduli.

BEFORE :
	- a first __m256i operand
	- b second __m256i operand
	- k Crandall numbers
	- m Moduli

AFTER :
	- rop contains (a - b) mod k

NEEDS :
	- rop allocated

ENSURES :
	- a UNCHANGED
	- b UNCHANGED
	- k UNCHANGED
	- m UNCHANGED
*/

////////////////////////////////////////
// NEW
////////////////////////////////////////
inline __m512i avx512_sub_mod_cr(__m512i a, __m512i b, __m512i k, __m512i m)
{

	__m512i tmp_mask = _mm512_slli_epi64(_mm512_set1_epi64(1), TAILLE_MODULE);
	__m512i mask = _mm512_sub_epi64(tmp_mask, _mm512_set1_epi64(1));

	__m512i tmp1 = _mm512_add_epi64(a, m);

	__m512i tmp = _mm512_sub_epi64(tmp1, b);

	__m512i up = _mm512_srli_epi64(tmp, TAILLE_MODULE); // La retenue sortante

	__m512i lo = _mm512_and_si512(tmp, mask); // La partie basse de la somme

	__m512i tmp_res = _mm512_madd_epi16(up, k); // mul et add ? Il ne manque pas un terme ?

	__m512i res = _mm512_add_epi64(lo, tmp_res);

	return res;
}

inline void avx512_sub_rns_cr(__m512i *rop, struct rns_base_t *base, __m512i *pa, __m512i *pb)
{
	int j;

	for (j = 0; j < (base->size) / 8; j += 1)
	{

		rop[j] = avx512_sub_mod_cr(pa[j], pb[j], base->avx512_k[j], base->avx512_m[j]);
	}
}
//////////////////////////////////////////


// ----------------------------------------------------------------------------------------------------------
// Multiplication
// --------------



////////////////////////////////////////////////
// NEW
////////////////////////////////////////////////
static __m512i mask1_512 = (__m512i){0x7fffffffffffffffUL, 0x7fffffffffffffffUL, 0x7fffffffffffffffUL, 0x7fffffffffffffffUL, 0x7fffffffffffffffUL, 0x7fffffffffffffffUL, 0x7fffffffffffffffUL, 0x7fffffffffffffffUL};
static __m512i mask2_512 = (__m512i){0x8000000000000000UL, 0x8000000000000000UL, 0x8000000000000000UL, 0x8000000000000000UL, 0x8000000000000000UL, 0x8000000000000000UL, 0x8000000000000000UL, 0x8000000000000000UL}; // INUTILE ?

/*__m512i addition of 2 terms

BEFORE :
	- a first __m512i operand
	- b second __m512i operand

AFTER :
	- rop_up contains the upper part of (a + b)
	- rop_lo contains the lower part of (a + b)

NEEDS :
	- rop_up allocated
	- rop_lo allocated

ENSURES :
	- a UNCHANGED
	- b UNCHANGED
*/
static inline void avx512_add_aux_2e(__m512i *rop_up, __m512i *rop_lo, __m512i a, __m512i b)
{
	__m512i sum = _mm512_add_epi64(a, b);
	*rop_lo = _mm512_and_si512(sum, mask1_512);
	*rop_up = _mm512_srli_epi64(sum, 63);
}

/*__m512i addition of 3 terms

BEFORE :
	- a first __m512i operand
	- b second __m512i operand
	- c third __m512i operand

AFTER :
	- rop_up contains the upper part of (a + b + c)
	- rop_lo contains the lower part of (a + b + c)

NEEDS :
	- rop_up allocated
	- rop_lo allocated

ENSURES :
	- a UNCHANGED
	- b UNCHANGED
	- c UNCHANGED
*/
static inline void avx512_add_aux_3e(__m512i *rop_up, __m512i *rop_lo, __m512i a, __m512i b, __m512i c)
{

	__m512i up, lo, up2, lo2;
	avx512_add_aux_2e(&up, &lo, a, b);
	avx512_add_aux_2e(&up2, rop_lo, lo, c);
	*rop_up = _mm512_add_epi64(up, up2);
}


static __m512i zero_512 = (__m512i){0x0UL,};

/*__m512i multiplication of 2 terms

BEFORE :
	- a first __m512i operand
	- b second __m512i operand

AFTER :
	- rop_up contains the upper part of (a * b)
	- rop_lo contains the lower part of (a * b)

NEEDS :
	- rop_up allocated
	- rop_lo allocated

ENSURES :
	- a UNCHANGED
	- b UNCHANGED
*/
static inline void avx512_mul_aux(__m512i *rop_up, __m512i *rop_lo, __m512i a, __m512i b)
{
												
	*rop_lo = _mm512_madd52lo_epu64(zero_512,a,b);
	
	*rop_up = _mm512_madd52hi_epu64(zero_512,a,b);
	
}

// IDEM mask1_512 et mask2_512

static __m512i mask_512 = (__m512i){0xfffffffffffffUL, 0xfffffffffffffUL, 0xfffffffffffffUL, 0xfffffffffffffUL, 0xfffffffffffffUL, 0xfffffffffffffUL, 0xfffffffffffffUL, 0xfffffffffffffUL};	  


/*__m512i modular multiplication of 2 terms with Crandall moduli

BEFORE :
	- a first __m512i operand
	- b second __m512i operand
	- k Crandall numbers

AFTER :
	- rop contains the upper part of (a * b) mod (n^63-k)

NEEDS :
	- rop

ENSURES :
	- a UNCHANGED
	- b UNCHANGED
	- k UNCHANGED
*/
static inline __m512i avx512_mul_mod_cr(__m512i a, __m512i b, __m512i k)
{
	
	
	__m512i up_times_k, ret1;
	__m512i up, lo, up2, lo2, up3, lo3;

	lo = _mm512_madd52lo_epu64(zero_512,a,b);
	
	up = _mm512_madd52hi_epu64(zero_512,a,b);
	
	up_times_k = _mm512_mullo_epi64(up,k);
	
	__m512i res = _mm512_add_epi64(lo,up_times_k);

	ret1 = _mm512_madd_epi16(_mm512_srli_epi64(res,TAILLE_MODULE),k);
	res = _mm512_add_epi64(_mm512_and_si512(res,mask_512),ret1);
	
	ret1 = _mm512_madd_epi16(_mm512_srli_epi64(res,TAILLE_MODULE),k);
	res = _mm512_add_epi64(_mm512_and_si512(res,mask_512),ret1);
	
	return res;
	
}

inline void avx512_mul_rns_cr(__m512i *rop, struct rns_base_t *base, __m512i *pa, __m512i *pb)
{
	int j;

	for (j = 0; j < (base->size) >> 3; j += 1)
	{
		rop[j] = avx512_mul_mod_cr(pa[j], pb[j], base->avx512_k[j]);
	}
}
//////////////////////////////////////////////////

// ----------------------------------------------------------------------------------------------------------
// Multiplication using Crandall moduli
// -------------------------------------


////////////////////////////////////////////////
// Using Mixed Radix base conversion
////////////////////////////////////////////////
// a[i] en externe avec op[i] dedans
inline void avx512_base_conversion_cr(__m512i *rop, struct conv_base_t *conv_base, __m512i *op, int64_t *a)
{
	int i, j;
	__m512i avx_tmp, avx_tmp2;
	int64_t tmp;
	int size = conv_base->rns_a->size;

//printf("1. %p \n", conv_base->avx512_mrsa_to_b);

    /////////////////////////////////////
    // Calcule les chiffres mixed radix
	for (i = 0; i < size - 1; i++)
	{
		for (j = i + 1; j < size; j++)
		{
			tmp = a[j] - a[i];
			a[j] = mul_mod_cr(tmp, conv_base->inva_to_b[i][j], conv_base->rns_a->k[j]);
		}
	}
    // a[] contient les chiffre mixed radix

//printf("2. %p \n", conv_base->avx512_mrsa_to_b);

    /////////////////////////////////////
    // Calcule les résidus du nombre MRS dans la base de destination
	__m512i a0_512 = _mm512_set1_epi64(a[0]);

//printf("3. %p \n", conv_base->avx512_mrsa_to_b);

	for (j = 0; j < size / 8; j++)
	{
		///!!!!!!!!!!!!!!!!!!!!!!!!!! loadu n'est pas trouvé on a mis load à la place
		__m512i b512 = _mm512_loadu_si512((__m512i *)&conv_base->rns_b->m[j*8]);
		///§!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

		__mmask8 cmp8 = _mm512_cmpgt_epi64_mask(a0_512, b512);
	    rop[j] = _mm512_mask_sub_epi64(a0_512, cmp8, a0_512, b512);    // Computes a[0]%m'[i]
	}

//printf("4. %p \n", conv_base->avx512_mrsa_to_b);

	for (j = 0; j < size / 8; j++)
	{
		for (i = 1; i < size; i++)
		{
			avx_tmp2 = _mm512_set1_epi64(a[i]);

//printf("5. %p  \n", conv_base->avx512_mrsa_to_b);//, conv_base->avx512_mrsa_to_b[0]);

			avx_tmp = avx512_mul_mod_cr(avx_tmp2, conv_base->avx512_mrsa_to_b[i - 1][j], conv_base->rns_b->avx512_k[j]);
			rop[j] = avx512_add_mod_cr(rop[j], avx_tmp, conv_base->rns_b->avx512_k[j]);
		}
	}
}




inline void avx512_mult_mod_rns_cr(__m512i *ropa, __m512i *ropb, __m512i *pa, __m512i *pab, __m512i *pb,
								__m512i *pbb, struct mod_mul_t *mult, __m512i *tmp0, __m512i *tmp1, __m512i *tmp2, int64_t *a)
{

	avx512_mul_rns_cr(tmp0, mult->conv_atob->rns_a, pa, pb);					  //A*B
	avx512_mul_rns_cr(tmp1, mult->conv_atob->rns_b, pab, pbb);					  //A*B in base2
	avx512_mul_rns_cr(tmp2, mult->conv_atob->rns_a, tmp0, mult->avx512_inv_p_modMa); //Q*{P-1}
	from_m512i_to_int64_t_rns(a, mult->conv_atob->rns_a, tmp2);				  //storing tmp2 in a 
	avx512_base_conversion_cr(tmp0, mult->conv_atob, tmp2, a);					  // Q in base2
	avx512_mul_rns_cr(tmp2, mult->conv_atob->rns_b, tmp0, mult->avx512_p_modMb);	  // Q*P base2
	avx512_add_rns_cr(tmp0, mult->conv_atob->rns_b, tmp1, tmp2);				  // A*B + Q*P in base 2
	avx512_mul_rns_cr(ropb, mult->conv_atob->rns_b, tmp0, mult->avx512_inv_Ma_modMb); // Division by Ma
	from_m512i_to_int64_t_rns(a, mult->conv_atob->rns_b, ropb);				  //storing ropb in a 
	avx512_base_conversion_cr(ropa, mult->conv_btoa, ropb, a);					  // ropa in base a

}

////////////////////////////////////////////////

// ----------------------------------------------------------------------------------------------------------
// Multiplication using Cox-Rower method
// -------------------------------------


////////////////////////////////////////////////
// NEW
////////////////////////////////////////////////

// ----------------------------------------------------------------------------------------------------------
// Multiplication using Bajard-Imbert approach
// -------------------------------------------

//Ajuster pour TAILLE_MODULE 52

static __m512i cox512_mask = (__m512i){0xFE00000000000UL, 0xFE00000000000UL, 0xFE00000000000UL, 0xFE00000000000UL, 0xFE00000000000UL, 0xFE00000000000UL, 0xFE00000000000UL, 0xFE00000000000UL};
static __m512i cox512_mask2 = (__m512i){0x80UL, 0x80UL, 0x80UL, 0x80UL, 0x80UL, 0x80UL, 0x80UL, 0x80UL};
static __m512i cox512_sigma = (__m512i){0x40UL, 0x40UL, 0x40UL, 0x40UL, 0x40UL, 0x40UL, 0x40UL, 0x40UL};




inline void avx512_base_conversion_cox(__m512i *rop, struct conv_base_t *conv_base, __m512i *op, int64_t *a)
{
	int i, j;
	int size = conv_base->rns_a->size;

	int r = TAILLE_MODULE;
	int q = 7;

	__m512i sigma = cox512_sigma;//zero_512;//
//printf("2. %p \n", conv_base->avx512_mrsa_to_b);
	__m512i xi512[NB_COEFF>>3];
	int64_t *xi64 = (int64_t *) xi512;

	for (i = 0; i < size / 8; i++)
	{
		rop[i] = _mm512_set1_epi64(0);
		xi512[i] = avx512_mul_mod_cr(conv_base->rns_a->avx512_inv_Mi[i], op[i], conv_base->rns_a->avx512_k[i]);

	}

	__m512i xi, trunk, k_i;

	__m512i tmp0, tmp1, tmp2;

	for (i = 0; i < size; i++)
	{
		xi = _mm512_set1_epi64(xi64[i]);
		
		trunk = xi & cox512_mask; 
		sigma = _mm512_add_epi64(sigma, _mm512_srli_epi64(trunk, r - q));
		k_i = sigma & cox512_mask2;

		sigma = _mm512_sub_epi64(sigma, k_i);
		k_i = _mm512_srli_epi64(k_i, q);

		for (j = 0; j < size / 8; j++)
		{
			tmp0 = avx512_mul_mod_cr(xi, conv_base->avx512_Mi_modPi[i][j], conv_base->rns_b->avx512_k[j]);
			tmp1 = conv_base->avx512_invM_modPi[j] * k_i;
			tmp2 = avx512_add_mod_cr(tmp0, tmp1, conv_base->rns_b->avx512_k[j]);
			rop[j] = avx512_add_mod_cr(rop[j], tmp2, conv_base->rns_b->avx512_k[j]);
		}
	}
}

// Bajard-Imbert first base extension

inline void avx512_base_conversion_sans_alpha(__m512i *rop, struct conv_base_t *conv_base, __m512i *op)
{
	int i, j;
	int size = conv_base->rns_a->size;
	__m512i sigma512[NB_COEFF>>3];
	int64_t *sigma64 = (int64_t *) sigma512;

//printf("2. %p \n", conv_base->avx512_mrsa_to_b);

	for (i = 0; i < size / 8;i++)
	{
		rop[i] = _mm512_set1_epi64(0);
		sigma512[i] = avx512_mul_mod_cr(conv_base->rns_a->avx512_inv_Mi[i], op[i], conv_base->rns_a->avx512_k[i]);
	}
	
	__m512i sigma;

	__m512i tmp0, tmp1, tmp2;

	for (i = 0; i < size; i++)
	{
		sigma = _mm512_set1_epi64(sigma64[i]);
		
		for (j = 0; j < size / 8; j++)
		{
			tmp0 = avx512_mul_mod_cr(sigma, conv_base->avx512_Mi_modPi[i][j], conv_base->rns_b->avx512_k[j]);
			rop[j] = avx512_add_mod_cr(rop[j], tmp0, conv_base->rns_b->avx512_k[j]);
		}
	}
}



inline void avx512_mult_mod_rns_cr_cox(__m512i *ropa, __m512i *ropb, __m512i *pa, __m512i *pab, __m512i *pb,
									__m512i *pbb, struct mod_mul_t *mult, __m512i *tmp0, __m512i *tmp1, __m512i *tmp2, int64_t *a)
{

	int i;

	avx512_mul_rns_cr(tmp0, mult->conv_atob->rns_a, pa, pb);					  //A*B
	avx512_mul_rns_cr(tmp1, mult->conv_atob->rns_b, pab, pbb);					  //A*B in base2
	avx512_mul_rns_cr(tmp2, mult->conv_atob->rns_a, tmp0, mult->avx512_inv_p_modMa); //Q*{P-1}
	avx512_base_conversion_sans_alpha(tmp0, mult->conv_atob, tmp2);
	avx512_mul_rns_cr(tmp2, mult->conv_atob->rns_b, tmp0, mult->avx512_p_modMb);	  // Q*P base2
	avx512_add_rns_cr(tmp0, mult->conv_atob->rns_b, tmp1, tmp2);				  // A*B + Q*P in base 2
	avx512_mul_rns_cr(ropb, mult->conv_atob->rns_b, tmp0, mult->avx512_inv_Ma_modMb); // Division by Ma
	avx512_base_conversion_cox(ropa, mult->conv_btoa, ropb, a);					  // ropa in base a
	
}
