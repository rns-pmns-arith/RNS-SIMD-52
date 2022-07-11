#include <gmp.h>

#include <stdint.h>

#include <immintrin.h>

#ifndef STRUCTS_DATA

#define STRUCTS_DATA

union int512_t{
    uint64_t i64[8];
    __m128i i128[4];
    __m256i i256[2];
    __m512i i512[1];
 
};
 
typedef union int512_t int512;

#define TAILLE_MODULE 52


//#define NB_COEFF 8		  // number of coeffs for every polynomial



typedef __int128 int128;
typedef unsigned __int128 uint128;

struct rns_base_t
{
	unsigned int size;	 // RNS base size
	int64_t *m;			 // moduli
	int *k;				 // m=2^XX -k for crandal base
	__m256i *avx_k;		 // k vectored
	int64_t *p;			 // modulus p expressed in the RNS base
	int64_t *inv_p;		 // p^{-1} mod M in the RNS base
	mpz_t *Mi;			 // Mi for the CRT conversion
	mpz_t *inv_Mi;		 // Mi^{-1} mod mi for the CRT conversion
	int64_t *int_inv_Mi; // Mi^{-1} mod mi for the Cox conversion
	mpz_t M;			 // M for the CRT conversion
	__m256i *avx_inv_Mi; // inv_Mi vectored
	__m256i *avx_m;		 // m vectored for substraction
	__m512i *avx512_k;		 // k vectored 512
	__m512i *avx512_inv_Mi; // inv_Mi vectored 512
	__m512i *avx512_m;		 // m vectored 512 for substraction
};

struct conv_base_t //Constants for the RNSa -> RNSb conversion
{
	struct rns_base_t *rns_a; // RNSa
	struct rns_base_t *rns_b; // RNSb
	int64_t **inva_to_b;	  // modular inverses of RNSa modulo RNSb
	int64_t **mrsa_to_b;	  // MRS Radices modulo RNSb
	__m256i **avx_mrsa_to_b;  // mrsa_to_b vectored

	int64_t *invM_modPi; // -M^{-1}mod pi for Cox conversion /////////////////////
	int64_t **Mi_modPi;	 // Mi mod pj  for Cox conversion    /////////////////////
	int64_t *Ma_modPi; //Ma mod pj (for Shenoy's conversion)
	__m256i *avx_invM_modPi;
	__m256i **avx_Mi_modPi;

	__m512i **avx512_mrsa_to_b;  // mrsa_to_b vectored 512
	__m512i *avx512_invM_modPi;
	__m512i **avx512_Mi_modPi;

};

struct mod_mul_t // Constants for modular multiplication from RNSa to RNSb
{
	int64_t *inv_p_modMa; //(-P)^-1 mod M2
	__m256i *avx_inv_p_modMa;
	int64_t *p_modMb; // P mod M2
	__m256i *avx_p_modMb;
	int64_t *inv_Ma_modMb; // M1^-1 mod M2
	__m256i *avx_inv_Ma_modMb;
	
	int64_t *inv_Mb_modMa; // M2^-2 mod M1
	__m256i *avx_inv_Mb_modMa;
	
	
	
	__m512i *avx512_inv_p_modMa; //(-P)^-1 mod M2
	__m512i *avx512_p_modMb;
	__m512i *avx512_inv_Ma_modMb;
	
	
	
	struct conv_base_t *conv_atob; //Constants for the RNSa -> RNSb conversion

	struct conv_base_t *conv_btoa; //Constants for the RNSa -> RNSb conversion

};
#endif


