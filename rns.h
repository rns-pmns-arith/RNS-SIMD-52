#include "structs_data.h"

#ifndef RNS_H

#define RNS_H



// RNS arithmetic functions
void add_rns(int64_t *rop, struct rns_base_t *base, int64_t *pa, int64_t *pb);
void sub_rns(int64_t *rop, struct rns_base_t *base, int64_t *pa, int64_t *pb);
void mul_rns(int64_t *rop, struct rns_base_t *base, int64_t *pa, int64_t *pb);

void mult_mod_rns(int64_t *rop_rnsa, int64_t *rop_rnsb, int64_t *pa, int64_t *pab, int64_t *pb,
				  int64_t *pbb, struct mod_mul_t *mult, int64_t *tmp[3]);

unsigned int rns_equal(struct rns_base_t base, int64_t *pa, int64_t *pb);
void init_rns(struct rns_base_t *base);
void from_int_to_rns(int64_t *rop, struct rns_base_t *base, mpz_t op);
void from_rns_to_int_crt(mpz_t rop, struct rns_base_t *base, int64_t *op);
void initialize_inverses_base_conversion(struct conv_base_t *conv_base);
void base_conversion(int64_t *rop, struct conv_base_t *conv_base, int64_t *op);

void base_conversion_cr(int64_t *rop, struct conv_base_t *conv_base, int64_t *op, int64_t *a);
void add_rns_cr(int64_t *rop, struct rns_base_t *base, int64_t *pa, int64_t *pb);
void sub_rns_cr(int64_t *rop, struct rns_base_t *base, int64_t *pa, int64_t *pb);
void mul_rns_cr(int64_t *rop, struct rns_base_t *base, int64_t *pa, int64_t *pb);
void mult_mod_rns_cr(int64_t *rop_rnsa, int64_t *rop_rnsb, int64_t *pa, int64_t *pab, int64_t *pb,
					 int64_t *pbb, struct mod_mul_t *mult, int64_t *tmp[4]);

int64_t add_mod_cr(int64_t a, int64_t b, int k);
int64_t mul_mod_cr(int64_t a, int64_t b, int k);
int64_t mul_mod_cr_t(int64_t a, int64_t b, int k);

int compute_k_cox(int64_t *op, struct rns_base_t *base, int r, int q, int alpha);
void base_conversion_cox(int64_t *rop, struct conv_base_t *conv_base, int64_t *op, int r, int q, int alpha);

void base_conversion_sans_alpha(int64_t *rop, struct conv_base_t *conv_base, int64_t *op);
						 


void mult_mod_rns_cr_cox(int64_t *rop_rnsa, int64_t *rop_rnsb, int64_t *pa, int64_t *pab, int64_t *pb,
						 int64_t *pbb, struct mod_mul_t *mult, int64_t *tmp[4]);




#endif

					 
