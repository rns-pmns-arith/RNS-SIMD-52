#include <stdlib.h>

#include <stdio.h>

#include <gmp.h>

#include <math.h>

#include <time.h>

#include "rns.h"

#include "tests.c"

#include "rnsv_AVX512.h"

#define bool unsigned int

#define true 1

#define false 0


// Shenoy and Kumaresan's auxiliary module m_r
extern const int64_t m_r, mask_m_r;


void gmp_montg_mod_mult_rns(mpz_t C, mpz_t A,mpz_t B, mpz_t inv_p_modM, mpz_t M, mpz_t modul_p){

	mpz_t D;
	mpz_init(D);
    mpz_mul(C, A, B);// A*B
    mpz_mul(D, C, inv_p_modM);
    mpz_mod(D,D,M);// Q = A*B*(-P)^(-1) mod M
    mpz_mul(D, D, modul_p);// D = Q*P
    mpz_add(D,C,D);// D = AB+Q*P
    mpz_divexact(C,D,M);// C = (AB+Q*P)/M
    
    mpz_clear(D);
}


int main(void)
{

    /////////////////////////////
    // INIT
    /////////////////////////////

    // Init Random
    unsigned long int i, seed;
    int counter=0;
;
    gmp_randstate_t r_state;

	srand(time(NULL));
	
	seed = time(NULL);
    gmp_randinit_default(r_state);
    gmp_randseed_ui(r_state, seed);

    // Test variables
    bool test;

    // Variables
    int64_t op1[NB_COEFF];
    int64_t op2[NB_COEFF];
    int64_t res[NB_COEFF];

    mpz_t A, B, C;
    mpz_inits(A, B, C, NULL);

 
	#include "base64x52.h"

    // Base
    struct rns_base_t rns_a;
    rns_a.size = NB_COEFF;

    rns_a.m = m1;

    rns_a.k = k1;

    int64_t tmp_k[NB_COEFF];


    init_rns(&rns_a);

    for (int j = 0; j < NB_COEFF; j++)
    {
        tmp_k[j] = (int64_t)m1[j];
    }


     for (int j = 0; j < NB_COEFF; j++)
    {
        tmp_k[j] = (int64_t)k1[j];
    }

    avx512_init_rns(&rns_a);

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



    for (int j = 0; j < NB_COEFF; j++)
    {
        tmp_k[j] = (int64_t)k2[j];
    }
    
   
     avx512_init_rns(&rns_b);

    for (int j = 0; j < NB_COEFF; j++)
    {
        tmp_k[j] = (int64_t)k2[j];
    }
    __m512i avx512_k2[NB_COEFF / 8];
    from_int64_t_to_m512i_rns(avx512_k2, &rns_b, tmp_k);
    rns_b.avx512_k = avx512_k2;
    
    
    printf("\t\tRNS tests\n\n NB_COEFF = %d\n\n", NB_COEFF);
	
   /////////////////////////////
    // TEST CONVERSION INT -> RNS
    /////////////////////////////

    mpz_urandomm(A, r_state, rns_a.M);
    from_int_to_rns(op1, &rns_a, A);

    mpz_t R;
    mpz_inits(R, NULL);
    long unsigned int r;

    test = true;

    for (i = 0; i < rns_a.size; ++i)
    {
        r = mpz_fdiv_r_ui(R, A, rns_a.m[i]);
    }

    mpz_clear(R);

    printf("Conversion from int to RNS... ");
    if (test)
        printf("OK\n");
    else
        printf("ERROR\n");

    /////////////////////////////
    // TEST CONVERSION RNS -> INT
    /////////////////////////////
    from_rns_to_int_crt(B, &rns_a, op1);

    printf("Conversion from RNS to int... ");
    if (mpz_cmp(A, B) == 0)
        printf("OK\n");
    else
        printf("ERROR\n");

    /////////////////////////////
    // TEST SEQUENTIAL ADDITION
    /////////////////////////////

    mpz_t D;
    mpz_inits(D, NULL);

    mpz_urandomm(A, r_state, rns_a.M);
    mpz_urandomm(B, r_state, rns_a.M);
    from_int_to_rns(op1, &rns_a, A);
    from_int_to_rns(op2, &rns_a, B);

    add_rns_cr(res, &rns_a, op1, op2);

    from_rns_to_int_crt(D, &rns_a, res);

    mpz_add(C, A, B);
    from_int_to_rns(op2, &rns_a, C);

    printf("Int64_t RNS addition... ");
    if (rns_equal(rns_a, res, op2))
        printf("OK\n");
    else
        printf("ERROR\n");

    /////////////////////////////
    // TEST SEQUENTIAL SUBSTRACTION
    /////////////////////////////

    mpz_urandomm(A, r_state, rns_a.M);
    mpz_urandomm(B, r_state, rns_a.M);
    from_int_to_rns(op1, &rns_a, A);
    from_int_to_rns(op2, &rns_a, B);

    sub_rns_cr(res, &rns_a, op1, op2);

    from_rns_to_int_crt(D, &rns_a, res);

    mpz_sub(C, A, B);
    from_int_to_rns(op2, &rns_a, C);

    printf("Int64_t RNS substraction... ");
    if (rns_equal(rns_a, res, op2))
        printf("OK\n");
    else
        printf("ERROR\n");

    /////////////////////////////
    // TEST SEQUENTIAL MULTIPLICATION
    /////////////////////////////

    mpz_urandomm(A, r_state, rns_a.M);
    mpz_urandomm(B, r_state, rns_a.M);
    from_int_to_rns(op1, &rns_a, A);
    from_int_to_rns(op2, &rns_a, B);

    mul_rns_cr(res, &rns_a, op1, op2);

    from_rns_to_int_crt(D, &rns_a, res);
    
    mpz_mul(C, A, B);
    mpz_mod(C,C,rns_a.M);
    from_int_to_rns(op2, &rns_a, C);

    printf("Int64_t RNS multiplication... ");
    if (rns_equal(rns_a, res, op2))
        printf("OK\n");
    else
        printf("ERROR\n");

    /////////////////////////////
    // TEST SEQUENTIAL BASE CONVERSION
    /////////////////////////////

    struct conv_base_t conv_atob;
    conv_atob.rns_a = &rns_a;
    conv_atob.rns_b = &rns_b;
    initialize_inverses_base_conversion(&conv_atob);
	int64_t a[NB_COEFF];

	mpz_set_str(C, "1", 16);
		
	mpz_fdiv_q(C,rns_a.M,C);
	gmp_printf("C = %Zx\n",C);
	
	
	counter = 0;
    for(int i=0;i<1000;)
    {
		mpz_urandomm(A, r_state, C);

		from_int_to_rns(op1, &rns_a, A);

		base_conversion_cr(op2, &conv_atob, op1, a);
		from_rns_to_int_crt(B, &rns_b, op2);

		if (!mpz_cmp(A,B))
		    i++;
		else
			counter++,i++;
		
	}
	printf("Int64_t RNS base conversion cr atob...");
    if(!counter)
        printf("OK\n");
    else
        printf("ERROR, counter =%d\n",counter);

    struct conv_base_t conv_btoa;
    conv_btoa.rns_a = &rns_b;
    conv_btoa.rns_b = &rns_a;
    initialize_inverses_base_conversion(&conv_btoa);

    
 	counter = 0;
    for(int i=0;i<1000;)
    {
		mpz_urandomm(A, r_state, C);

		from_int_to_rns(op1, &rns_b, A);

		base_conversion_cr(op2, &conv_btoa, op1, a);
		from_rns_to_int_crt(B, &rns_a, op2);

		if (!mpz_cmp(A,B))
		    i++;
		else
			counter++,i++;
		
	}
		printf("Int64_t RNS base conversion cr btoa...");
    if(!counter)
        printf("OK\n");
    else
        printf("ERROR, counter =%d\n",counter);
   
    
    /////////////////////////////
    // TEST SEQUENTIAL BASE CONVERSION COX
    /////////////////////////////

	//gmp_printf("A = %Zx\n",A);
	// Borne pour cox-rower : moins 1 bit !
	mpz_set_str(C, "2", 16);
		
	mpz_fdiv_q(C,rns_a.M,C);

	counter = 0;
    for(int i=0;i<1000;)
    {
		mpz_urandomm(A, r_state, C);
		
		from_int_to_rns(op1, &rns_a, A);
		base_conversion_cox(op2, &conv_atob, op1, 0, 0, 0);
		from_rns_to_int_crt(B, &rns_b, op2);


		if (!mpz_cmp(A,B))
		    i++;
		else
			counter++,i++;
		
	}

    printf("Int64_t RNS base conversion cox-rower atob... ");
    if(!counter)
        printf("OK\n");
    else
        printf("ERROR, counter = %d\n",counter);



 	counter = 0;
    for(int i=0;i<1000;)
    {
		mpz_urandomm(A, r_state, C);
		
		from_int_to_rns(op1, &rns_b, A);
		base_conversion_cox(op2, &conv_btoa, op1, 0, 0, 0);
		from_rns_to_int_crt(B, &rns_a, op2);


		if (!mpz_cmp(A,B))
		    i++;
		else
			counter++,i++;
		
	}

    printf("Int64_t RNS base conversion cox-rower btoa... ");
    if(!counter)
        printf("OK\n");
    else
        printf("ERROR, counter = %d\n",counter);

       
        
        
        
        
    //goto fin;
    

    /////////////////////////////
    //
    // 			AVX512
    //
    /////////////////////////////


    /////////////////////////////
    // TEST CONVERSION RNS -> AVX-512
    /////////////////////////////

    __m512i avx512_op1[NB_COEFF / 8];

    mpz_urandomm(A, r_state, rns_a.M);
    from_int_to_rns(op1, &rns_a, A);
    from_int64_t_to_m512i_rns(avx512_op1, &rns_a, op1);

    __m256i high, low;

    test = true;
    
    printf("rns_a.size = %d\n",rns_a.size);
    for (int i = 0; i < rns_a.size / 8; i++)
    {
        low = _mm512_extracti64x4_epi64 (avx512_op1[i], 0);
        high = _mm512_extracti64x4_epi64 (avx512_op1[i], 1);

        test = test && (_mm256_extract_epi64(low, 0) == op1[8 * i]);
        test = test && (_mm256_extract_epi64(low, 1) == op1[8 * i + 1]);
        test = test && (_mm256_extract_epi64(low, 2) == op1[8 * i + 2]);
        test = test && (_mm256_extract_epi64(low, 3) == op1[8 * i + 3]);
        test = test && (_mm256_extract_epi64(high, 0) == op1[8 * i + 4]);
        test = test && (_mm256_extract_epi64(high, 1) == op1[8 * i + 5]);
        test = test && (_mm256_extract_epi64(high, 2) == op1[8 * i + 6]);
        test = test && (_mm256_extract_epi64(high, 3) == op1[8 * i + 7]);
    }

    printf("Conversion from RNS to AVX-512... ");
    if (test)
        printf("OK\n");
    else
        printf("ERROR\n");

    /////////////////////////////
    // TEST CONVERSION AVX-512 -> RNS
    /////////////////////////////

    from_m512i_to_int64_t_rns(op2, &rns_a, avx512_op1);

    printf("Conversion from AVX-512 to RNS... ");
    if (rns_equal(rns_a, op1, op2))
        printf("OK\n");
    else
        printf("ERROR\n");

    /////////////////////////////
    // TEST PARALLEL ADDITION
    /////////////////////////////

    //mpz_set(M, rns_a.M);

    __m512i avx512_op2[NB_COEFF / 8];
    __m512i avx512_res[NB_COEFF / 8];

    from_int_to_rns(op1, &rns_a, A);
    from_int_to_rns(op2, &rns_a, B);
    from_int64_t_to_m512i_rns(avx512_op1, &rns_a, op1);
    from_int64_t_to_m512i_rns(avx512_op2, &rns_a, op2);

    avx512_add_rns_cr(avx512_res, &rns_a, avx512_op1, avx512_op2);
    add_rns_cr(res, &rns_a, op1, op2);

    from_m512i_to_int64_t_rns(op1, &rns_a, avx512_res);

    printf("AVX-512 RNS addition... ");
    if (rns_equal(rns_a, op1, res))
        printf("OK\n");
    else
        printf("ERROR\n");

    /////////////////////////////
    // TEST PARALLEL SUBSTRACTION
    /////////////////////////////

    from_int_to_rns(op1, &rns_a, A);
    from_int_to_rns(op2, &rns_a, B);
    from_int64_t_to_m512i_rns(avx512_op1, &rns_a, op1);
    from_int64_t_to_m512i_rns(avx512_op2, &rns_a, op2);

    avx512_sub_rns_cr(avx512_res, &rns_a, avx512_op1, avx512_op2);
    sub_rns_cr(res, &rns_a, op1, op2);

    from_m512i_to_int64_t_rns(op1, &rns_a, avx512_res);

    printf("AVX-512 RNS substraction... ");
    if (rns_equal(rns_a, op1, res))
        printf("OK\n");
    else
        printf("ERROR\n");

    /////////////////////////////
    // TEST PARALLEL MULTIPLICATION
    /////////////////////////////

    from_int_to_rns(op1, &rns_a, A);
    from_int_to_rns(op2, &rns_a, B);
    from_int64_t_to_m512i_rns(avx512_op1, &rns_a, op1);
    from_int64_t_to_m512i_rns(avx512_op2, &rns_a, op2);

    avx512_mul_rns_cr(avx512_res, &rns_a, avx512_op1, avx512_op2);
    mul_rns_cr(res, &rns_a, op1, op2);

    from_m512i_to_int64_t_rns(op1, &rns_a, avx512_res);

    printf("AVX-512 RNS multiplication... ");
    if (rns_equal(rns_a, op1, res))
        printf("OK\n");
    else
        printf("ERROR\n");
        
    //goto fin;

    /////////////////////////////
    // TEST PARALLEL BASE CONVERSION
    /////////////////////////////

    avx512_init_mrs(&conv_atob);

    avx512_initialize_inverses_base_conversion(&conv_atob);
    mpz_set_str(B, "16174817301453483504153245823054680454784778746814874128407814768681478719", 10);
    
    counter=0;
    
    for(int i=0;i<1000;){
		mpz_urandomm(A, r_state, B);
		
		from_int_to_rns(op1, &rns_a, A);

		from_int64_t_to_m512i_rns(avx512_op1, &rns_a, op1);

		avx512_base_conversion_cox(avx512_op2, &conv_atob, avx512_op1, op1);
		from_m512i_to_int64_t_rns(op1, &rns_b, avx512_op2);
		from_int_to_rns(op2, &rns_b, A);
    if (rns_equal(rns_b, op1, op2))
        i++;
    else
    	counter++,i++;
		
	}

    printf("AVX-512 RNS base conversion cox-rower... ");
    if(!counter)
        printf("OK\n");
    else
        printf("ERROR, counter =%d\n",counter);



    /////////////////////////////
    // TEST PARALLEL MOD MULT COX
    /////////////////////////////

    // Variables
    int64_t pa[NB_COEFF];
    int64_t pab[NB_COEFF];
    int64_t pb[NB_COEFF];
    int64_t pbb[NB_COEFF];
    int64_t ropa[NB_COEFF];
    int64_t ropb[NB_COEFF];
    int64_t pa_mr[1];
    int64_t pb_mr[1];
    int64_t r_hat_mr[1];
    
    
   
     __m512i avx512_pa[NB_COEFF / 8];
     __m512i avx512_pab[NB_COEFF / 8];
     __m512i avx512_pb[NB_COEFF / 8];
     __m512i avx512_pbb[NB_COEFF / 8];
     __m512i avx512_ropa[NB_COEFF / 8];
     __m512i avx512_ropb[NB_COEFF / 8];
     __m512i avx512_tmp0[NB_COEFF / 8];
     __m512i avx512_tmp1[NB_COEFF / 8];
     __m512i avx512_tmp2[NB_COEFF / 8];
    
	mpz_t inv_p_modM, inv_M_modMp, modul_p, R_A, R_B;
	mpz_inits(inv_p_modM, inv_M_modMp, modul_p, R_A, R_B, NULL);

/*********************

	From here, we set modul_p according to NB_COEFF, that is, the size of the RNS base.

**********************/

	
#if NB_COEFF==8

	//module sur 401 bits pour #define NB_COEFF 8
	mpz_set_str (modul_p, "3653232203086218934955973655030869070722999178105346302138244960631343306387218512293583627872011030526507750192262510093", 10);

	//module sur 255 bits pour #define NB_COEFF 8 (p=2^255-19)
	//mpz_set_str (modul_p, "57896044618658097711785492504343953926634992332820282019728792003956564819949", 10);

	//module sur 343 bits pour #define NB_COEFF 8 (p=2^342+15)
	//mpz_set_str (modul_p, "8958978968711216842229769122273777112486581988938598139599956403855167484720643781523509973086428463119", 10);
	
#elif NB_COEFF==16
	//module sur 807 bits pour #define NB_COEFF 16
	mpz_set_str (modul_p, "698033474628833970060004031611394929652758120921727771275292079331285929128542646906281890006813563090278833633717858625064077570302129322570835181515672650950715027384612563562860089046113000766770244160626000792052217924946083727700900872683", 10);
	
#elif NB_COEFF==24
	mpz_set_str (modul_p, "143275048395430897314594349830930633402399313572531637691340678539442066297226363453175716609759323134510237329216305934809700540406078587888662250123492480482205788998360224061645347038954602607873418015702747111158051343443902485727669484868178253536404670068913014031585653119158456215222335204668612015500173779030259794276972468527912604545663694675338750473773", 10);
	
#elif NB_COEFF==32
	mpz_set_str (modul_p, "59233945103636786036853796609535758601827112931834223350778816830959064126361520764475536623611778053255930905105375074955807763851837647254438942266227846735038636552866752197360976490074485327903501052207697544344146105408891289837933863941181196592364271941808687065508585059056886641964774040293692583100838182351794418362326210239972075743360587642668115809656543419762519592692809392820277639030759822175555200399307248856336098085420807919028388167673101908397576628714229473604307", 10);
	
#elif NB_COEFF==40
	mpz_set_str (modul_p, "44086236753949207920637983514861383149257673947602698336003597774852228537442538699311552291999148561661591726029663641576052758441960669906270620352648463891883994695765370500089153458240835568711493351886058343122866700575935327093579369959113562548438102717317460199099348558344901561405189134770477807590752201914379009988207710653195485856435654144295742037841143381344551411260972722769130924053142209824561978011640694162404331703907815691233019452200770844011043454636153705415832075077475365834052368934080670114983097318039911463271365937349501964484896058660471705898799666172211417864834006489489773", 10);

#elif NB_COEFF==48
	mpz_set_str (modul_p, "15926785601034017652597688965642382288067825175930698716358344026572368065986672663012414611984553289419183160005921730973833172922104722418619405421123439799786864450243239730812998583804723116335621502019505976390375029830451440114798215201069277857994638135635659393811802681357471501026460455234964770195044337163284826543047878226247513417879926777450626391710833885101073846548044443214873631989320807409429937976557972213764087705454773110694164608530550543060968501684715090365738317974156965140615237650491169382331346715007267520044707061064695349759840758203818248732191051277899830638189202761019641789630623949450619549067324117844609086999534727777539125675140524476569271034435623443713764525024256006072071247340801939", 10);

#elif NB_COEFF==56
	mpz_set_str (modul_p, "7822126819527056218802607089799723688277078923429881893774431984780683979050944261621918589336396407952502557270927247590973487015040028923590853476103749692938431661010438523369196953471629677631432885259364894860912554815667925599435059109549093446109655405477787683166270363660638570848643117803512279036561922794756949776849740051447341326277035272537590858582927158208347236769078631867973411626657148525743870686053389163912608774915888111233282369715873434752039635716449640739656593831649265944256993481497293342063978844373452286067972893588920526862959538504448909415504347870095870767902454984722647912788177994367632615555380874928105521103488524838463769790894781508916382678310574318484278971645145998208894686678157739959198290600286793663408917427697455276762106872308434241571910556553001625981763454185596072831183063410527103715208043159", 10);

#else
	mpz_set_str (modul_p, "3488943444255984364423638560731936428715107471723434054443952878868966919559379757428364349975781134968509334011980095527219462812461399771310244963248735999426700992090962902533883577229606875094117258741240688354631659733595050827192335631065187160955376529971403733009933931908648669900859281899372924380630271154780195968191696533001063230839809281455678505250494866870781190074054358018048148778014881789500760707801931505502405218013479877169969055294695492106267316438134502313779569945022254877476581257902983722757598039734354320138759717939447050291943920789957487741792306326465351936129660504659379073456789393144147455519373745409507050627214056259245664156620348835904974622867848099392592473209749829084065122386071954467135944925632782097627505111743756462706029263492488732968950814159737211755062471407851531914473431428918003810660327243409693357152864209017449181337883131283254172508954989673076520796121761852136826119438420446446568445641649839317681986361", 10);

#endif	



	
 	int64_t pp1[NB_COEFF];
	int64_t pp2[NB_COEFF+1];// +1 : Shenoy's extra module
	int64_t pp3[NB_COEFF];

    
	// pour changements de bases dans la multiplication modulaire
	struct mod_mul_t mult;
	mpz_t tmp_gcd, t, tmp_inv;

	mpz_init(tmp_gcd);
	mpz_init(t);
	mpz_init(tmp_inv);
	from_int_to_rns(pp2, &rns_b, modul_p); // P mod Mb
	mpz_fdiv_r_ui(t, modul_p, m_r);

	pp2[NB_COEFF] = mpz_get_ui(t);

	mpz_sub(tmp_inv, rns_a.M, modul_p);
	mpz_gcdext(tmp_gcd, inv_p_modM, t, tmp_inv, rns_a.M);
	from_int_to_rns(pp1, &rns_a, inv_p_modM); //(-P)^-1 mod Ma
	
	//vérification
	
	mpz_mul(A,inv_p_modM,tmp_inv);
	mpz_mod(A,A, rns_a.M);
    gmp_printf("Vérification inv_p_modM\nA = 0x%Zx\n",A);   
	
	

	mpz_gcdext(tmp_gcd, inv_M_modMp, t, rns_a.M, rns_b.M);
	from_int_to_rns(pp3, &rns_b, inv_M_modMp); // Ma^{-1} mod Mb
	

	mult.inv_p_modMa = pp1;
	mult.p_modMb = pp2;
	mult.inv_Ma_modMb = pp3;
	mult.conv_atob = &conv_atob;
	
	mult.conv_btoa = &conv_btoa;



    avx512_init_mrs(&conv_btoa);

    avx512_initialize_inverses_base_conversion(&conv_btoa);
   
	__m512i avx512_pp1[NB_COEFF / 8];
	__m512i avx512_pp2[NB_COEFF / 8];
	__m512i avx512_pp3[NB_COEFF / 8];

	from_int64_t_to_m512i_rns(avx512_pp1, &rns_a, pp1);
	from_int64_t_to_m512i_rns(avx512_pp2, &rns_b, pp2);
	from_int64_t_to_m512i_rns(avx512_pp3, &rns_b, pp3);

 	mult.avx512_inv_p_modMa = avx512_pp1;
	mult.avx512_p_modMb = avx512_pp2;
	mult.avx512_inv_Ma_modMb = avx512_pp3;

    // Génération opérandes de la multiplication modulaire
    mpz_urandomm(A, r_state, modul_p);
    mpz_urandomm(B, r_state, modul_p);
    from_int_to_rns(pa, &rns_a, A);
    from_int_to_rns(pb, &rns_a, B);
    from_int64_t_to_m512i_rns(avx512_pa, &rns_a, pa);
    from_int64_t_to_m512i_rns(avx512_pb, &rns_a, pb);
    
    from_int_to_rns(pab, &rns_b, A);
    from_int_to_rns(pbb, &rns_b, B);
    from_int64_t_to_m512i_rns(avx512_pab, &rns_b, pab);
    from_int64_t_to_m512i_rns(avx512_pbb, &rns_b, pbb);
  
    gmp_printf("rns_a.M = 0x%Zx\n",rns_a.M);
    gmp_printf("rns_b.M = 0x%Zx\n",rns_b.M);
    
    	
    gmp_printf("A = 0x%Zx\n",A);   
    gmp_printf("B = 0x%Zx\n",B);//*/


    mpz_mul(C, A, B);// A*B
    mpz_mul(D, C, inv_p_modM);
    mpz_mod(D,D,rns_a.M);// Q = A*B*(-P)^(-1) mod M
    mpz_mul(D, D, modul_p);// D = Q*P
    mpz_add(D,C,D);// D = AB+Q*P
    gmp_printf("Avant division par M\nD   = 0x%Zx\n",D);   
    mpz_divexact(C,D,rns_a.M);// C = (AB+Q*P)/M
    mpz_mul(R_A, C, rns_a.M);
   gmp_printf("vérification mul par M par M\nR_A = 0x%Zx\n",R_A);//*/
    printf("mpz_cmp() %d\n",mpz_cmp(D,R_A));
    
    
    from_int_to_rns(op1, &rns_a, C);
    
    from_int_to_rns(op2, &rns_b, C);
    
    gmp_printf("gmp  : C = 0x%Zx\n",C);   
    mpz_mod(C,C,rns_a.M);
    gmp_printf("gmp  : C = 0x%Zx\n",C);   
    
    
    printf("\nInt64_t mult_mod_rns_cr :\n-------------------------\n");
    
 	int64_t *tmp[4]; // RNS modular multiplication intermediate results
	// One more for the base conversion
	tmp[0] = (int64_t *)malloc(NB_COEFF * sizeof(int64_t));
	tmp[1] = (int64_t *)malloc(NB_COEFF * sizeof(int64_t));
	tmp[2] = (int64_t *)malloc(NB_COEFF * sizeof(int64_t));
	tmp[3] = (int64_t *)malloc(NB_COEFF * sizeof(int64_t));
   
 	mult_mod_rns_cr(ropa, ropb, pa, pab, pb, pbb, &mult, tmp);
  	  	
    from_rns_to_int_crt(R_A, &rns_a, ropa);
    from_rns_to_int_crt(R_B, &rns_b, ropb);
    
    gmp_printf("gmp  : p   = 0x%Zx\n",modul_p);   
    gmp_printf("ropa : R_A = 0x%Zx\n",R_A);   
    gmp_printf("ropb : R_B = 0x%Zx\n",R_B);//*/
    mpz_mod(R_A,R_A,modul_p);
    mpz_mod(R_B,R_B,modul_p);
    gmp_printf("ropa : R_A = 0x%Zx\n",R_A);   
    gmp_printf("ropb : R_B = 0x%Zx\n",R_B);//*/
    printf("mpz_cmp() %d\n",mpz_cmp(R_A,R_B));
    
    printf("sequential RNS mult_mod... ");
    if (rns_equal(rns_a, op1, ropa))
        printf("rns_a : OK\t");
    else
        printf("rns_a : ERROR\t");
    
    if (rns_equal(rns_b, op2, ropb))
        printf("rns_b : OK\n");
    else
        printf("rns_b : ERROR\n");

 	counter = 0;
    for(int i=0;i<1000;)
    {
		mpz_urandomm(R_A, r_state, modul_p);
		mpz_urandomm(R_B, r_state, modul_p);
		from_int_to_rns(pa, &rns_a, R_A);
		from_int_to_rns(pb, &rns_a, R_B);
		
		from_int_to_rns(pab, &rns_b, R_A);
		from_int_to_rns(pbb, &rns_b, R_B);

 		mult_mod_rns_cr(ropa, ropb, pa, pab, pb, pbb, &mult, tmp);
		gmp_montg_mod_mult_rns(D, R_A, R_B, inv_p_modM, rns_a.M, modul_p);
		
		
		from_rns_to_int_crt(R_A, &rns_a, ropa);
		from_rns_to_int_crt(R_B, &rns_b, ropb);

		if ((!mpz_cmp(R_A,R_B))&&(!mpz_cmp(R_A,D)))
		    i++;
		else
			counter++,i++;
		
	}
    if(!counter)
        printf("1000 samples : OK\n");
    else
        printf("ERROR, counter =%d\n",counter);
   
    
    
    printf("\nInt64_t mult_mod_rns_cr_cox (first base conversion Bajard-Imbert, second Kawamura):\n-----------------------------------------------------------------------------------------\n");
    
    from_m512i_to_int64_t_rns(pa, &rns_a, avx512_pa);
    from_m512i_to_int64_t_rns(pab, &rns_b, avx512_pab);
    from_m512i_to_int64_t_rns(pb, &rns_a, avx512_pb);
    from_m512i_to_int64_t_rns(pbb, &rns_b, avx512_pbb);
	from_rns_to_int_crt(R_A, &rns_a, pa);
	from_rns_to_int_crt(R_B, &rns_b, pb);

	mpz_fdiv_r_ui(t, R_A, m_r);
	*pa_mr = mpz_get_ui(t);
	mpz_fdiv_r_ui(t, R_B, m_r);
	*pb_mr = mpz_get_ui(t);
  
	mult_mod_rns_cr_cox(ropa, ropb, pa, pab, pb, pbb, &mult, tmp);
  
  	  	
    from_rns_to_int_crt(R_A, &rns_a, ropa);
    from_rns_to_int_crt(R_B, &rns_b, ropb);
    
    gmp_printf("gmp  : p   = 0x%Zx\n",modul_p);   
    gmp_printf("ropa : R_A = 0x%Zx\n",R_A);   
    gmp_printf("ropb : R_B = 0x%Zx\n",R_B);//*/
    mpz_mod(R_A,R_A,modul_p);
    mpz_mod(R_B,R_B,modul_p);
    gmp_printf("ropa : R_A = 0x%Zx\n",R_A);   
    gmp_printf("ropb : R_B = 0x%Zx\n",R_B);//*/
    printf("mpz_cmp() %d\n",mpz_cmp(R_A,R_B));
    gmp_printf("gmp  : C   = 0x%Zx\n",C);   
    /*mpz_sub(R_A,C,R_A);
    mpz_mod(R_A,R_A,modul_p);
    gmp_printf("gmp  : C-R_A mod p = 0x%Zx\n",R_A);  //*/
    
    printf("sequential RNS mult_mod... ");
    if (!mpz_cmp(R_A,C))
        printf("rns_a : OK\t");
    else
        printf("rns_a : ERROR\t");
    
    if (!mpz_cmp(R_B,C))
        printf("rns_b : OK\n");
    else
        printf("rns_b : ERROR\n");
        
        
    //goto fin;

 	counter = 0;
    for(int i=0;i<1000;)
    {
		mpz_urandomm(R_A, r_state, modul_p);
		mpz_urandomm(R_B, r_state, modul_p);
		from_int_to_rns(pa, &rns_a, R_A);
		from_int_to_rns(pb, &rns_a, R_B);
		
		from_int_to_rns(pab, &rns_b, R_A);
		from_int_to_rns(pbb, &rns_b, R_B);
		mpz_fdiv_r_ui(t, R_A, m_r);
		*pa_mr = mpz_get_ui(t);
		mpz_fdiv_r_ui(t, R_B, m_r);
		*pb_mr = mpz_get_ui(t);
  

		mult_mod_rns_cr_cox(ropa, ropb, pa, pab, pb, pbb, &mult, tmp);
		gmp_montg_mod_mult_rns(D, R_A, R_B, inv_p_modM, rns_a.M, modul_p);
		
		
		from_rns_to_int_crt(R_A, &rns_a, ropa);
		from_rns_to_int_crt(R_B, &rns_b, ropb);
		mpz_mod(R_A,R_A,modul_p);
		mpz_mod(R_B,R_B,modul_p);

		if ((!mpz_cmp(R_A,R_B))&&(!mpz_cmp(R_A,D)))
		    i++;
		else
			counter++,i++;
		
	}
    if(!counter)
        printf("1000 samples : OK\n");
    else
        printf("ERROR, counter =%d\n",counter);

      
        
    printf("\nAVX512 mult_mod_rns_cr_cox (first base conversion Bajard-Imbert, second Kawamura):\n----------------------------------------------------------------------------------\n");
	avx512_mult_mod_rns_cr_cox(avx512_ropa, avx512_ropb, avx512_pa, avx512_pab, avx512_pb, avx512_pbb, &mult, avx512_tmp0, avx512_tmp1, avx512_tmp2, a);//*/
 
    from_m512i_to_int64_t_rns(ropa, &rns_a, avx512_ropa);
    from_m512i_to_int64_t_rns(ropb, &rns_b, avx512_ropb);
    from_rns_to_int_crt(R_A, &rns_a, ropa);
    from_rns_to_int_crt(R_B, &rns_b, ropb);
    
        
    gmp_printf("gmp  : p   = 0x%Zx\n",modul_p);   
    gmp_printf("ropa : R_A = 0x%Zx\n",R_A);   
    gmp_printf("ropb : R_B = 0x%Zx\n",R_B);//*/
    mpz_mod(R_A,R_A,modul_p);
    mpz_mod(R_B,R_B,modul_p);
    gmp_printf("ropa : R_A = 0x%Zx\n",R_A);   
    gmp_printf("ropb : R_B = 0x%Zx\n",R_B);//*/
    printf("mpz_cmp() %d\n",mpz_cmp(R_A,R_B));
    gmp_printf("gmp  : C   = 0x%Zx\n",C);   
    
    printf("sequential RNS mult_mod... ");
    if (!mpz_cmp(R_A,C))
        printf("rns_a : OK\t");
    else
        printf("rns_a : ERROR\t");
    
    if (!mpz_cmp(R_B,C))
        printf("rns_b : OK\n");
    else
        printf("rns_b : ERROR\n");
        
        
    //goto fin;
    
    
 	counter = 0;
    for(int i=0;i<1000;)
    {
		mpz_urandomm(R_A, r_state, modul_p);
		mpz_urandomm(R_B, r_state, modul_p);
		from_int_to_rns(pa, &rns_a, R_A);
		from_int_to_rns(pb, &rns_a, R_B);
		from_int64_t_to_m512i_rns(avx512_pa, &rns_a, pa);
		from_int64_t_to_m512i_rns(avx512_pb, &rns_a, pb);
		
		from_int_to_rns(pab, &rns_b, R_A);
		from_int_to_rns(pbb, &rns_b, R_B);
		from_int64_t_to_m512i_rns(avx512_pab, &rns_b, pab);
		from_int64_t_to_m512i_rns(avx512_pbb, &rns_b, pbb);

		avx512_mult_mod_rns_cr_cox(avx512_ropa, avx512_ropb, avx512_pa, avx512_pab, avx512_pb, avx512_pbb, &mult, avx512_tmp0, avx512_tmp1, avx512_tmp2, a);//*/
		gmp_montg_mod_mult_rns(C, R_A, R_B, inv_p_modM, rns_a.M, modul_p);

		from_m512i_to_int64_t_rns(ropa, &rns_a, avx512_ropa);
		from_m512i_to_int64_t_rns(ropb, &rns_b, avx512_ropb);
		from_rns_to_int_crt(R_A, &rns_a, ropa);
		from_rns_to_int_crt(R_B, &rns_b, ropb);
		mpz_mod(R_A,R_A,modul_p);
		mpz_mod(R_B,R_B,modul_p);

		if ((!mpz_cmp(R_A,R_B))&&(!mpz_cmp(R_A,C)))
		    i++;
		else
			counter++,i++;
		
	}
    if(!counter)
        printf("1000 samples : OK\n");
    else
        printf("ERROR, counter =%d\n",counter);
   

	mpz_clears(inv_p_modM, inv_M_modMp, modul_p, R_A, R_B, tmp_gcd, t, tmp_inv,NULL);
fin:

	free(tmp[0]);
	free(tmp[1]);
	free(tmp[2]);
	free(tmp[3]);
	
	clear_rns(&rns_a);
	clear_rns(&rns_b);
	

	mpz_clears(A,B,C,NULL);
	
    return 0;
}
