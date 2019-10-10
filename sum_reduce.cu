//
// Created by xiehaoina on 10/9/19.
//

#include "sum_reduce.cuh"



typedef decimal_digit_t dec1;
typedef long long dec2;

#define DIG_PER_DEC1 9
#define DIG_MASK 100000000
#define DIG_BASE 1000000000
#define DIG_MAX (DIG_BASE - 1)


__device__ __const__ dec1 powers10[DIG_PER_DEC1 + 1] = {
        1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000};

__device__ __const__ dec1 frac_max[DIG_PER_DEC1 - 1] = {900000000, 990000000, 999000000,
                                                999900000, 999990000, 999999000,
                                                999999900, 999999990};

#define ROUND_UP(X) (((X) + DIG_PER_DEC1 - 1) / DIG_PER_DEC1)


#define sanity(d) DBUG_ASSERT((d)->len > 0)

#define FIX_INTG_FRAC_ERROR(len, intg1, frac1, error) \
  do {                                                \
    if ( (intg1 + frac1 > (len))) {            \
      if ( (intg1 > (len))) {                  \
        intg1 = (len);                                \
        frac1 = 0;                                    \
        error = E_DEC_OVERFLOW;                       \
      } else {                                        \
        frac1 = (len)-intg1;                          \
        error = E_DEC_TRUNCATED;                      \
      }                                               \
    } else                                            \
      error = E_DEC_OK;                               \
  } while (0)

#define ADD(to, from1, from2, carry) /* assume carry <= 1 */ \
  do {                                                       \
    dec1 a = (from1) + (from2) + (carry);                    \
    DBUG_ASSERT((carry) <= 1);                               \
    if (((carry) = a >= DIG_BASE)) /* no division here! */   \
      a -= DIG_BASE;                                         \
    (to) = a;                                                \
  } while (0)

#define ADD2(to, from1, from2, carry)             \
  do {                                            \
    dec2 a = ((dec2)(from1)) + (from2) + (carry); \
    if (((carry) = a >= DIG_BASE)) a -= DIG_BASE; \
    if ( (a >= DIG_BASE)) {                \
      a -= DIG_BASE;                              \
      carry++;                                    \
    }                                             \
    (to) = (dec1)a;                               \
  } while (0)

#define SUB(to, from1, from2, carry) /* to=from1-from2 */ \
  do {                                                    \
    dec1 a = (from1) - (from2) - (carry);                 \
    if (((carry) = a < 0)) a += DIG_BASE;                 \
    (to) = a;                                             \
  } while (0)

#define SUB2(to, from1, from2, carry) /* to=from1-from2 */ \
  do {                                                     \
    dec1 a = (from1) - (from2) - (carry);                  \
    if (((carry) = a < 0)) a += DIG_BASE;                  \
    if ( (a < 0)) {                                 \
      a += DIG_BASE;                                       \
      carry++;                                             \
    }                                                      \
    (to) = a;                                              \
  } while (0)


__device__ inline void decimal_make_zero_kernel(decimal_t *dec) {
    dec->buf[0] = 0;
    dec->intg = 1;
    dec->frac = 0;
    dec->sign = 0;
}

__device__ void max_decimal_kernel(int precision, int frac, decimal_t *to) {
    int intpart;
    dec1 *buf = to->buf;
    DBUG_ASSERT(precision && precision >= frac);

    to->sign = 0;
    if ((intpart = to->intg = (precision - frac))) {
        int firstdigits = intpart % DIG_PER_DEC1;
        if (firstdigits) *buf++ = powers10[firstdigits] - 1; /* get 9 99 999 ... */
        for (intpart /= DIG_PER_DEC1; intpart; intpart--) *buf++ = DIG_MAX;
    }

    if ((to->frac = frac)) {
        int lastdigits = frac % DIG_PER_DEC1;
        for (frac /= DIG_PER_DEC1; frac; frac--) *buf++ = DIG_MAX;
        if (lastdigits) *buf = frac_max[lastdigits - 1];
    }
}




__device__ void do_add(const decimal_t *from1, const decimal_t *from2,
                  decimal_t *to, int *err) {
    int intg1 = ROUND_UP(from1->intg), intg2 = ROUND_UP(from2->intg),
            frac1 = ROUND_UP(from1->frac), frac2 = ROUND_UP(from2->frac),
            frac0 = MY_MAX(frac1, frac2), intg0 = MY_MAX(intg1, intg2), error;
    dec1 *buf1, *buf2, *buf0, *stop, *stop2, x, carry;

    sanity(to);

    /* is there a need for extra word because of carry ? */
    x = intg1 > intg2
        ? from1->buf[0]
        : intg2 > intg1 ? from2->buf[0] : from1->buf[0] + from2->buf[0];
    if ( (x > DIG_MAX - 1)) /* yes, there is */
    {
        intg0++;
        to->buf[0] = 0; /* safety */
    }

    FIX_INTG_FRAC_ERROR(to->len, intg0, frac0, error);
    if ( (error == E_DEC_OVERFLOW)) {
        max_decimal_kernel(to->len * DIG_PER_DEC1, 0, to);
        *err = error;
        return;
    }

    buf0 = to->buf + intg0 + frac0;

    to->sign = from1->sign;
    to->frac = MY_MAX(from1->frac, from2->frac);
    to->intg = intg0 * DIG_PER_DEC1;
    if ( (error)) {
        set_if_smaller(to->frac, frac0 * DIG_PER_DEC1);
        set_if_smaller(frac1, frac0);
        set_if_smaller(frac2, frac0);
        set_if_smaller(intg1, intg0);
        set_if_smaller(intg2, intg0);
    }

    /* part 1 - max(frac) ... min (frac) */
    if (frac1 > frac2) {
        buf1 = (dec1 *)from1->buf + intg1 + frac1;
        stop = (dec1 *)from1->buf + intg1 + frac2;
        buf2 = (dec1 *)from2->buf + intg2 + frac2;
        stop2 = (dec1 *)from1->buf + (intg1 > intg2 ? intg1 - intg2 : 0);
    } else {
        buf1 = (dec1 *)from2->buf + intg2 + frac2;
        stop = (dec1 *)from2->buf + intg2 + frac1;
        buf2 = (dec1 *)from1->buf + intg1 + frac1;
        stop2 = (dec1 *)from2->buf + (intg2 > intg1 ? intg2 - intg1 : 0);
    }
    while (buf1 > stop) *--buf0 = *--buf1;

    /* part 2 - min(frac) ... min(intg) */
    carry = 0;
    while (buf1 > stop2) {
        ADD(*--buf0, *--buf1, *--buf2, carry);
    }

    /* part 3 - min(intg) ... max(intg) */
    buf1 = intg1 > intg2 ? ((stop = (dec1 *)from1->buf) + intg1 - intg2)
                         : ((stop = (dec1 *)from2->buf) + intg2 - intg1);
    while (buf1 > stop) {
        ADD(*--buf0, *--buf1, 0, carry);
    }

    if ( (carry)) *--buf0 = 1;
    DBUG_ASSERT(buf0 == to->buf || buf0 == to->buf + 1);
}

template <typename T> __device__ void inline swap(T a, T b)
{
    T c(a); a=b; b=c;
}

/* to=from1-from2.
   if to==0, return -1/0/+1 - the result of the comparison */
__device__ void do_sub(const decimal_t *from1, const decimal_t *from2,
                  decimal_t *to, int *err) {
    int intg1 = ROUND_UP(from1->intg), intg2 = ROUND_UP(from2->intg),
            frac1 = ROUND_UP(from1->frac), frac2 = ROUND_UP(from2->frac);
    int frac0 = MY_MAX(frac1, frac2), error;
    dec1 *buf1, *buf2, *buf0, *stop1, *stop2, *start1, *start2, carry = 0;

    /* let carry:=1 if from2 > from1 */
    start1 = buf1 = (dec1 *)from1->buf;
    stop1 = buf1 + intg1;
    start2 = buf2 = (dec1 *)from2->buf;
    stop2 = buf2 + intg2;
    if ( (*buf1 == 0)) {
        while (buf1 < stop1 && *buf1 == 0) buf1++;
        start1 = buf1;
        intg1 = (int)(stop1 - buf1);
    }
    if ( (*buf2 == 0)) {
        while (buf2 < stop2 && *buf2 == 0) buf2++;
        start2 = buf2;
        intg2 = (int)(stop2 - buf2);
    }
    if (intg2 > intg1)
        carry = 1;
    else if (intg2 == intg1) {
        dec1 *end1 = stop1 + (frac1 - 1);
        dec1 *end2 = stop2 + (frac2 - 1);
        while ( ((buf1 <= end1) && (*end1 == 0))) end1--;
        while ( ((buf2 <= end2) && (*end2 == 0))) end2--;
        frac1 = (int)(end1 - stop1) + 1;
        frac2 = (int)(end2 - stop2) + 1;
        while (buf1 <= end1 && buf2 <= end2 && *buf1 == *buf2) buf1++, buf2++;
        if (buf1 <= end1) {
            if (buf2 <= end2)
                carry = *buf2 > *buf1;
            else
                carry = 0;
        } else {
            if (buf2 <= end2)
                carry = 1;
            else /* short-circuit everything: from1 == from2 */
            {
                if (to == 0) { /* decimal_cmp() */
                    *err = 0;
                    return;
                }
                decimal_make_zero_kernel(to);
                *err = E_DEC_OK;
                return;
            }
        }
    }

    if (to == 0) /* decimal_cmp() */
        return ;

    to->sign = from1->sign;

    /* ensure that always from1 > from2 (and intg1 >= intg2) */
    if (carry) {
        swap(from1, from2);
        swap(start1, start2);
        swap(intg1, intg2);
        swap(frac1, frac2);
        to->sign = 1 - to->sign;
    }

    FIX_INTG_FRAC_ERROR(to->len, intg1, frac0, error);
    buf0 = to->buf + intg1 + frac0;

    to->frac = MY_MAX(from1->frac, from2->frac);
    to->intg = intg1 * DIG_PER_DEC1;
    if ( (error)) {
        set_if_smaller(to->frac, frac0 * DIG_PER_DEC1);
        set_if_smaller(frac1, frac0);
        set_if_smaller(frac2, frac0);
        set_if_smaller(intg2, intg1);
    }
    carry = 0;

    /* part 1 - max(frac) ... min (frac) */
    if (frac1 > frac2) {
        buf1 = start1 + intg1 + frac1;
        stop1 = start1 + intg1 + frac2;
        buf2 = start2 + intg2 + frac2;
        while (frac0-- > frac1) *--buf0 = 0;
        while (buf1 > stop1) *--buf0 = *--buf1;
    } else {
        buf1 = start1 + intg1 + frac1;
        buf2 = start2 + intg2 + frac2;
        stop2 = start2 + intg2 + frac1;
        while (frac0-- > frac2) *--buf0 = 0;
        while (buf2 > stop2) {
            SUB(*--buf0, 0, *--buf2, carry);
        }
    }

    /* part 2 - min(frac) ... intg2 */
    while (buf2 > start2) {
        SUB(*--buf0, *--buf1, *--buf2, carry);
    }

    /* part 3 - intg2 ... intg1 */
    while (carry && buf1 > start1) {
        SUB(*--buf0, *--buf1, 0, carry);
    }

    while (buf1 > start1) *--buf0 = *--buf1;

    while (buf0 > to->buf) *--buf0 = 0;
    *err = error;
    return;
}

__device__ void decimal_add_kernel(const decimal_t *from1, const decimal_t *from2, decimal_t *to) {
    int err;
    if ( (from1->sign == from2->sign)) return do_add(from1, from2, to, &err);
    do_sub(from1, from2, to, &err);
}

__global__ void sumReduce(decimal_t *p_decimals, decimal_t *output, unsigned int len) {
    extern __shared__ decimal_t sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x  + threadIdx.x;
    if(i < len){
        memcpy(&sdata[tid], &p_decimals[i], sizeof(decimal_t));
    }else{
       decimal_make_zero_kernel(&sdata[tid]);
    }
    __syncthreads();
    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s > 0; s /= 2) {
        decimal_t tmp;
        tmp.len = DECIMAL_LEN;
        if (tid < s) {
            decimal_add_kernel(&sdata[tid + s], &sdata[tid], &tmp);
            memcpy(&sdata[tid], &tmp, sizeof(decimal_t));
            //sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) {
        memcpy(&output[blockIdx.x], &sdata[0], sizeof(decimal_t));
        //printf("%d shared mem: %d %d %d %d\n",blockIdx.x, sdata[0].buf[0] ,sdata[0].buf[1], sdata[0].buf[2], sdata[0].buf[3]);
    }
}

__global__ void sumReduce2(decimal_t *p_decimals, decimal_t *output,unsigned int len) {
    // each thread loads one element from global to shared mem
    decimal_t tmp;
    decimal_t sum;
    tmp.len = DECIMAL_LEN;
    sum.len = DECIMAL_LEN;
    decimal_make_zero_kernel(&tmp);
    for(int i = 0; i < len; i++){
        decimal_add_kernel(&p_decimals[i], &tmp, &sum);
        memcpy(&tmp, &sum, sizeof(decimal_t));
    }
    memcpy(output, &sum, sizeof(decimal_t));
}

void sumDecimal(decimal_t *p_decimals, decimal_t *output,unsigned int len){
    uint32_t dimBlock = BLOCK_SIZE;
    uint32_t dimGrid = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    decimal_t *p_d_outputs1, *p_d_outputs2;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    gpuErrchk(cudaMalloc((void **)&p_d_outputs1, sizeof(struct decimal_t) * dimGrid));

    for(int i , i <)
    cudaMemcpy(p_d_decimals, p_decimals , sizeof(struct decimal_t) * DECIMAL_ARRAY_LEN , cudaMemcpyHostToDevice);
    sumReduce <<< dimGrid, dimBlock, BLOCK_SIZE * sizeof(decimal_t) >>> (p_d_decimals, p_d_outputs1, len);

    gpuErrchk(cudaMalloc((void **)&p_d_outputs2, sizeof(struct decimal_t) * dimGrid));
    len = dimGrid;
    dimGrid = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    while(len > 1) {
        sumReduce <<< dimGrid, dimBlock, BLOCK_SIZE * sizeof(decimal_t) >>> (p_d_outputs1, p_d_outputs2, len);
        len = dimGrid;
        dimGrid = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;
        sumReduce <<< dimGrid, dimBlock, BLOCK_SIZE * sizeof(decimal_t) >>> (p_d_outputs2, p_d_outputs1, len);
        len = dimGrid;
        dimGrid = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    }
    cudaMemcpy(output, &p_d_outputs2[0] , sizeof(struct decimal_t) , cudaMemcpyDeviceToHost);
    //memcpy(output, &p_d_outputs2[0], sizeof(decimal_t));
    //sumReduce2<<<1, 1>>>(p_decimals, output, len);
}