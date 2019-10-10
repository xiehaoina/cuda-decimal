//
// Created by xiehaoina on 10/8/19.
//

#include <iostream>
#include <chrono>
#include <cstring>
#include "decimal.h"

#define CUDA 1


#if CUDA
#include "sum_reduce.cuh"
#endif

#define DECIMAL_ARRAY_LEN 1000*1000*100

int main(){
    using Time = std::chrono::high_resolution_clock;
    using fsec = std::chrono::duration<float>;

    struct decimal_t num1, num2 , sum, sum2, to;
    num1.len = DECIMAL_LEN;
    num2.len = DECIMAL_LEN;
    sum.len = DECIMAL_LEN;
    sum2.len = DECIMAL_LEN;
    to.len = DECIMAL_LEN;

    char str[] = "1234567890123.5512521";
    const char *terminal1 = &str[23];

    struct decimal_t * p_decimals;
#if not CUDA
    p_decimals = (struct decimal_t *) malloc(sizeof(struct decimal_t) * DECIMAL_ARRAY_LEN);
    if(p_decimals == NULL)
        exit(-1);
#else
    gpuErrchk(cudaMallocHost((void **)&p_decimals, sizeof(struct decimal_t) * DECIMAL_ARRAY_LEN));

#endif
    std::cout << "decimal size:" << sizeof(struct decimal_t) << " array size:" << DECIMAL_ARRAY_LEN << std::endl;

    struct decimal_t const_num;
    const_num.len = 8;
    string2decimal(str, &const_num, &terminal1);

    for(int i = 0; i < DECIMAL_ARRAY_LEN ; i++){
        memcpy(&p_decimals[i], &const_num, sizeof(struct decimal_t));
    }

#if CUDA

    struct decimal_t * p_d_sum;

    gpuErrchk(cudaMallocManaged ((void **)&p_d_sum, sizeof(struct decimal_t)));
#endif

    decimal_make_zero(&sum);
    decimal_make_zero(&sum2);
    decimal_make_zero(&to);
    auto start = Time::now();
    for(int i = 0; i < DECIMAL_ARRAY_LEN ; i++) {
        decimal_add(&p_decimals[i], &to, &sum);
        to.frac = sum.frac;
        to.intg = sum.intg;
        for(int j = 0; j < to.len; j++){
            to.buf[j] = sum.buf[j];
        }
    }
    auto end = Time::now();
    fsec cpu_duration = end - start;

    longlong2decimal(DECIMAL_ARRAY_LEN,&num2);


    std::cout << "CPU time: " << cpu_duration.count() << " s" << std::endl;
    decimal_div(&sum, &num2, &num1, 0);


#if CUDA
    start = Time::now();

    auto copy = Time::now();
    sumDecimal(p_decimals, &sum2, DECIMAL_ARRAY_LEN);
    //cudaMemcpy(&sum2, p_d_sum , sizeof(struct decimal_t) , cudaMemcpyDeviceToHost);
    end = Time::now();
    fsec gpu_duration = end - start;
    std::cout << "GPU total time: " << gpu_duration.count() << " s" << std::endl;

    int result = decimal_cmp(&sum, &sum2);
    if(result == 0)
        std::cout<<"GPU'results equal CPU's"<<std::endl;
    else
        std::cout<<"failed to verify"<<std::endl;

    if(p_d_sum)
        cudaFree(p_d_sum);
    if(p_decimals)
        cudaFreeHost(p_decimals);
#else
    if(p_decimals)
        free(p_decimals);
#endif
    return 0;
}