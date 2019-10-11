//
// Created by xiehaoina on 10/8/19.
//

#include <iostream>
#include <chrono>
#include <cstring>
#include "decimal.h"

#define CUDA 0


#if CUDA
#include "sum_reduce.cuh"
#endif

static int decimal_array_len = 1000*1000*100;

#define INIT_DECIMAL(x)  do{x.len = DECIMAL_LEN; decimal_make_zero(&x);} while(0)

using Time = std::chrono::high_resolution_clock;
using fsec = std::chrono::duration<float>;

void calc_average(decimal_t sum){
    decimal_t div, div_result;
    INIT_DECIMAL(div);
    INIT_DECIMAL(div_result);
    longlong2decimal(decimal_array_len,&div);
    decimal_div(&sum, &div, &div_result, 0);
}


void calc_sum_cpu(decimal_t * p_decimals , decimal_t & sum){
    decimal_t to;
    INIT_DECIMAL(to);
    auto start = Time::now();
    for(int i = 0; i < decimal_array_len ; i++) {
        decimal_add(&p_decimals[i], &to, &sum);
        //WAR: decimal_add not support computing pattern  sum += a;
        to.frac = sum.frac;
        to.intg = sum.intg;
        for(int j = 0; j < to.len; j++){
            to.buf[j] = sum.buf[j];
        }
    }
    auto end = Time::now();
    fsec cpu_duration = end - start;
    std::cout << "CPU time: " << cpu_duration.count() << " s" << std::endl;
}

void calc_sum_gpu(decimal_t * p_decimals , decimal_t & sum){
#if CUDA
    auto start = Time::now();
    sum_decimal(p_decimals, &sum, decimal_array_len);
    auto end = Time::now();
    fsec gpu_duration = end - start;
    std::cout << "GPU total time: " << gpu_duration.count() << " s" << std::endl;
#endif
}


decimal_t* init_mem(){
    //allocate mem to store decimal array
    decimal_t * p_decimals;
    decimal_t const_decimal;
    INIT_DECIMAL(const_decimal);

#if not CUDA
    p_decimals = (struct decimal_t *) malloc(sizeof(struct decimal_t) * decimal_array_len);
#else
    gpuErrchk(cudaMallocHost((void **)&p_decimals, sizeof(struct decimal_t) * decimal_array_len));
#endif

    if(p_decimals == NULL)
        return NULL;

    std::cout << "decimal size:" << sizeof(struct decimal_t) << " array size:" << decimal_array_len << std::endl;


    char str[] = "1234567890123.5512521";
    const char *terminal1 = &str[23];
    string2decimal(str, &const_decimal, &terminal1);

    //fill mem with const decimal
    for(int i = 0; i < decimal_array_len ; i++){
        ASSIGN_DECIMAL(p_decimals[i], const_decimal);
        //memcpy(&p_decimals[i], &const_decimal, sizeof(struct decimal_t));
    }
    return p_decimals;
}


void destory_mem(decimal_t * p){
    if(p)

#if CUDA
        cudaFreeHost(p);
#else
        free(p);
#endif
}

int main(int argc, const char * argv[]){


    if(argc == 2){
        decimal_array_len = atoi(argv[1]);
    }

    decimal_t  cpu_sum, gpu_sum;

    INIT_DECIMAL(cpu_sum);

    INIT_DECIMAL(gpu_sum);


    decimal_t * p_decimals = init_mem();

    if(p_decimals == NULL){
        exit(-1);
    }

    calc_sum_cpu(p_decimals, cpu_sum);

    calc_average(cpu_sum);


#if CUDA
    calc_sum_gpu(p_decimals, gpu_sum);

    int result = decimal_cmp(&cpu_sum, &gpu_sum);
    if(result == 0)
        std::cout<<"RESULTS VERIFY SUCCESS !"<<std::endl;
    else
        std::cout<<"RESULTS VERIFY FAILUARE !"<<std::endl;

#endif
    return 0;
}