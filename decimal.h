/* Copyright (c) 2000, 2019, Oracle and/or its affiliates. All rights reserved.
   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License, version 2.0,
   as published by the Free Software Foundation.
   This program is also distributed with certain software (including
   but not limited to OpenSSL) that is licensed under separate terms,
   as designated in a particular file or component or in included license
   documentation.  The authors of MySQL hereby grant you an additional
   permission to link the program and your derivative works with the
   separately licensed software that they have included with MySQL.
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License, version 2.0, for more details.
   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301  USA */

#ifndef DECIMAL_INCLUDED
#define DECIMAL_INCLUDED

#ifndef MYSQL_ABI_CHECK
#include <stdlib.h>
#endif
#include <stdint.h>

typedef enum {
    TRUNCATE = 0,
    HALF_EVEN,
    HALF_UP,
    CEILING,
    FLOOR
} decimal_round_mode;

typedef int32_t decimal_digit_t;
typedef unsigned char uchar;
#define  DBUG_ASSERT(x)  do {}while(0);
#define  DBUG_PRINT(x, y)  do {}while(0);
#define  ALWAYS_INLINE inline
#define DBUG_ENTER(x) do {} while(0);
#define DBUG_RETURN(x) do {} while(0);
#define MY_TEST(x)  ((x) ? 1 : 0)
#define MY_MIN(x, y)   (x > y ? y : x)
#define MY_MAX(x, y)   (x < y ? y : x)
#define set_if_bigger(a, b)   do {if ((a) < (b)) (a) = (b);} while (0)
#define set_if_smaller(a, b)   do {if ((a) > (b)) (a) = (b);} while (0)

#define my_isspace(c) (c == '\t')
#define NOT_FIXED_DEC 31
#define DECIMAL_LEN  4

#define ASSIGN_DECIMAL(a, b)\
 do { \
    a.intg = b.intg; \
    a.frac = b.frac; \
    a.len = b.len; \
    a.sign = b.sign; \
    for (int ijk = 0; ijk < DECIMAL_LEN ; ijk++)\
        a.buf[ijk] = b.buf[ijk]; \
}while(0)


typedef unsigned char decimal_len_t;

/**
    intg is the number of *decimal* digits (NOT number of decimal_digit_t's !)
         before the point
    frac is the number of decimal digits after the point
    len  is the length of buf (length of allocated space) in decimal_digit_t's,
         not in bytes
    sign false means positive, true means negative
    buf  is an array of decimal_digit_t's
 */

struct decimal_t {
    decimal_len_t intg, frac, len;
    bool sign;
    decimal_digit_t buf[DECIMAL_LEN];
};

#ifndef MYSQL_ABI_CHECK
void widen_fraction(int new_frac, decimal_t *d);
int string2decimal(const char *from, decimal_t *to, const char **end);
int decimal2string(const decimal_t *from, char *to, int *to_len,
                   int fixed_precision, int fixed_decimals, char filler);
int decimal2ulonglong(decimal_t *from, unsigned long long *to);
int ulonglong2decimal(unsigned long long from, decimal_t *to);
int decimal2longlong(decimal_t *from, long long *to);
int longlong2decimal(long long from, decimal_t *to);
int decimal2double(const decimal_t *from, double *to);
int double2decimal(double from, decimal_t *to);
int decimal_actual_fraction(const decimal_t *from);
int decimal2bin(decimal_t *from, unsigned char *to, int precision, int scale);
int bin2decimal(const unsigned char *from, decimal_t *to, int precision, int scale,
                bool keep_prec = false);

/**
  Convert decimal to lldiv_t.
  The integer part is stored in to->quot.
  The fractional part is multiplied to 10^9 and stored to to->rem.
  @param  from  Decimal value
  @param  [out] to    lldiv_t value
  @retval 0     on success
  @retval !0    in error
*/
int decimal2lldiv_t(const decimal_t *from, lldiv_t *to);

/**
  Convert doube to lldiv_t.
  The integer part is stored in to->quot.
  The fractional part is multiplied to 10^9 and stored to to->rem.
  @param  nr  Decimal value
  @param  [out] lld    lldiv_t value
  @retval 0     on success
  @retval !0    in error
*/

int double2lldiv_t(double nr, lldiv_t *lld);
int decimal_size(int precision, int scale);
int decimal_bin_size(int precision, int scale);
int decimal_result_size(decimal_t *from1, decimal_t *from2, char op, int param);

int decimal_intg(const decimal_t *from);
int decimal_add(const decimal_t *from1, const decimal_t *from2, decimal_t *to);
int decimal_sub(const decimal_t *from1, const decimal_t *from2, decimal_t *to);
int decimal_cmp(const decimal_t *from1, const decimal_t *from2);
int decimal_mul(const decimal_t *from1, const decimal_t *from2, decimal_t *to);
int decimal_div(const decimal_t *from1, const decimal_t *from2, decimal_t *to,
                int scale_incr);
int decimal_mod(const decimal_t *from1, const decimal_t *from2, decimal_t *to);
int decimal_round(const decimal_t *from, decimal_t *to, int new_scale,
                  decimal_round_mode mode);
int decimal_is_zero(const decimal_t *from);
void max_decimal(int precision, int frac, decimal_t *to);
int decimal_shift(decimal_t *dec, int shift);

/* set a decimal_t to zero */
static inline void decimal_make_zero(decimal_t *dec) {
    dec->buf[0] = 0;
    dec->intg = 1;
    dec->frac = 0;
    dec->sign = 0;
}

/**
  Returns the length of the buffer to hold string representation
  of the decimal (including decimal dot, possible sign and \0)
*/
static inline int decimal_string_size(const decimal_t *dec) {
    return (dec->intg ? dec->intg : 1) + dec->frac + (dec->frac > 0) + 2;
}

/*
  conventions:
    decimal_smth() == 0     -- everything's ok
    decimal_smth() <= 1     -- result is usable, but precision loss is
  possible decimal_smth() <= 2     -- result can be unusable, most significant
  digits could've been lost decimal_smth() >  2     -- no result was generated
*/

#define E_DEC_OK 0
#define E_DEC_TRUNCATED 1
#define E_DEC_OVERFLOW 2
#define E_DEC_DIV_ZERO 4
#define E_DEC_BAD_NUM 8
#define E_DEC_OOM 16

#define E_DEC_ERROR 31
#define E_DEC_FATAL_ERROR 30

#endif  // MYSQL_ABI_CHECK

#endif  // DECIMAL_INCLUDED
