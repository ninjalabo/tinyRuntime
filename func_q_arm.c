#include <math.h>
#include <arm_neon.h>
// Doc: https://arm-software.github.io/acle/neon_intrinsics/advsimd.html

#include "properties.h"
#include "func_common.h"

inline static int32x4_t mul_sum_i8x16_i32x4(int8x16_t x, int8x16_t y) {
	// multiply lower 8 numbers elementwise
        int16x8_t p0 = vmull_s8(vget_low_s8(x), vget_low_s8(y));
	// multiply higher 8 numbers elementwise
        int16x8_t p1 = vmull_s8(vget_high_s8(x), vget_high_s8(y));
	// sum until 4 32 bit integers
        return vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1));
}

static void vec_dot_64(int32x4_t *sum0, int32x4_t *sum1, int8_t *q, int8_t *w)
{
	// vectorize w to 4 16x8 bits vectors
	int8x16_t w0 = vld1q_s8(w);
	int8x16_t w1 = vld1q_s8(w + 16);
	int8x16_t w2 = vld1q_s8(w + 32);
	int8x16_t w3 = vld1q_s8(w + 48);
	// vectorize q to 4 16x8 bits vectors
	int8x16_t q0 = vld1q_s8(q);
	int8x16_t q1 = vld1q_s8(q + 16);
	int8x16_t q2 = vld1q_s8(q + 32);
	int8x16_t q3 = vld1q_s8(q + 48);
	// multiply elementwise and sum
	int32x4_t ival0 =
	    vaddq_s32(mul_sum_i8x16_i32x4(w0, q0), mul_sum_i8x16_i32x4(w1, q1));
	int32x4_t ival1 =
	    vaddq_s32(mul_sum_i8x16_i32x4(w2, q2), mul_sum_i8x16_i32x4(w3, q3));
	*sum0 = vaddq_s32(*sum0, ival0);
	*sum1 = vaddq_s32(*sum1, ival1);
}

static inline int32_t vec_dot_32(int8_t *q, int8_t *w)
{
	// vectorize w to 2 16x8 bits vectors
	int8x16_t w0 = vld1q_s8(w);
	int8x16_t w1 = vld1q_s8(w + 16);
	// vectorize q to 2 16x8 bits vectors
	int8x16_t q0 = vld1q_s8(q);
	int8x16_t q1 = vld1q_s8(q + 16);
	// multiply elementwise and sum
	int32x4_t sum0 = mul_sum_i8x16_i32x4(w0, q0);
	int32x4_t sum1 = mul_sum_i8x16_i32x4(w1, q1);
	int32_t ival = vaddvq_s32(vaddq_s32(sum0, sum1));

	return ival;
}

static inline int32_t vec_dot_16(int8_t *q, int8_t *w)
{
	// vectorize
	int8x16_t wv = vld1q_s8(w);
	int8x16_t xv = vld1q_s8(q);
	// multiply elementwise and sum
	int32_t ival = vaddvq_s32(mul_sum_i8x16_i32x4(wv, xv));

	return ival;
}

// TODO: possible use padding to reduce loop overhead instead of checking for remaining elements
// TODO: add test in workflow
static float vec_dot(int8_t *q, float *sq, int8_t *w, float *sw, int size,
		     int gs) 
{
	int ngroups = size / gs;
        float val = 0.0f;
        for (int group = 0; group < ngroups; group++) {
		int start_idx = group * gs;
		int32_t ival = 0;
		// multiply and sum in groups of 64
		int32x4_t sum0 = vdupq_n_f32(0.0f);
		int32x4_t sum1 = vdupq_n_f32(0.0f);
		for (int k = 0; k < gs / 64; k++) {
			vec_dot_64(&sum0, &sum1, &q[start_idx],
				   &w[start_idx]);
			start_idx += 64;
		}
		ival += vaddvq_s32(vaddq_s32(sum0, sum1));
		// process remaining elements in groups of 32, 16 and 1
		if ((gs % 64) / 32 > 0) {
			ival += vec_dot_32(&q[start_idx], &w[start_idx]);
			start_idx += 32;
		}
		if ((gs % 32) / 16 > 0) {
			ival += vec_dot_16(&q[start_idx], &w[start_idx]);
			start_idx += 16;
		}
		for (int k = 0; k < gs % 16; k++) {
                        ival +=
                            ((int32_t) q[start_idx + k]) *
                            ((int32_t) w[start_idx + k]);
                }
		val += (float) ival * sq[group] * sw[group];
        }

	return val;
}

void linear_q(float *xout, QuantizedTensor * x, int8_t * p, float *sf,
	      LinearConfigQ lc)
{
	// w(nrows,ncols) @ x (ncols,) + b(nrows,) -> xout (nrows,)
	int ncols = lc.in;
	int nrows = lc.out;
	int gs_w = lc.gs_weight;
	int gs_b = lc.gs_bias;

	int8_t *w = p + lc.qoffset;
	int8_t *b = w + ncols * nrows;
	float *sw = sf + lc.soffset;
	float *sb = sw + ncols * nrows / gs_w;
	for (int bs = 0; bs < batch_size; bs++) {
		for (int i = 0; i < nrows; i++) {
			float val =
			    vec_dot(&x->q[bs * ncols], &x->s[bs * ncols / gs_w],
				    &w[i * ncols], &sw[i * ncols / gs_w], ncols,
				    gs_w);
			float bias_val =
			    gs_b > 0 ? ((float) b[i]) * sb[i / gs_b] : 0.0f;
			xout[bs * nrows + i] = val + bias_val;
		}
	}
}

void conv_q(float *xout, QuantizedTensor * x, int8_t * p, float *sf,
	    ConvConfigQ cc, int height, int width)
{
	// w (nchannels,nrows) @ x.T (nrows,ncols) + b(nchannels,) -> xout (nchannels,ncols) <-> xout (nchannels, height, width)
	int nchannels = cc.oc;
	int ncols = height * width;
	int nrows = cc.ic * cc.ksize * cc.ksize;
	int gs_w = cc.gs_weight;
	int gs_b = cc.gs_bias;

	int8_t *w = p + cc.qoffset;
	int8_t *b = w + nchannels * nrows;
	float *sw = sf + cc.soffset;
	float *sb = sw + nchannels * nrows / gs_w;
	for (int bs = 0; bs < batch_size; bs++) {
		int x_idx = bs * nrows * ncols;
		int xout_idx = bs * nchannels * ncols;
		for (int c = 0; c < nchannels; c++) {
			for (int i = 0; i < ncols; i++) {
				float val =
				    vec_dot(&x->q[x_idx + i * nrows],
					    &x->s[(x_idx + i * nrows) / gs_w],
					    &w[c * nrows],
					    &sw[c * nrows / gs_w], nrows, gs_w);
				float bias_val =
				gs_b > 0 ? (float) b[c] * sb[c / gs_b] : 0.0f;
				xout[xout_idx + c * ncols + i] = val + bias_val;
			}
		}
	}
}

static float quantize_find_wmax(float *x, int gs, int start_idx)
{
	// find the maximum absolute value in the current group
	float wmax = 0.0;
	float val;
	for (int i = 0; i < gs - 3; i += 4) {
		val = vmaxvq_f32(vabsq_f32(vld1q_f32(&x[i + start_idx])));
		wmax = wmax > val ? wmax : val;
	}
	for (int i = gs - (gs % 4); i < gs; i++) {
		val = fabsf(x[i + start_idx]);
		wmax = wmax > val ? wmax : val;
	}
	return wmax;
}

/* NOTE: small difference in results between `func_q.c` and `func_q_arm.c` occurs here. The difference comes from: 
1. `roundf` and `vcvtnq_s32_f32` probably treat midway values differently
2. value here is multiplied by inverse scale instead of dividing by scale (since limited precision truncate furthest decimal places)
*/
static void quantize_scale(QuantizedTensor * qx, float *x, int gs,
			   int start_idx, float scale)
{
	float inv_scale = 1.0f / scale;
	// scale and save weights
	for (int i = 0; i < gs - 3; i += 4) {
		float32x4_t quant_value = 
		    vmulq_n_f32(vld1q_f32(&x[i + start_idx]), inv_scale);
		int32x4_t quantized = vcvtnq_s32_f32(quant_value);
		qx->q[i + start_idx] = vgetq_lane_s32(quantized, 0);
		qx->q[i + start_idx + 1] = vgetq_lane_s32(quantized, 1);
		qx->q[i + start_idx + 2] = vgetq_lane_s32(quantized, 2);
		qx->q[i + start_idx + 3] = vgetq_lane_s32(quantized, 3);
	}
	for (int i = gs - (gs % 4); i < gs; i++) {
		float quant_value = x[i + start_idx] * inv_scale;
		qx->q[i + start_idx] = roundf(quant_value);
	}
}

void quantize(QuantizedTensor * qx, float *x, int n, int gs)
{
	int ngroups = batch_size * n / gs;
	float Q_MAX = 127.0f;

	for (int group = 0; group < ngroups; group++) {
		int start_idx = group * gs;
		float wmax = quantize_find_wmax(x, gs, start_idx);
		float scale = wmax / Q_MAX;
		qx->s[group] = scale;
		quantize_scale(qx, x, gs, start_idx, scale);
	}
}

void quantize2d(QuantizedTensor * qx, float *x, ConvConfigQ cc, int nrows)
{
	int ncols = cc.ic * cc.ksize * cc.ksize;
	int gs = cc.gs_weight;
	int ngroups = batch_size * nrows * ncols / gs;
	float Q_MAX = 127.0f;

	for (int group = 0; group < ngroups; group++) {
		int start_idx = group * gs;
		float wmax = quantize_find_wmax(x, gs, start_idx);
		float scale = wmax / Q_MAX;
		qx->s[group] = scale;
		quantize_scale(qx, x, gs, start_idx, scale);
	}
}
