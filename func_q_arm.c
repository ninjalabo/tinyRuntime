#include <math.h>
#include <arm_neon.h>

#include "properties.h"

inline static int32_t vmul(int8x16_t x, int8x16_t y) {
        int16x8_t p0 = vmull_s8(vget_low_s8(x), vget_low_s8(y));
        int16x8_t p1 = vmull_s8(vget_high_s8(x), vget_high_s8(y));

        return vaddvq_s32(vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)));
}

// TODO: loop unrolling
// TODO: compare speed vs. ggml
// TODO: optimize more, e.g., vectorize bias addition
// TODO: add test in workflow
static float vec_dot(int8_t *q, float *sq, int8_t *w, float *sw, int size,
		     int gs) 
{
        int num_groups = size / gs;
        float val = 0.0f;
        for (int group = 0; group < num_groups; group++) {
		int start_idx = group * gs;
		int32_t ival = 0;
		for (int k = 0; k < gs - 15; k += 16) {
			int8x16_t wv = vld1q_s8(&w[start_idx + k]);
			int8x16_t xv = vld1q_s8(&q[start_idx + k]);
			ival += vmul(wv, xv);
		}
		// process remaining elements
                for (int k = gs - (gs % 16); k < gs; k++) {
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

	for (int i = 0; i < nrows; i++) {
		float val =  
		    vec_dot(x->q, x->s, &w[i * ncols], &sw[i * ncols / gs_w],
		    	    ncols, gs_w);
		float bias_val = 
		    gs_b > 0 ? ((float) b[i]) * sb[i / gs_b] : 0.0f;
		xout[i] = val + bias_val;
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
	for (int c = 0; c < nchannels; c++) {
		for (int i = 0; i < ncols; i++) {
			float val =
			    vec_dot(&x->q[i * nrows], &x->s[i * nrows / gs_w],
			    	      &w[c * nrows], &sw[c * nrows / gs_w],
				      nrows, gs_w);
			float bias_val =
			    gs_b > 0 ? ((float) b[c]) * sb[i / gs_b] : 0.0f;
			xout[c * ncols + i] = val + bias_val;
		}
	}
}

static float quantize_find_wmax(float *x, int gs, int start_idx)
{
	// find the maximum absolute value in the current group
	float wmax = 0.0;
	float val;
	for (int i = 0; i < gs - 3; i += 4) {
		val = 
		    vmaxvq_f32(vabsq_f32(vld1q_f32(&x[i + start_idx])));
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
	int num_groups = n / gs;
	float Q_MAX = 127.0f;

	for (int group = 0; group < num_groups; group++) {
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
	int num_groups = ncols / gs;
	float Q_MAX = 127.0f;

	for (int row = 0; row < nrows; row++) {
		for (int group = 0; group < num_groups; group++) {
			int start_idx = row * ncols + group * gs;
			float wmax = quantize_find_wmax(x, gs, start_idx);
			float scale = wmax / Q_MAX;
			qx->s[row * num_groups + group] = scale;
			quantize_scale(qx, x, gs, start_idx, scale);
		}
	}
}