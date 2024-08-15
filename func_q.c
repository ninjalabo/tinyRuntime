#include <math.h>
#include <stdio.h>
#include <stdbool.h>

#include "properties.h"
#include "func_common.h"

static float vec_dot_no_zero_point(int8_t *q, float *sq, int *zq, int8_t *w,
				   float *sw, int *zw, int size, int gs)
{
	int ngroups = size / gs;
	float val = 0.0f;
	for (int group = 0; group < ngroups; group++) {
		int start_idx = group * gs;
		int32_t ival = 0;
		for (int k = 0; k < gs; k++) {
			ival +=
			    ((int32_t) q[start_idx + k]) *
			    ((int32_t) w[start_idx + k]);
		}
		val += (float) ival * sq[group] * sw[group];
	}

	return val;
}

static float vec_dot_zero_point(int8_t *q, float *sq, int *zq, int8_t *w,
				float *sw, int *zw, int size, int gs)
{
	int ngroups = size / gs;
	float val = 0.0f;

	for (int group = 0; group < ngroups; group++) {
		int start_idx = group * gs;
		int32_t ival = 0;
		for (int k = 0; k < gs; k++) {
			ival +=
			    ((int32_t) q[start_idx + k] - zq[group]) *
			    ((int32_t) w[start_idx + k] - zw[group]);
		}
		val += (float) ival * sq[group] * sw[group];
	}

	return val;
}

float (*vec_dot)(int8_t *, float *, int *, int8_t *, float *, int *, int,
		 int);

void linear_q(float *xout, QuantizedTensor * x, int8_t * p, float *sf,
	      LinearConfigQ lc)
{
	// w(nrows,ncols) @ x (ncols,) + b(nrows,) -> xout (nrows,)
	int ncols = lc.in;
	int nrows = lc.out;
	int gs_w = lc.gs_weight;
	int gs_b = lc.gs_bias;
	bool use_zero_point = x->zp != NULL;

	int8_t *w = p + lc.qoffset;
	int8_t *b = w + ncols * nrows;
	float *sw = sf + lc.soffset;
	int *zw = (int *) (use_zero_point ? sw + ncols * nrows / gs_w : sw);
	float *sb = NULL;
	int *zb = NULL;
	if (gs_b > 0) { // if layer has bias
		sb = (float *) (zw + ncols * nrows / gs_w);
		zb = (int *) (sb + nrows / gs_b);
	}
	// Determine which function to use
	vec_dot = use_zero_point ? vec_dot_zero_point : vec_dot_no_zero_point;

	for (int bs = 0; bs < batch_size; bs++) {
		for (int i = 0; i < nrows; i++) {
			xout[bs * nrows + i] =
			    vec_dot(&x->q[bs * ncols], &x->s[bs * ncols / gs_w],
				    &x->zp[bs * ncols / gs_w], &w[i * ncols],
				    &sw[i * ncols / gs_w],
				    &zw[i * ncols / gs_w], ncols, gs_w);
			if (gs_b > 0) {
				float bias_val =
				    use_zero_point ?
				    (b[i] - zb[i / gs_b]) * sb[i / gs_b] :
				    b[i] * sb[i / gs_b];
				xout[bs * nrows + i] += bias_val;
			}
		}
	}
}

void conv_q(float *xout, QuantizedTensor * x, int8_t * p, float *sf,
	    ConvConfigQ cc, int height, int width)
{
	// w (nchannels,in) @ x (nrows,ncols) + b(nchannels,) -> xout (nchannels,ncols) <-> xout (nchannels, height, width)
	int nchannels = cc.oc;
	int ncols = height * width;
	int nrows = cc.ic * cc.ksize * cc.ksize;
	int gs_w = cc.gs_weight;
	int gs_b = cc.gs_bias;
	bool use_zero_point = x->zp != NULL;

	int8_t *w = p + cc.qoffset;
	int8_t *b = w + nchannels * nrows;
	float *sw = sf + cc.soffset;
	int *zw = (int *) (use_zero_point ? sw + nchannels * nrows / gs_w : sw);
	float *sb = NULL;
	int *zb = NULL;
	if (gs_b > 0) { // if layer has bias
		sb = (float *) (zw + nchannels * nrows / gs_w);
		zb = (int *) (sb + nchannels / gs_b);
	}
	// Determine which function to use
	vec_dot = use_zero_point ? vec_dot_zero_point : vec_dot_no_zero_point;

	for (int bs = 0; bs < batch_size; bs++) {
		int x_idx = bs * nrows * ncols;
		int xout_idx = bs * nchannels * ncols;
		for (int c = 0; c < nchannels; c++) {
			for (int i = 0; i < ncols; i++) {
				xout[xout_idx + c * ncols + i] =
				    vec_dot(&x->q[x_idx + i * nrows],
					    &x->s[(x_idx + i * nrows) / gs_w],
					    &x->zp[(x_idx + i * nrows) / gs_w],
					    &w[c * nrows],
					    &sw[c * nrows / gs_w],
					    &zw[c * nrows / gs_w],
					    nrows, gs_w);
				if (gs_b > 0) {
					float bias_val =
					    use_zero_point ?
					    (b[c] - zb[c / gs_b]) *
					    sb[c / gs_b] : b[c] * sb[c / gs_b];
					xout[xout_idx + c * ncols + i] +=
					    bias_val;
				}
			}
		}
	}
}

static float quantize_find_wmax(float *x, int gs, int start_idx)
{
	// find the maximum absolute value in the current group
	float wmax = 0.0;
	for (int i = 0; i < gs; i++) {
		float val = fabsf(x[i + start_idx]);
		wmax = wmax > val ? wmax : val;
	}
	return wmax;
}

static void quantize_scale(QuantizedTensor * qx, float *x, int gs,
			   int start_idx, float scale)
{
	bool use_zero_point = qx->zp != NULL;
	// scale and save weights
	for (int i = 0; i < gs; i++) {
		int quant = roundf(x[i + start_idx] / scale);
		if (use_zero_point) {
			quant += qx->zp[start_idx / gs];
		}
		qx->q[i + start_idx] = quant;
	}
}

void quantize_no_zero_point(QuantizedTensor * qx, float *x, int n, int gs)
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

static void quantize_find_wmax_wmin(float *x, int gs, int start_idx,
				     float *wmax, float *wmin)
{
	// find the maximum absolute value in the current group
	*wmax = -INFINITY;
	*wmin = INFINITY;
	for (int i = 0; i < gs; i++) {
		float val = x[i + start_idx];
		*wmax = *wmax < val ? val: *wmax;
		*wmin = *wmin > val ? val : *wmin;
	}
}

void quantize_zero_point(QuantizedTensor * qx, float *x, int n, int gs)
{
	int ngroups = batch_size * n / gs;
	float Q_RANGE = 255.0f;
	float Q_MIN = - 128.0f;

	for (int group = 0; group < ngroups; group++) {
		int start_idx = group * gs;
		float wmax, wmin;
		quantize_find_wmax_wmin(x, gs, start_idx, &wmax, &wmin);
		float scale = (wmax - wmin) / Q_RANGE;
		if (scale == 0.0f) {
			scale = 1e-4; // avoid division by zero
		}
		qx->s[group] = scale;
		qx->zp[group] = roundf(Q_MIN - wmin / scale);
		quantize_scale(qx, x, gs, start_idx, scale);
	}
}

void quantize(QuantizedTensor * qx, float *x, int n, int gs)
{
	bool use_zero_point = qx->zp != NULL;
	if (use_zero_point) {
		quantize_zero_point(qx, x, n, gs);
	} else {
		quantize_no_zero_point(qx, x, n, gs);
	}
}
