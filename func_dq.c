#include <math.h>
#include <stdio.h>
#include <stdbool.h>

#include "config_common.h"
#include "config_dq.h"
#include "func_common.h"

void im2col_q(float *col, float *im, ConvConfigQ cc, int *height, int *width)
{
	im2col_generic(col, im, height, width, cc.ic, cc.ksize, cc.stride,
		       cc.pad, FLOAT);
}

void maxpool_q(float *xout, float *x, int *height, int *width, int nchannels,
	       int ksize, int stride, int pad)
{
	pool_generic(xout, x, height, width, nchannels, ksize, stride, pad,
		     pool_get_pixel_float, MAX_POOL, false);
}

void avgpool_q(float *xout, float *x, int *height, int *width, int nchannels,
	       int ksize, int stride, int pad)
{
	pool_generic(xout, x, height, width, nchannels, ksize, stride, pad,
		     pool_get_pixel_float, AVG_POOL, false);
	for (int i = 0; i < batch_size * nchannels * (*height) * (*width); i++)
		xout[i] /= (ksize * ksize);
}

void relu_q(float *x, int size)
{
	// apply ReLU (Rectified Linear Unit) activation
	for (int i = 0; i < batch_size * size; i++) {
		x[i] = x[i] > 0.0f ? x[i] : 0.0f;
	}
}

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

void linear_q(float *xout, QuantizedTensor *qx, int8_t *p, float *f,
	      LinearConfigQ lc)
{
	// w(nrows,ncols) @ x (ncols,) + b(nrows,) -> xout (nrows,)
	int ncols = lc.in;
	int nrows = lc.out;
	int gs_w = lc.gs_weight;
	int gs_b = lc.gs_bias;
	bool use_zero_point = qx->zero_point != NULL;

	int8_t *w = p + lc.qoffset;
	int8_t *b = w + ncols * nrows;
	float *sw = f + lc.soffset;
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
			    vec_dot(&qx->q[bs * ncols],
				    &qx->scale[bs * ncols / gs_w],
				    &qx->zero_point[bs * ncols / gs_w],
				    &w[i * ncols], &sw[i * ncols / gs_w],
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

void conv_q(float *xout, QuantizedTensor *qx, int8_t *p, float *f,
	    ConvConfigQ cc, int height, int width)
{
	// w (nchannels,in) @ x (nrows,ncols) + b(nchannels,) -> xout (nchannels,ncols) <-> xout (nchannels, height, width)
	int nchannels = cc.oc;
	int ncols = height * width;
	int nrows = cc.ic * cc.ksize * cc.ksize;
	int gs_w = cc.gs_weight;
	int gs_b = cc.gs_bias;
	bool use_zero_point = qx->zero_point != NULL;

	int8_t *w = p + cc.qoffset;
	int8_t *b = w + nchannels * nrows;
	float *sw = f + cc.soffset;
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
				    vec_dot(&qx->q[x_idx + i * nrows],
					    &qx->scale[(x_idx + i * nrows) /
					    gs_w], &qx->zero_point[(x_idx + i *
					    nrows) / gs_w], &w[c * nrows],
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

static void quantize_scale(QuantizedTensor *qx, float *x, int gs,
			   int start_idx, float scale)
{
	bool use_zero_point = qx->zero_point != NULL;
	// scale and save weights
	for (int i = 0; i < gs; i++) {
		int quant = roundf(x[i + start_idx] / scale);
		if (use_zero_point) {
			quant += qx->zero_point[start_idx / gs];
		}
		qx->q[i + start_idx] = quant;
	}
}

void quantize_no_zero_point(QuantizedTensor *qx, float *x, int n, int gs)
{
	int ngroups = batch_size * n / gs;
	float Q_MAX = 127.0f;

	for (int group = 0; group < ngroups; group++) {
		int start_idx = group * gs;
		float wmax = quantize_find_wmax(x, gs, start_idx);
		float scale = wmax / Q_MAX;
		qx->scale[group] = scale;
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

void quantize_zero_point(QuantizedTensor *qx, float *x, int n, int gs)
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
		qx->scale[group] = scale;
		qx->zero_point[group] = roundf(Q_MIN - wmin / scale);
		quantize_scale(qx, x, gs, start_idx, scale);
	}
}

void quantize(QuantizedTensor *qx, float *x, int n, int gs)
{
	bool use_zero_point = qx->zero_point != NULL;
	if (use_zero_point) {
		quantize_zero_point(qx, x, n, gs);
	} else {
		quantize_no_zero_point(qx, x, n, gs);
	}
}

// FIX: unify functions below with func_sq.c
void head(Runstate *s, float *images, int8_t *p, float *f, ConvConfigQ *cc,
	  ActivationConfigQ *ac, int *h, int *w)
{
	float *x = s->x;
	float *x2 = s->x2;
	QuantizedTensor *xq = &s->xq;

	im2col_q(x, images, cc[0], h, w);
	quantize(xq, x, cc[0].ic * cc[0].ksize * cc[0].ksize * (*h) * (*w),
		 cc[0].gs_weight);
	conv_q(x, xq, p, f, cc[0], *h, *w);
	relu_q(x, cc[0].oc * (*h) * (*w));
	maxpool_q(x2, x, h, w, cc[0].oc, 3, 2, 1);
	matcopy_float(x, x2, cc[0].oc * (*h) * (*w));
}

void tail(Runstate *s, int8_t *p, float *f, ConvConfigQ *cc,
	  LinearConfigQ *lc, BnConfig *bc, ActivationConfigQ *ac, int h, int w,
	  int i, int ia)
{
	float *x = s->x;
	float *x2 = s->x2;
	QuantizedTensor *xq = &s->xq;

	concat_pool(x2, x, &h, &w, cc[i - 1].oc, h, 1, 0);
	batchnorm(x, x2, f, bc[0], 1);
	quantize(xq, x, lc[0].in, lc[0].gs_weight);
	linear_q(x2, xq, p, f, lc[0]);
	relu_q(x2, lc[0].out);
	batchnorm(x, x2, f, bc[1], 1);
	quantize(xq, x, lc[1].in, lc[1].gs_weight);
	linear_q(x2, xq, p, f, lc[1]);
	matcopy_float(x, x2, lc[1].out);
}

void basic_block(Runstate *s, int8_t *p, float *f, ConvConfigQ *cc,
		 ActivationConfigQ *ac, int *h, int *w, int *i, int *ia,
		 bool downsample)
{
	float *x = s->x;
	float *x2 = s->x2;
	float *x3 = s->x3;
	QuantizedTensor *xq = &s->xq;

	int h_prev = *h;
	int w_prev = *w;

	im2col_q(x3, x, cc[*i], h, w);
	quantize(xq, x3, cc[*i].ic * cc[*i].ksize * cc[*i].ksize * (*h) * (*w),
		 cc[*i].gs_weight);
	conv_q(x3, xq, p, f, cc[*i], *h, *w);
	relu_q(x3, cc[*i].oc * (*h) * (*w));
	im2col_q(x, x3, cc[*i + 1], h, w);
	quantize(xq, x, cc[*i + 1].ic * cc[*i + 1].ksize * cc[*i + 1].ksize *
		 (*h) * (*w), cc[*i + 1].gs_weight);
	conv_q(x, xq, p, f, cc[*i + 1], *h, *w);
	// perform downsampling for skip connection
	if (downsample) {
		im2col_q(x3, x2, cc[*i + 2], &h_prev, &w_prev);
		quantize(xq, x3, cc[*i + 2].ic * cc[*i + 2].ksize *
			 cc[*i + 2].ksize * (*h) * (*w), cc[*i + 2].gs_weight);
		conv_q(x2, xq, p, f, cc[*i + 2], *h, *w);
	}

	int oc = downsample ? cc[*i + 2].oc : cc[*i + 1].oc;
	int size = oc * (*h) * (*w);
	*i += downsample ? 3 : 2;
	matadd(x, x2, size);
	relu_q(x, size);
	matcopy_float(x2, x, size);
}

void bottleneck(Runstate *s, int8_t *p, float *f, ConvConfigQ *cc,
		ActivationConfigQ *ac, int *h, int *w, int *i, int *ia,
		bool downsample)
{
	float *x = s->x;
	float *x2 = s->x2;
	float *x3 = s->x3;
	QuantizedTensor *xq = &s->xq;

	int h_prev = *h;
	int w_prev = *w;

	im2col_q(x3, x, cc[*i], h, w);
	quantize(xq, x3, cc[*i].ic * cc[*i].ksize * cc[*i].ksize * (*h) * (*w),
		 cc[*i].gs_weight);
	conv_q(x3, xq, p, f, cc[*i], *h, *w);
	relu_q(x3, cc[*i].oc * (*h) * (*w));
	im2col_q(x, x3, cc[*i + 1], h, w);
	quantize(xq, x, cc[*i + 1].ic * cc[*i + 1].ksize * cc[*i + 1].ksize *
		 (*h) * (*w), cc[*i + 1].gs_weight);
	conv_q(x, xq, p, f, cc[*i + 1], *h, *w);
	relu_q(x, cc[*i + 1].oc * (*h) * (*w));
	im2col_q(x3, x, cc[*i + 2], h, w);
	quantize(xq, x3, cc[*i + 2].ic * cc[*i + 2].ksize * cc[*i + 2].ksize *
		 (*h) * (*w), cc[*i + 2].gs_weight);
	conv_q(x, xq, p, f, cc[*i + 2], *h, *w);
	// perform downsampling for skip connection
	if (downsample) {
		im2col_q(x3, x2, cc[*i + 3], &h_prev, &w_prev);
		quantize(xq, x3, cc[*i + 3].ic * cc[*i + 3].ksize *
			 cc[*i + 3].ksize * (*h) * (*w), cc[*i + 3].gs_weight);
		conv_q(x2, xq, p, f, cc[*i + 3], *h, *w);
	}

	int oc = downsample ? cc[*i + 3].oc : cc[*i + 2].oc;
	int size = oc * (*h) * (*w);
	*i += downsample ? 4 : 3;
	matadd(x, x2, size);
	relu_q(x, size);
	matcopy_float(x2, x, size);
}
