#include <math.h>
#include <immintrin.h>
// Doc: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#ig_expand=4100,5821,337,5821,337,337,678,2938

#include "properties.h"
#include "func_common.h"

static inline __m256i mul_sum_i32x8_i8x32(__m256i x, __m256i y) {
	// get absolute values of x vectors
	__m256i ax = _mm256_sign_epi8(x, x);
	// sign the values of the y vectors
	__m256i sy = _mm256_sign_epi8(y, x);
	// sum pairwise, 32x8 -> 16x16
	__m256i dot = _mm256_maddubs_epi16(ax, sy);
	// sum pairwise, 16x16 -> 8x32
	__m256i ones = _mm256_set1_epi16(1);
	__m256i summed_pairs = _mm256_madd_epi16(ones, dot);

	return summed_pairs;
}

static inline float sum_i8x32_f32(__m256 x) {
	// add pairwise until scalar value is calculated
	__m128 res = _mm256_extractf128_ps(x, 1);
	res = _mm_add_ps(res, _mm256_castps256_ps128(x));
	res = _mm_add_ps(res, _mm_movehl_ps(res, res));
	res = _mm_add_ss(res, _mm_movehdup_ps(res));
	return _mm_cvtss_f32(res);
}

// TODO: investigate why "vec_dot" here and in "funq_q.c" return slightly diffrent result
// TODO: pad values to multiple of 32 to reduce loop overhead
static float vec_dot(int8_t *q, float *sq, int8_t *w, float *sw, int size,
		     int gs)
{
	int ngroups = size / gs;
	float val = 0.0f;
	for (int group = 0; group < ngroups; group++) {
		int start_idx = group * gs;
		// vectorize product of scaling factors to 8x32 float
		__m256 coef = _mm256_set1_ps(sq[group] * sw[group]);
		__m256i sum = _mm256_setzero_si256();
		// multiply and sum in groups of 32
		for (int k = 0; k < gs / 32; k++) {
			// load 32 8 bits int, multiply and sum
			__m256i wv =
			    _mm256_loadu_si256((const __m256i *)
					       &w[start_idx]);
			__m256i qv =
			    _mm256_loadu_si256((const __m256i *)
					       &q[start_idx]);
			sum =
			    _mm256_add_epi32(sum, mul_sum_i32x8_i8x32(wv, qv));
			start_idx += 32;
		}
		// process remaining elements in groups of 16
		if ((gs % 32) / 16 > 0) {
			__m256i wv =
			    _mm256_zextsi128_si256(_mm_loadu_si128((const
						   __m128i *) &w[start_idx]));
			__m256i qv =
			    _mm256_zextsi128_si256(_mm_loadu_si128((const
						   __m128i *) &q[start_idx]));
			sum =
			    _mm256_add_epi32(sum, mul_sum_i32x8_i8x32(wv, qv));
			start_idx += 16;
		}
		// multiply sum by scaling factor and add to result
		__m256 ival8 = _mm256_mul_ps(coef, _mm256_cvtepi32_ps(sum));
		val += sum_i8x32_f32(ival8);
		// process remaining elements
		int32_t ival = 0;
		for (int k = 0; k < gs % 16; k++) {
			ival +=
			    ((int32_t) q[start_idx + k]) *
			    ((int32_t) w[start_idx + k]);
		}
		val += (float) ival * sq[group] * sw[group];
	}

	return val;
}

void linear_q_add_bias(float *xout, int8_t *b, float *sb, int nrows, int gs) {
	int ngroups = nrows / gs;
	for (int group = 0; group < ngroups; group++) {
		for (int k = 0; k < gs - 7; k += 8) {
			// 8x8 int -> 8x32 float
			__m128i q16i =
			    _mm_loadu_si128((__m128i*) &b[group * nrows]);
			__m256i q8i = _mm256_cvtepi8_epi32(q16i);
			__m256 q8f = _mm256_cvtepi32_ps(q8i);
			// multiply scaling factor and bias vector
			__m256 sf = _mm256_set1_ps(sb[group]);
			__m256 val = _mm256_mul_ps(q8f, sf);
			// add bias and store result
			val = _mm256_add_ps(_mm256_loadu_ps(&xout[k]), val);
			_mm256_storeu_ps(&xout[k], val);
		}
		for (int k = gs - (gs % 8); k < gs; k++) {
			xout[k] += (float) b[k] * sb[k / gs];
		}
	}
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
		// multiply
		for (int i = 0; i < nrows; i++) {
			xout[bs * nrows + i] =
			    vec_dot(&x->q[bs * ncols], &x->s[bs * ncols / gs_w],
				    &w[i * ncols], &sw[i * ncols / gs_w], ncols,
				    gs_w);
		}
		// add bias
		if (gs_b > 0) {
			linear_q_add_bias(&xout[bs * nrows], b, sb, nrows,
					  gs_b);
		}
	}
}

// TODO: vectorize bias multiplication
static void conv_q_add_bias(float *xout, int8_t *b, float *sb, int nchannels,
			    int ncols, int gs) {
	for (int c = 0; c < nchannels; c++) {
		for (int i = 0; i < ncols - 7; i += 8) {
			__m256 bias = _mm256_set1_ps((float) b[c] * sb[c / gs]);
			__m256 val = _mm256_loadu_ps(&xout[c * ncols + i]);
			val = _mm256_add_ps(val, bias);
			_mm256_storeu_ps(&xout[c * ncols + i], val);
		}
		for (int i = ncols - (ncols % 8); i < ncols; i++) {
			xout[c * ncols + i] += (float) b[c] * sb[c / gs];
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
				xout[xout_idx + c * ncols + i] =
				    vec_dot(&x->q[x_idx + i * nrows],
					    &x->s[(x_idx + i * nrows) / gs_w],
					    &w[c * nrows],
					    &sw[c * nrows / gs_w], nrows, gs_w);
			}
		}
		if (gs_b > 0) {
			conv_q_add_bias(&xout[bs * nchannels * ncols], b, sb,
					nchannels, ncols, gs_b);
		}
	}
}

static float quantize_find_wmax(float *x, int gs, int start_idx)
{
	// find the maximum absolute value in the current group
	float wmax = 0.0;
	for (int i = 0; i < gs - 7; i += 8) {
		// load 8 32 bits float values
		__m256 x8 = _mm256_loadu_ps(&x[i + start_idx]); // 8x32
		// broadcast -0.0f (100...00 in 32 bits) to 256 bits
		__m256 sign_bit = _mm256_set1_ps(-0.0f); // 8x32
		// ~sign_bit & val
		__m256 max_abs8 = _mm256_andnot_ps(sign_bit, x8);
		// pairwise max, 8x32 -> 4x32
		__m128 max_abs4 =
			_mm_max_ps(_mm256_castps256_ps128(max_abs8),
				_mm256_extractf128_ps(max_abs8, 1));
		// compare upper and lower 2 elements in max_abs4
		max_abs4 =
		    _mm_max_ps(max_abs4, _mm_movehl_ps(max_abs4, max_abs4));
		// compare remaining two values and update current max
		max_abs4 = _mm_max_ss(max_abs4, _mm_movehdup_ps(max_abs4));
		float val = _mm_cvtss_f32(max_abs4);
		wmax = wmax > val ? wmax : val;
	}
	for (int i = gs - (gs % 8); i < gs; i++) {
		float val = fabsf(x[i + start_idx]);
		wmax = wmax > val ? wmax : val;
	}
	return wmax;
}

static void quantize_scale(QuantizedTensor * qx, float *x, int gs,
			   int start_idx, float scale)
{
	float inv_scale = 1.0f / scale;
	__m256 inv_scale8 = _mm256_set1_ps(inv_scale);
	// scale and save weights
	for (int i = 0; i < gs - 7; i += 8) {
		// load 8 32 bits float
		__m256 val = _mm256_loadu_ps(&x[i + start_idx]);
		__m256 quant_value = _mm256_mul_ps(val, inv_scale8);
		quant_value = _mm256_round_ps(quant_value, _MM_ROUND_NEAREST);
		__m256i quantized = _mm256_cvtps_epi32(quant_value);
		// 8x32 -> 16x16
		quantized = _mm256_packs_epi32(quantized, quantized); // (0-4, 0-4, 4-8, 4-8)
		// 16x16 -> 32x8
		quantized = _mm256_packs_epi16(quantized, quantized); // (0-4, 0-4, 0-4, 0-4, 4-8, 4-8, 4-8, 4-8)
		// order is now wrong, following fix order to (0-8, ...)
		__m256i perm = _mm256_setr_epi32(0, 4, 1, 2, 3, 5, 6, 7);
		quantized = _mm256_permutevar8x32_epi32(quantized, perm);
		_mm_storeu_si64(&qx->q[i + start_idx],
				_mm256_castsi256_si128(quantized));
	}
	for (int i = gs - (gs % 8); i < gs; i++) {
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
