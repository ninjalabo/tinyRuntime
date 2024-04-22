#include "CppUTest/TestHarness.h"

#include "func.h"
#include "func_q.h"

#define TOL 0.00001
#define eps 0.00001f
#define Q_MAX 127.0f

TEST_GROUP(TestGroup)
{
        void setup() {
		batch_size = 2;
        }
        void teardown() {
        }
};

TEST(TestGroup, ReadImagenette)
{
	int nch = 3, h = 224, w = 224;
	int img_sz = nch * h * w;
	float ref[batch_size * img_sz];
	for (int i = 0; i < batch_size * img_sz; i++) {
		ref[i] = i;
	}
	const char *paths[] =
	    { "test_read_imagenette1.bin", "test_read_imagenette2.bin" };
	FILE *file = fopen(paths[0], "wb");
	fwrite(ref, sizeof(float), img_sz, file);
	fclose(file);
	file = fopen(paths[1], "wb");
	fwrite(&ref[img_sz], sizeof(float), img_sz, file);
	fclose(file);

	float images[batch_size * img_sz];
	read_imagenette_image((char**) paths, images);
	MEMCMP_EQUAL(ref, images, batch_size * img_sz * sizeof(float));

	remove(paths[0]);
	remove(paths[1]);
}

TEST(TestGroup, Linear)
{
        int in = 2, out = 2, offset = 1;
        int gs_w = 18, gs_b = 2;

        float x[] = { 1.0f, 2.0f, 3.0f, 4.0f };
        float p[] = { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f };
        LinearConfig lc = { in, out, offset, 1 };

        float xout[batch_size * out];
        linear(xout, x, p, lc);

        float ref[] = { 6.0f, 13.0f, 12.0f, 27.0f };
        for (int i = 0; i < batch_size * out; i++) {
                DOUBLES_EQUAL(ref[i], xout[i], TOL);
        }

        in = 36;
        int size_q = in * out + out + offset;
        int size_s = in * out / gs_w + out / gs_b + offset;
        int8_t qx_q[batch_size * in];
        float qx_sf[batch_size * in / gs_w];
        for (int i = 0; i < batch_size * in; i++)
                qx_q[i] = i / 9;
        for (int i = 0; i < batch_size * in / gs_w; i++)
                qx_sf[i] = i + 1;
        QuantizedTensor qx = {.q = qx_q,.s = qx_sf };
        int8_t pq[size_q];
        float sf[size_s];
        for (int i = 0; i < size_q; i++)
                pq[i] = (i - 1) / 9;
        for (int i = 0; i < size_s; i++)
                sf[i] = i;
        LinearConfigQ lcq = { in, out, offset, offset, gs_w, gs_b };

        float xout_q[batch_size * out];
        linear_q(xout_q, &qx, pq, sf, lcq);

        float ref_q[] = { 517.0f, 2551.0f, 2551.0f, 15601.0f };
        for (int i = 0; i < batch_size * out; i++) {
                DOUBLES_EQUAL(xout_q[i], ref_q[i],  TOL);
        }
}

TEST(TestGroup, Im2col)
{
        int h = 2, w = 2, hq = 2, wq = 2;
        int ksize = 2, stride = 2, pad = 1, ic = 2;
        int col_size = ksize * ksize * ic * h * w;

        float im[] =
	    { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
	    9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f };
        ConvConfig cc = { ksize, stride, pad, ic, 0, 0, 0 };	// 0 doesn't affect
        ConvConfigQ cc_q = { ksize, stride, pad, ic, 0, 0, 0, 0, 0 };

        float col[batch_size * col_size];
        im2col(col, im, cc, &h, &w);

        float col_q[batch_size * col_size];
        im2col_q(col_q, im, cc_q, &hq, &wq);

        float ref[] = {
                0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 5.0f,
		0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 6.0f, 0.0f,
		0.0f, 3.0f, 0.0f, 0.0f, 0.0f, 7.0f, 0.0f, 0.0f,
		4.0f, 0.0f, 0.0f, 0.0f, 8.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 9.0f, 0.0f, 0.0f, 0.0f, 13.0f,
		0.0f, 0.0f, 10.0f, 0.0f, 0.0f, 0.0f, 14.0f, 0.0f,
		0.0f, 11.0f, 0.0f, 0.0f, 0.0f, 15.0f, 0.0f, 0.0f,
		12.0f, 0.0f, 0.0f, 0.0f, 16.0f, 0.0f, 0.0f, 0.0f,
        };
        MEMCMP_EQUAL(ref, col, col_size * sizeof(float));
        MEMCMP_EQUAL(ref, col_q, col_size * sizeof(float));
        DOUBLES_EQUAL(h, hq, TOL);
        DOUBLES_EQUAL(w, wq, TOL);
}

TEST(TestGroup, Conv)
{
        int h = 1, w = 1;
        int ksize = 2, stride = 1, pad = 1, ic = 1, oc = 2;
        int offset = 1, bias = 1, gs_w = 20, gs_b = 2;

        float x[] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
        float p[] =
            { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 1.0f,
            2.0f };
        ConvConfig cc = { ksize, stride, pad, ic, oc, offset, bias };

        float xout[batch_size * oc * h * w];
        conv(xout, x, p, cc, h, w);

        float ref[] = { 31.0f, 72.0f, 71.0f, 176.0f };
        for (int i = 0; i < batch_size * oc * h * w; i++) {
                DOUBLES_EQUAL(ref[i], xout[i], TOL);
        }

        ic = 10;
        int nrows = ksize * ksize * ic;
        int size_q = nrows * oc + oc + offset;
        int size_s = nrows * oc / gs_w + oc / gs_b + offset;
        int8_t qx_q[batch_size * nrows];
        float qx_sf[batch_size * nrows / gs_w];
        for (int i = 0; i < batch_size * nrows; i++)
                qx_q[i] = i / 10;
        for (int i = 0; i < batch_size * nrows / gs_w; i++)
                qx_sf[i] = i + 1;
        QuantizedTensor qx = {.q = qx_q,.s = qx_sf };
        int8_t pq[size_q];
        float sf[size_s];
        for (int i = 0; i < size_q; i++)
                pq[i] = (i - 1) / 10;
        for (int i = 0; i < size_s; i++)
                sf[i] = i;
        ConvConfigQ cc_q =
            { ksize, stride, pad, ic, oc, offset, offset, gs_w, gs_b };

        float xout_q[batch_size * oc * h * w];
        conv_q(xout_q, &qx, pq, sf, cc_q, h, w);

        float ref_q[] = { 570.0f, 2830.0f, 2830.0f, 17330.0f };
        for (int i = 0; i < batch_size * oc * h * w; i++) {
                DOUBLES_EQUAL(ref_q[i], xout_q[i], TOL);
        }
}

TEST(TestGroup, Batchnorm)
{
        int h = 2, w = 1, ic = 2, offset = 1;

        float x[] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
        float p[] = { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 1.0f, 2.0f };
        BnConfig bc = { ic, offset };

        float xout[batch_size * ic * h * w];
        batchnorm(xout, x, p, bc, h * w);

        float ref[] =
            { 3.8999955f, 4.8999905001f, 7.9597880752f, 9.373998102f,
            7.8999755f, 8.8999705f, 13.61662818f, 15.03083821f };

        for (int i = 0; i < batch_size * ic * h * w; i++) {
                DOUBLES_EQUAL(ref[i], xout[i], TOL);
        }
}

TEST(TestGroup, Pool)
{
        int nch = 2, h_before = 1, w_before = 1;
        int ksize = 2, stride = 1, pad = 1;
        int h_after = 2, w_after = 2;
        int out_size = nch * h_after * w_after;

        float x[] = { 1.0f, -1.0f, 2.0f, -2.0f};
	float refs[2][batch_size * out_size];
	float ref_max[] =
	    { 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
	    2.0f, 2.0f, 2.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f };
	float ref_avg[] =
            { 0.25f, 0.25f, 0.25f, 0.25f, -0.25f, -0.25f, -0.25f, -0.25f,
            0.5f, 0.5f, 0.5f, 0.5f, -0.5f, -0.5f, -0.5f, -0.5f };
	memcpy(refs[0], ref_max, batch_size * out_size * sizeof(float));
	memcpy(refs[1], ref_avg, batch_size * out_size * sizeof(float));
        int h, w;
        float xout[batch_size * out_size];
        void (*pool_functions[])(float *, float *, int *, int *, int, int, int,
                                 int) = { maxpool, avgpool };

        for (int i = 0; i < 2; i++) {
                h = h_before;
                w = w_before;

                pool_functions[i] (xout, x, &h, &w, nch, ksize, stride, pad);
                DOUBLES_EQUAL(h_after, h, TOL);
                DOUBLES_EQUAL(w_after, w, TOL);

                for (int j = 0; j < batch_size * nch * h * w; j++) {
                        DOUBLES_EQUAL(refs[i][j], xout[j], TOL);
                }
        }
}

TEST(TestGroup, Matadd)
{
        float x[] = { 1.0f, 2.0f, 3.0f, 4.0f };
        float y[] = { 5.0f, 6.0f, 7.0f, 8.0f };
        matadd(x, y, 2);
        float ref[] = { 6.0f, 8.0f, 10.0f, 12.0f };
        for (int i = 0; i < 4; i ++) {
                DOUBLES_EQUAL(ref[i], x[i], TOL);
        }
}

TEST(TestGroup, Relu)
{
        float x[] = { 1.0f, -1.0f, 2.0f, -2.0f };
        relu(x, 2);
        float ref[] = { 1.0f, 0.0f, 2.0f, 0.0f };
        for (int i = 0; i < 4; i ++) {
                DOUBLES_EQUAL(ref[i], x[i], TOL);
        }
}

TEST(TestGroup, Softmax)
{
        float x[] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
        softmax(x, 3);
        DOUBLES_EQUAL(1.0, x[0] + x[1] + x[2], TOL);
        DOUBLES_EQUAL(1.0, x[3] + x[4] + x[5], TOL);
}

TEST(TestGroup, Matcopy)
{
	int size = 3;
	float x[] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
	float xout[batch_size * size];
	matcopy_float(xout, x, size);
	for (int i = 0; i < batch_size * size; i++) {
		DOUBLES_EQUAL(xout[i], x[i], TOL);
	}
}

TEST(TestGroup, Quantize)
{
        int size = 4, gs = 2;
        float x[] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
        float ref_sf[] =
            { 2.0f / Q_MAX, 4.0f / Q_MAX, 6.0f / Q_MAX, 8.0f / Q_MAX };
        int8_t ref_q[] = { 64, 127, 95, 127, 106, 127, 111, 127 };

        int8_t qx_q[batch_size * size];
        float qx_s[batch_size * size / gs];
        QuantizedTensor qx = {.q = qx_q,.s = qx_s };
        quantize(&qx, x, size, gs);

        for (int i = 0; i < batch_size * size / gs; i++) {
                DOUBLES_EQUAL(ref_sf[i], qx.s[i], TOL);
        }
        MEMCMP_EQUAL(ref_q, qx.q, batch_size * size * sizeof(int8_t));
}

TEST(TestGroup, Quantize2d)
{
        int ksize = 2, ic = 1, gs = 2, ncols = 2;
        int nrows = ksize * ksize * ic, size = nrows * ncols;

        float x[] =
            { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f };
        ConvConfigQ cc = { ksize, 0, 0, ic, 0, 0, 0, gs, 0 };
        float ref_sf[] =
            { 2.0f / Q_MAX, 4.0f / Q_MAX, 6.0f / Q_MAX, 8.0f / Q_MAX,
            10.0f / Q_MAX, 12.0f / Q_MAX, 14.0f / Q_MAX, 16.0f / Q_MAX };
        int8_t ref_q[] =
            { 64, 127, 95, 127, 106, 127, 111, 127,
            114, 127, 116, 127, 118, 127, 119, 127 };

        int8_t qx_q[batch_size * size];
        float qx_s[batch_size * size / gs];
        QuantizedTensor qx = {.q = qx_q,.s = qx_s };
        quantize2d(&qx, x, cc, ncols);

        for (int i = 0; i < batch_size * size / gs; i++) {
                DOUBLES_EQUAL(ref_sf[i], qx.s[i], TOL);
        }
        MEMCMP_EQUAL(ref_q, qx.q, batch_size * size * sizeof(int8_t));
}
