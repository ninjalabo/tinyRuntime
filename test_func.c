#include "CppUTest/TestHarness.h"
#include "func.h"

#define TOL 0.00001
#define eps 0.00001f
#define Q_MAX 127.0f

TEST_GROUP(TestGroup)
{
        void setup() {
        }
        void teardown() {
        }
};

TEST(TestGroup, Linear)
{
        int in = 2, out = 2, offset = 1;
        int gs_w = 2, gs_b = 2;

        float x[] = { 1.0f, 2.0f };
        float p[] =
            { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f };
        LinearConfig lc = { in, out, offset, 1 };

        int8_t qx_q[] = { 64, 127 };
        float qx_sf[] = { 2.0f / Q_MAX };
        QuantizedTensor qx = { .q = qx_q, .s = qx_sf };
        int8_t pq[] =  { 0, 64, 127, 95, 127, 64, 127};
        float sf[] = { 0.0f, 2.0f / Q_MAX, 4.0f / Q_MAX, 2.0f / Q_MAX };
        LinearConfigQ lcq = { in, out, offset, offset, gs_w, gs_b };

        float xout[out];
        linear(xout, x, p, lc);

        float xout_q[out];
        linear_q(xout_q, &qx, pq, sf, lcq);

        DOUBLES_EQUAL(6.0, xout[0], TOL);
        DOUBLES_EQUAL(13.0, xout[1], TOL);
        DOUBLES_EQUAL(6.02368404737, xout_q[0], TOL);
        DOUBLES_EQUAL(13.01568603137, xout_q[1], TOL);
}

TEST(TestGroup, Im2col)
{
        int h = 2, w = 2, hq = 2, wq = 2;
        int ksize = 3, stride = 1, pad = 1, ic = 2;
        int col_size = ksize * ksize * ic * h * w;

        float im[] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
        ConvConfig cc = { ksize, stride, pad, ic, 0, 0, 0 }; // 0 doesn't affect
        ConvConfigQ cc_q = { ksize, stride, pad, ic, 0, 0, 0, 0, 0 };

        float col[col_size];
        im2col(col, im, cc, &h, &w);

        float col_q[col_size];
        im2col_q(col_q, im, cc_q, &hq, &wq);

        float ref[] =
            { 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 2.0f,
            0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 1.0f, 0.0f, 3.0f,
            1.0f, 2.0f, 3.0f, 4.0f, 2.0f, 0.0f, 4.0f, 0.0f,
            0.0f, 3.0f, 0.0f, 0.0f, 3.0f, 4.0f, 0.0f, 0.0f,
            4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 5.0f,
            0.0f, 0.0f, 5.0f, 6.0f, 0.0f, 0.0f, 6.0f, 0.0f,
            0.0f, 5.0f, 0.0f, 7.0f, 5.0f, 6.0f, 7.0f, 8.0f,
            6.0f, 0.0f, 8.0f, 0.0f, 0.0f, 7.0f, 0.0f, 0.0f,
            7.0f, 8.0f, 0.0f, 0.0f, 8.0f, 0.0f, 0.0f, 0.0f };

        MEMCMP_EQUAL(ref, col, col_size * sizeof(float));
        MEMCMP_EQUAL(ref, col_q, col_size * sizeof(float));
}

TEST(TestGroup, Conv)
{
        int h = 1, w = 1;
        int ksize = 2, stride = 1, pad = 1, ic = 1, oc = 2;
        int offset = 1, bias = 1, gs_w = 2, gs_b = 2;

        float x[] = { 1.0f, 2.0f, 3.0f, 4.0f };
        float p[] =
            { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 1.0f,
            2.0f };
        ConvConfig cc = { ksize, stride, pad, ic, oc, offset, bias };

        float qx_sf[] = { 2.0f / Q_MAX, 4.0f / Q_MAX };
        int8_t qx_q[] = { 64, 127, 95, 127 };
        QuantizedTensor qx = { .q = qx_q, .s = qx_sf };
        int8_t pq[] =  { 0, 64, 127, 95, 127, 106, 127, 111, 127, 64, 127};
        float sf[] =
            { 0.0f, 2.0f / Q_MAX, 4.0f / Q_MAX, 6.0f / Q_MAX, 8.0f / Q_MAX,
            2.0f / Q_MAX };
        ConvConfigQ cc_q =
            { ksize, stride, pad, ic, oc, offset, offset, gs_w, gs_b };

        float xout[oc * h * w];
        conv(xout, x, p, cc, h, w);

        float xout_q[oc * h * w];
        conv_q(xout_q, &qx, pq, sf, cc_q, h, w);

        DOUBLES_EQUAL(31.0, xout[0], TOL);
        DOUBLES_EQUAL(72.0, xout[1], TOL);
        DOUBLES_EQUAL(30.976501953, xout_q[0], TOL);
        DOUBLES_EQUAL(71.9686279373, xout_q[1], TOL);
}

TEST(TestGroup, Batchnorm)
{
        int h = 2, w = 1, ic = 2, offset = 1;

        float x[] = { 1.0f, 2.0f, 3.0f, 4.0f };
        float p[] = { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 1.0f, 2.0f };
        BnConfig bc = { ic, offset };

        float xout[ic * h * w];
        batchnorm(xout, x, p, bc, h, w);

        float ref[] =
            { 3.8999955f, 4.8999905001f, 7.9597880752f, 9.373998102f };

        for (int i = 0; i < ic * h * w; i++) {
                DOUBLES_EQUAL(ref[i], xout[i], TOL);
        }
}

TEST(TestGroup, Pool)
{
        int nch = 2, h_before = 1, w_before = 1;
        int ksize = 2, stride = 1, pad = 1;
        int h_after = 2, w_after = 2;
        int out_size = nch * h_after * w_after;

        float x[] = { 1.0f, -1.0f };
        float refs[][8] = {
            { 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f },
            { 0.25f, 0.25f, 0.25f, 0.25f, -0.25f, -0.25f, -0.25f, -0.25f }
        };
        int h, w;
        float xout[out_size];
        void (*pool_functions[])(float*, float*, int*, int*, int, int, int,
                                 int) = { maxpool, avgpool };

        for (int i = 0; i < 2; i++) {
                h = h_before;
                w = w_before;

                pool_functions[i](xout, x, &h, &w, nch, ksize, stride, pad);
                DOUBLES_EQUAL(h_after, h, TOL);
                DOUBLES_EQUAL(w_after, w, TOL);

                for (int j = 0; j < nch * h * w; j++) {
                        DOUBLES_EQUAL(refs[i][j], xout[j], TOL);
                }
        }
}

TEST(TestGroup, Matadd)
{
        float x[] = { 1.0f, 2.0f };
        float y[] = { 3.0f, 4.0f };
        matadd(x, y, 2);
        DOUBLES_EQUAL(4.0, x[0], TOL);
        DOUBLES_EQUAL(6.0, x[1], TOL);
}

TEST(TestGroup, Relu)
{
        float x[] = { 1.0f, -1.0f };
        relu(x, 2);
        DOUBLES_EQUAL(1.0, x[0], TOL);
        DOUBLES_EQUAL(0.0, x[1], TOL);
}

TEST(TestGroup, Softmax)
{
        float x[] = { 1.0f, 2.0f, 3.0f };
        softmax(x, 3);
        DOUBLES_EQUAL(1.0, x[0] + x[1] + x[2], TOL);
}

TEST(TestGroup, Quantize)
{
        int size = 4, gs = 2;
        float x[] = { 1.0f, 2.0f, 3.0f, 4.0f };
        float ref_sf[] = { 2.0f / Q_MAX, 4.0f / Q_MAX };
        int8_t ref_q[] = { 64, 127, 95, 127 };
        QuantizedTensor ref = { .q = ref_q, .s = ref_sf };

        int8_t qx_q[size];
        float qx_s[size / gs];
        QuantizedTensor qx = { .q = qx_q, .s = qx_s };
        quantize(&qx, x, size, gs);

        for (int i = 0; i < size / gs; i++) {
                DOUBLES_EQUAL(ref.s[i], qx.s[i], TOL);
        }
        MEMCMP_EQUAL(ref.q, qx.q, size * sizeof(int8_t));
}

TEST(TestGroup, Quantize2d)
{
        int ksize = 2, ic = 1, gs = 2, ncols = 2, nrows = ksize * ksize * ic;
        int size = nrows * ncols;

        float x[] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
        ConvConfigQ cc = { ksize, 0, 0, ic, 0, 0, 0, gs, 0 };
        float ref_sf[] =
            { 3.0f / Q_MAX, 7.0f / Q_MAX, 4.0f / Q_MAX, 8.0f / Q_MAX};
        int8_t ref_q[] = { 42, 64, 127, 127, 91, 95, 127, 127 };
        QuantizedTensor ref = { .q = ref_q, .s = ref_sf };

        int8_t qx_q[size];
        float qx_s[size / gs];
        QuantizedTensor qx = { .q = qx_q, .s = qx_s };
        quantize2d(&qx, x, cc, ncols);

        for (int i = 0; i < size / gs; i++) {
                DOUBLES_EQUAL(ref.s[i], qx.s[i], TOL);
        }
        MEMCMP_EQUAL(ref.q, qx.q, size * sizeof(int8_t));
}