#include "../HandmadeTest.h"

TEST(RotorOps, Inverse)
{
    HMM_Rotor r1 = HMM_R(1.0f, 2.0f, 3.0f, 4.0f);
    HMM_Rotor inverse = HMM_InvR(r1);

    HMM_Rotor result = HMM_MulR(r1, inverse);

    EXPECT_FLOAT_EQ(result.YZ, 0.0f);
    EXPECT_FLOAT_EQ(result.XZ, 0.0f);
    EXPECT_FLOAT_EQ(result.XY, 0.0f);
    EXPECT_FLOAT_EQ(result.Scalar, 1.0f);
}

TEST(RotorOps, Dot)
{
    HMM_Rotor r1 = HMM_R(1.0f, 2.0f, 3.0f, 4.0f);
    HMM_Rotor r2 = HMM_R(5.0f, 6.0f, 7.0f, 8.0f);

    {
        float result = HMM_DotR(r1, r2);
        EXPECT_FLOAT_EQ(result, 70.0f);
    }
#if HANDMADE_MATH__USE_C11_GENERICS || defined(__cplusplus)
    {
        float result = HMM_Dot(r1, r2);
        EXPECT_FLOAT_EQ(result, 70.0f);
    }
#endif
}

TEST(RotorOps, Normalize)
{
    HMM_Rotor r = HMM_R(1.0f, 2.0f, 3.0f, 4.0f);

    {
        HMM_Rotor result = HMM_NormR(r);
        EXPECT_NEAR(result.YZ, 0.1825741858f, 0.001f);
        EXPECT_NEAR(result.XZ, 0.3651483717f, 0.001f);
        EXPECT_NEAR(result.XY, 0.5477225575f, 0.001f);
        EXPECT_NEAR(result.Scalar, 0.7302967433f, 0.001f);
    }
#if HANDMADE_MATH__USE_C11_GENERICS || defined(__cplusplus)
    {
        HMM_Rotor result = HMM_Norm(r);
        EXPECT_NEAR(result.YZ, 0.1825741858f, 0.001f);
        EXPECT_NEAR(result.XZ, 0.3651483717f, 0.001f);
        EXPECT_NEAR(result.XY, 0.5477225575f, 0.001f);
        EXPECT_NEAR(result.Scalar, 0.7302967433f, 0.001f);
    }
#endif
}

TEST(RotorOps, NLerp)
{
    HMM_Rotor from = HMM_R(0.0f, 0.0f, 0.0f, 1.0f);
    HMM_Rotor to = HMM_R(0.5f, 0.5f, -0.5f, 0.5f);

    HMM_Rotor result = HMM_NLerp(from, 0.5f, to);
    EXPECT_NEAR(result.YZ, 0.28867513f, 0.001f);
    EXPECT_NEAR(result.XZ, 0.28867513f, 0.001f);
    EXPECT_NEAR(result.XY, -0.28867513f, 0.001f);
    EXPECT_NEAR(result.Scalar, 0.86602540f, 0.001f);
}

TEST(RotorOps, SLerp)
{
    HMM_Rotor from = HMM_R(0.0f, 0.0f, 0.0f, 1.0f);
    HMM_Rotor to = HMM_R(0.5f, 0.5f, -0.5f, 0.5f);

    {
        HMM_Rotor result = HMM_SLerp(from, 0.0f, to);
        EXPECT_NEAR(result.YZ, 0.0f, 0.001f);
        EXPECT_NEAR(result.XZ, 0.0f, 0.001f);
        EXPECT_NEAR(result.XY, 0.0f, 0.001f);
        EXPECT_NEAR(result.Scalar, 1.0, 0.001f);
    }
    {
        HMM_Rotor result = HMM_SLerp(from, 0.25f, to);
        EXPECT_NEAR(result.YZ, 0.149429246f, 0.001f);
        EXPECT_NEAR(result.XZ, 0.149429246f, 0.001f);
        EXPECT_NEAR(result.XY, -0.149429246f, 0.001f);
        EXPECT_NEAR(result.Scalar, 0.965925812f, 0.001f);
    }
    {
        HMM_Rotor result = HMM_SLerp(from, 0.5f, to);
        EXPECT_NEAR(result.YZ, 0.28867513f, 0.001f);
        EXPECT_NEAR(result.XZ, 0.28867513f, 0.001f);
        EXPECT_NEAR(result.XY, -0.28867513f, 0.001f);
        EXPECT_NEAR(result.Scalar, 0.86602540f, 0.001f);
    }
    {
        HMM_Rotor result = HMM_SLerp(from, 0.75f, to);
        EXPECT_NEAR(result.YZ, 0.40824830f, 0.001f);
        EXPECT_NEAR(result.XZ, 0.40824830f, 0.001f);
        EXPECT_NEAR(result.XY, -0.40824830f, 0.001f);
        EXPECT_NEAR(result.Scalar, 0.70710676f, 0.001f);
    }
    {
        HMM_Rotor result = HMM_SLerp(from, 1.0f, to);
        EXPECT_NEAR(result.YZ, 0.5f, 0.001f);
        EXPECT_NEAR(result.XZ, 0.5f, 0.001f);
        EXPECT_NEAR(result.XY, -0.5f, 0.001f);
        EXPECT_NEAR(result.Scalar, 0.5f, 0.001f);
    }
}

TEST(RotorOps, RotorToMat4)
{
    const float abs_error = 0.001f;

    HMM_Rotor rot = HMM_R(0.707107f, 0.0f, 0.0f, 0.707107f);

    HMM_Mat4 result = HMM_RToM4(rot);

    EXPECT_NEAR(result.Elements[0][0], 1.0f, abs_error);
    EXPECT_NEAR(result.Elements[0][1], 0.0f, abs_error);
    EXPECT_NEAR(result.Elements[0][2], 0.0f, abs_error);
    EXPECT_NEAR(result.Elements[0][3], 0.0f, abs_error);

    EXPECT_NEAR(result.Elements[1][0], 0.0f, abs_error);
    EXPECT_NEAR(result.Elements[1][1], 0.0f, abs_error);
    EXPECT_NEAR(result.Elements[1][2], 1.0f, abs_error);
    EXPECT_NEAR(result.Elements[1][3], 0.0f, abs_error);

    EXPECT_NEAR(result.Elements[2][0], 0.0f, abs_error);
    EXPECT_NEAR(result.Elements[2][1], -1.0f, abs_error);
    EXPECT_NEAR(result.Elements[2][2], 0.0f, abs_error);
    EXPECT_NEAR(result.Elements[2][3], 0.0f, abs_error);

    EXPECT_NEAR(result.Elements[3][0], 0.0f, abs_error);
    EXPECT_NEAR(result.Elements[3][1], 0.0f, abs_error);
    EXPECT_NEAR(result.Elements[3][2], 0.0f, abs_error);
    EXPECT_NEAR(result.Elements[3][3], 1.0f, abs_error);
}

TEST(RotorOps, Mat4ToRotor)
{
    const float abs_error = 0.0001f;

    // Rotate 90 degrees on the X axis
    {
        HMM_Mat4 m = HMM_Rotate_RH(HMM_AngleDeg(90.0f), HMM_V3(1, 0, 0));
        HMM_Rotor result = HMM_M4ToR_RH(m);

        float cosf = 0.707107f; // cos(90/2 degrees)
        float sinf = 0.707107f; // sin(90/2 degrees)

        EXPECT_NEAR(result.YZ, sinf, abs_error);
        EXPECT_NEAR(result.XZ, 0.0f, abs_error);
        EXPECT_NEAR(result.XY, 0.0f, abs_error);
        EXPECT_NEAR(result.Scalar, cosf, abs_error);
    }

    // Rotate 90 degrees on the Y axis (axis not normalized, just for fun)
    {
        HMM_Mat4 m = HMM_Rotate_RH(HMM_AngleDeg(90.0f), HMM_V3(0, 2, 0));
        HMM_Rotor result = HMM_M4ToR_RH(m);

        float cosf = 0.707107f; // cos(90/2 degrees)
        float sinf = 0.707107f; // sin(90/2 degrees)

        EXPECT_NEAR(result.YZ, 0.0f, abs_error);
        EXPECT_NEAR(result.XZ, sinf, abs_error);
        EXPECT_NEAR(result.XY, 0.0f, abs_error);
        EXPECT_NEAR(result.Scalar, cosf, abs_error);
    }

    // Rotate 90 degrees on the Z axis
    {
        HMM_Mat4 m = HMM_Rotate_RH(HMM_AngleDeg(90.0f), HMM_V3(0, 0, 1));
        HMM_Rotor result = HMM_M4ToR_RH(m);

        float cosf = 0.707107f; // cos(90/2 degrees)
        float sinf = 0.707107f; // sin(90/2 degrees)

        EXPECT_NEAR(result.YZ, 0.0f, abs_error);
        EXPECT_NEAR(result.XZ, 0.0f, abs_error);
        EXPECT_NEAR(result.XY, sinf, abs_error);
        EXPECT_NEAR(result.Scalar, cosf, abs_error);
    }

    // Rotate 45 degrees on the X axis (this hits case 4)
    {
        HMM_Mat4 m = HMM_Rotate_RH(HMM_AngleDeg(45.0f), HMM_V3(1, 0, 0));
        HMM_Rotor result = HMM_M4ToR_RH(m);

        float cosf = 0.9238795325f; // cos(90/2 degrees)
        float sinf = 0.3826834324f; // sin(90/2 degrees)

        EXPECT_NEAR(result.YZ, sinf, abs_error);
        EXPECT_NEAR(result.XZ, 0.0f, abs_error);
        EXPECT_NEAR(result.XY, 0.0f, abs_error);
        EXPECT_NEAR(result.Scalar, cosf, abs_error);
    }

    /* NOTE(lcf): Left-handed cases. Since both Rotate and M4ToR are LH results should be
        the same with no changes to input. */
    // Rotate 90 degrees on the X axis
    {
        HMM_Mat4 m = HMM_Rotate_LH(HMM_AngleDeg(90.0f), HMM_V3(1, 0, 0));
        HMM_Rotor result = HMM_M4ToR_LH(m);

        float cosf = 0.707107f; // cos(90/2 degrees)
        float sinf = 0.707107f; // sin(90/2 degrees)

        EXPECT_NEAR(result.YZ, sinf, abs_error);
        EXPECT_NEAR(result.XZ, 0.0f, abs_error);
        EXPECT_NEAR(result.XY, 0.0f, abs_error);
        EXPECT_NEAR(result.Scalar, cosf, abs_error);
    }

    // Rotate 90 degrees on the Y axis (axis not normalized, just for fun)
    {
        HMM_Mat4 m = HMM_Rotate_LH(HMM_AngleDeg(90.0f), HMM_V3(0, 2, 0));
        HMM_Rotor result = HMM_M4ToR_LH(m);

        float cosf = 0.707107f; // cos(90/2 degrees)
        float sinf = 0.707107f; // sin(90/2 degrees)

        EXPECT_NEAR(result.YZ, 0.0f, abs_error);
        EXPECT_NEAR(result.XZ, sinf, abs_error);
        EXPECT_NEAR(result.XY, 0.0f, abs_error);
        EXPECT_NEAR(result.Scalar, cosf, abs_error);
    }

    // Rotate 90 degrees on the Z axis
    {
        HMM_Mat4 m = HMM_Rotate_LH(HMM_AngleDeg(90.0f), HMM_V3(0, 0, 1));
        HMM_Rotor result = HMM_M4ToR_LH(m);

        float cosf = 0.707107f; // cos(90/2 degrees)
        float sinf = 0.707107f; // sin(90/2 degrees)

        EXPECT_NEAR(result.YZ, 0.0f, abs_error);
        EXPECT_NEAR(result.XZ, 0.0f, abs_error);
        EXPECT_NEAR(result.XY, sinf, abs_error);
        EXPECT_NEAR(result.Scalar, cosf, abs_error);
    }

    // Rotate 45 degrees on the X axis (this hits case 4)
    {
        HMM_Mat4 m = HMM_Rotate_LH(HMM_AngleDeg(45.0f), HMM_V3(1, 0, 0));
        HMM_Rotor result = HMM_M4ToR_LH(m);

        float cosf = 0.9238795325f; // cos(90/2 degrees)
        float sinf = 0.3826834324f; // sin(90/2 degrees)

        EXPECT_NEAR(result.YZ, sinf, abs_error);
        EXPECT_NEAR(result.XZ, 0.0f, abs_error);
        EXPECT_NEAR(result.XY, 0.0f, abs_error);
        EXPECT_NEAR(result.Scalar, cosf, abs_error);
    }
}

TEST(RotorOps, RotateVectorAxisAngle)
{
    {
        HMM_Vec3 axis = HMM_V3(0.0f, 1.0f, 0.0f);
        float angle = HMM_AngleTurn(1.0/4);
        HMM_Vec3 result = HMM_RotateV3AxisAngle_LH(HMM_V3(1.0f, 0.0f, 0.0f), axis, angle);
        EXPECT_NEAR(result.X, 0.0f, 0.001f);
        EXPECT_NEAR(result.Y, 0.0f, 0.001f);
        EXPECT_NEAR(result.Z, 1.0f, 0.001f);
    }
    {
        HMM_Vec3 axis = HMM_V3(1.0f, 0.0f, 0.0f);
        float angle = HMM_AngleTurn(1.0/8);
        HMM_Vec3 result = HMM_RotateV3AxisAngle_RH(HMM_V3(0.0f, 0.0f, 1.0f), axis, angle);
        EXPECT_NEAR(result.X, 0.0f, 0.001f);
        EXPECT_NEAR(result.Y, -0.707170f, 0.001f);
        EXPECT_NEAR(result.Z, 0.707170f, 0.001f);
    }
}

TEST(RotorOps, RotorFromPairs)
{
    {
        HMM_Vec3 n1 = HMM_V3(0.0f, 1.0f, 0.0f);
        HMM_Vec3 n2 = HMM_V3(0.0f, 0.0f, 1.0f);
        HMM_Rotor r = HMM_RFromNormPair(n1, n2);
        HMM_Vec3 result = HMM_RotateV3R(n1, r);
        EXPECT_NEAR(result.X, n2.X, 0.001f);
        EXPECT_NEAR(result.Y, n2.Y, 0.001f);
        EXPECT_NEAR(result.Z, n2.Z, 0.001f);
    }
    {
        HMM_Vec3 v1 = HMM_V3(2.0f, 2.0f, 2.0f);
        HMM_Vec3 v2 = HMM_V3(3.0f, 0.0f, 0.0f);
        HMM_Rotor r = HMM_RFromVecPair(v1, v2);
        HMM_Vec3 result = HMM_RotateV3R(HMM_V3(0.0f, 1.0f, 0.0f), r);
        EXPECT_NEAR(result.X, 0.577350, 0.001f);
        EXPECT_NEAR(result.Y, 0.788675, 0.001f);
        EXPECT_NEAR(result.Z, -0.211325, 0.001f);
    }
}
