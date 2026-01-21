package org.cobweb.cobweb2.impl.ai;

import java.util.Arrays;
import java.util.Random;

/**
 * Helper class for Matrix and Probability operations required by Active
 * Inference.
 * Implements essential Tensor/Matrix math without external dependencies.
 */
public class Matrices {

    private static final double EPSILON = 1e-12;

    /**
     * Normalizes a vector to sum to 1.0 (Probability Distribution).
     */
    public static double[] normalize(double[] array) {
        double sum = 0.0;
        for (double v : array)
            sum += v;
        if (sum == 0)
            return createUniform(array.length);

        double[] result = new double[array.length];
        for (int i = 0; i < array.length; i++) {
            result[i] = array[i] / sum;
        }
        return result;
    }

    /**
     * Creates a uniform distribution vector of given size.
     */
    public static double[] createUniform(int size) {
        double[] result = new double[size];
        double val = 1.0 / size;
        Arrays.fill(result, val);
        return result;
    }

    /**
     * Softmax function: exp(x) / sum(exp(x))
     */
    public static double[] softmax(double[] input) {
        double max = -Double.MAX_VALUE;
        for (double v : input)
            max = Math.max(max, v); // Stability shift

        double sum = 0.0;
        double[] result = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            result[i] = Math.exp(input[i] - max);
            sum += result[i];
        }
        for (int i = 0; i < input.length; i++) {
            result[i] /= sum;
        }
        return result;
    }

    /**
     * Natural Logarithm of a matrix/vector. Handles 0 by using EPSILON.
     */
    public static double[] log(double[] input) {
        double[] result = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            result[i] = Math.log(Math.max(input[i], EPSILON));
        }
        return result;
    }

    public static double[][] log(double[][] input) {
        double[][] result = new double[input.length][input[0].length];
        for (int i = 0; i < input.length; i++) {
            result[i] = log(input[i]);
        }
        return result;
    }

    /**
     * Dot product of two vectors.
     */
    public static double dot(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++)
            sum += a[i] * b[i];
        return sum;
    }

    /**
     * Multiplies a Matrix by a Vector (M * v).
     * 
     * @param M Matrix [rows][cols]
     * @param v Vector [cols]
     * @return Result Vector [rows]
     */
    public static double[] multiply(double[][] M, double[] v) {
        int rows = M.length;
        int cols = M[0].length;
        if (v.length != cols)
            throw new IllegalArgumentException(
                    "Dimension mismatch: Matrix cols " + cols + " != Vector len " + v.length);

        double[] result = new double[rows];
        for (int i = 0; i < rows; i++) {
            result[i] = 0.0;
            for (int j = 0; j < cols; j++) {
                result[i] += M[i][j] * v[j];
            }
        }
        return result;
    }

    /**
     * Multiplies a Vector by a Matrix (v^T * M).
     * Typically used for likelihood mapping inverted.
     * 
     * @param v Vector [rows]
     * @param M Matrix [rows][cols]
     * @return Result Vector [cols]
     */
    public static double[] multiplyIsTransposed(double[] v, double[][] M) {
        int rows = M.length;
        int cols = M[0].length;
        if (v.length != rows)
            throw new IllegalArgumentException("Dimension mismatch");

        double[] result = new double[cols];
        for (int j = 0; j < cols; j++) {
            result[j] = 0.0;
            for (int i = 0; i < rows; i++) {
                result[j] += v[i] * M[i][j];
            }
        }
        return result;
    }

    /**
     * Element-wise multiplication.
     */
    public static double[] multiplyElementwise(double[] a, double[] b) {
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++)
            result[i] = a[i] * b[i];
        return result;
    }

    /**
     * Entropy of a distribution H(P) = - sum P(x) ln P(x)
     */
    public static double entropy(double[] p) {
        double h = 0.0;
        for (double val : p) {
            if (val > EPSILON) {
                h -= val * Math.log(val);
            }
        }
        return h;
    }

    /**
     * Calculates the expected value of a Dirichlet distribution.
     * E[x_i] = alpha_i / sum(alpha)
     */
    public static double[] dirichletExpectation(double[] alpha) {
        return normalize(alpha); // For mean.
        // Note: For variational inference, we often use digamma(alpha) -
        // digamma(sum(alpha)) for ln A expectations.
        // But for this simulation, using the mean of the distribution (normalized
        // counts) is a sufficient approximation for A/B matrices.
    }

    public static double[][] dirichletExpectation(double[][] alpha) {
        double[][] result = new double[alpha.length][];
        for (int i = 0; i < alpha.length; i++) {
            result[i] = normalize(alpha[i]);
        }
        return result;
    }

    /**
     * Adds scalar to vector elements (for pseudocounts).
     */
    public static double[] addScalar(double[] v, double scalar) {
        double[] res = new double[v.length];
        for (int i = 0; i < v.length; i++)
            res[i] = v[i] + scalar;
        return res;
    }

    /**
     * Adds scalar to matrix elements.
     */
    public static double[][] addScalar(double[][] m, double scalar) {
        double[][] res = new double[m.length][m[0].length];
        for (int i = 0; i < m.length; i++) {
            res[i] = addScalar(m[i], scalar);
        }
        return res;
    }

    /**
     * Sample from a categorical distribution.
     */
    public static int sample(double[] probs, Random rng) {
        double r = rng.nextDouble();
        double cum = 0.0;
        for (int i = 0; i < probs.length; i++) {
            cum += probs[i];
            if (r < cum)
                return i;
        }
        return probs.length - 1;
    }
}
