import java.util.Random;

public class MLToolkit {

    // Constants
    public static final Random RANDOM = new Random();

    /**
     * A method to convert a given double to a binary string
     * @param value the value to be converted
     * @param precision the amount of bits after the dot
     * @return A binary string representation of the given value
     */
    public static String doubleToBinaryString(double value, int precision) {
        // Convert the decimal part
        StringBuilder binary = new StringBuilder();
        // Append sign bit
        if (value < 0) {
            binary.append(1);
            value = -value;
        }
        else binary.insert(0, 0);
        // Separate the fractional and decimal part
        int decimalPart = (int)value;
        double fractionPart = value - decimalPart;
        do {
            binary.append(decimalPart % 2);
            decimalPart /= 2;
        }
        while (decimalPart != 0);
        binary.append(".");
        // Convert the fractional part to a given precision
        for (int i = 0; i < precision; i++) {
            fractionPart *= 2;
            if (fractionPart > 1) {
                binary.append(1);
                fractionPart -= 1;
            }
            else binary.append(0);
        }
        // Return the new binary string
        return binary.toString();
    }

    /**
     * A method to convert a binary string to a double
     * @param binary A binary representation of a double
     * @return The double number computed from the string
     */
    public static double binaryStringToDouble(String binary) {
        // Split the string around the dot
        String[] binSplit = binary.split("\\.");
        double sum = 0.0;
        int sign = 1;
        // Convert the part before the dot
        if (binSplit[0].length() > 0) {
            // Check sign bit
            if (binSplit[0].charAt(0) == '1')
                sign = -1;
            for (int i = 1; i < binSplit[0].length(); i++)
                if (binSplit[0].charAt(i) == '1')
                    sum += Math.pow(2, binSplit[0].length() - i - 1);
        }
        // Convert the part after the dot
        if (binSplit[1].length() > 0)
            for (int i = 0; i < binSplit[1].length(); i++)
                if (binSplit[1].charAt(i) == '1')
                    sum += Math.pow(2, -(i + 1));
        return sum * sign;
    }

    /**
     * A method to take 2 double numbers and breed them to make a new one
     * @param val1 The "father" of the breeding process
     * @param val2 The "mother" of the breeding process
     * @param mutation_chance The chance of a "mutation" occurring (a random change to the "child" value being computed)
     * @return A "child" value made from the "father" and "mother" values
     */
    public static double breedAndMutate(double val1, double val2, double mutation_chance) {
        // Convert both values into bits
        StringBuilder val1Bits = new StringBuilder(doubleToBinaryString(val1, 16));
        StringBuilder val2Bits = new StringBuilder(doubleToBinaryString(val2, 16));
        // Make sure both binary strings are the same length
        while (val1Bits.length() != val2Bits.length()) {
            if (val1Bits.length() < val2Bits.length())
                val1Bits.insert(0, "0");
            else val2Bits.insert(0, "0");
        }
        // Randomly combine the bits into a new string
        StringBuilder newBinaryBuilder = new StringBuilder();
        for (int i = 0; i < val1Bits.length(); i++) {
            char newBit = RANDOM.nextBoolean() ? val1Bits.charAt(i) : val2Bits.charAt(i);
            // Randomly mutate bits
            if (newBit != '.' && RANDOM.nextDouble() < mutation_chance)
                newBit = (char)(-newBit + 97);
            newBinaryBuilder.append(newBit);
        }
        // Convert the new binary string to a double and return it
        return binaryStringToDouble(newBinaryBuilder.toString());
    }

    /**
     * @param input the vector to be sent to the function
     * @return the result of the softmax function
     */
    public static Matrix softmax(Matrix input) {
        Matrix e_input = input.preformOnMatrix(x -> Math.pow(Math.E, x));
        double sum = e_input.sum();
        return e_input.preformOnMatrix(x -> x / sum);
    }

    /**
     * @param input the vector to be sent to the function
     * @return the result of the softmax derivative function
     */
    public static Matrix softmaxDerivative(Matrix input) {
        Matrix softmax = softmax(input);
        for (int i = 0; i < input.getWidth(); i++)
            for (int j = 0; j < input.getHeight(); j++)
                softmax.set(i, j, softmax.get(i, j) * (i == j ? 0 : 1 - softmax.get(i, j)));
        return softmax;
    }

    /**
     * A method to add zero padding around a given input
     * @param input the input to be padded
     * @param inputShape the input shape (Width, Height, Depth)
     * @param padding a array containing (horizontal padding, vertical padding)
     * @return a padded version of the input
     */
    public static Matrix pad(Matrix input, int[] inputShape, int[] padding) {
        // Compute the shape of the output matrix
        int[] outputShape = {
                inputShape[0] + 2 * padding[0],
                inputShape[1] + 2 * padding[1],
                inputShape[2]
        };
        // Compute jumps needed to move between channels
        int input_dim_jump = inputShape[0] * inputShape[1];
        int padded_dim_jump = outputShape[0] * outputShape[1];
        // Initialize result matrix
        Matrix result = new Matrix(1, outputShape[0] * outputShape[1] * outputShape[2]);
        // Fill result matrix
        for (int i = 0; i < inputShape[0]; i++)
            for (int j = 0; j < inputShape[1]; j++)
                for (int k = 0; k < inputShape[2]; k++)
                    result.set(0, (i + padding[0]) + (j + padding[1]) * outputShape[0] + k * padded_dim_jump, input.get(0, i + j * inputShape[0] + k * input_dim_jump));
        // Return the result
        return result;
    }

    /**
     * A method to remove padding from a given input
     * @param input the input to remove the padding from <br>
     * @param outputShape the shape of the input WITHOUT the padding, a array of (width, depth, height) <br>
     * @param padding the padding applied to the input, a array (horizontal padding, vertical padding) <br>
     * @return the input without padding
     */
    public static Matrix removePadding(Matrix input, int[] outputShape, int[] padding) {
        // Compute shape with padding
        int[] inputShape = {
                outputShape[0] + 2 * padding[0],
                outputShape[1] + 2 * padding[1],
                outputShape[2]
        };
        // Build result matrix
        Matrix result = new Matrix(1, outputShape[0] * outputShape[1] * outputShape[2]);
        // Compute area of dimension in output/input to save time later
        int padded_dim_jump = inputShape[0] * inputShape[1];
        int output_dim_jump = outputShape[0] * outputShape[1];
        // For each cell in the output move its corresponding cell from the padded input
        for (int i = 0; i < outputShape[0]; i++)
            for (int j = 0; j < outputShape[1]; j++)
                for (int k = 0; k < outputShape[2]; k++)
                    result.set(0, i + j * outputShape[0] + k * output_dim_jump, input.get(0, (i + padding[0]) + (j + padding[1]) * outputShape[0] + k * padded_dim_jump));
        // Return the result
        return result;
    }

    /**
     * Builds a table used to speed up convolutions with these specified dimensions
     * Method time complexity O(n^6)
     * @param inputShape the shape of the expected input, a array (width, height, depth)
     * @param filterShape the shape of the expected filter, a array (width, height, depth)
     * @param padding the amount of zero padding to be added to the expected input, a array (horizontal padding, vertical padding)
     * @param stride a array containing the amount of steps to take (horizontally after a patch, vertically after a row)
     * @return a table with instructions of preforming a convolution with the specified dimensions and a appropriate filter matrix(see line 14 in source code)
     */
    public static Matrix generateConvTable(int[] inputShape, int[] filterShape, int[] padding, int[] stride) {
        // Compute shape with padding
        int[] paddedShape = {
                inputShape[0] + 2 * padding[0],
                inputShape[1] + 2 * padding[1],
                inputShape[2]
        };
        // Compute output shape
        int[] outputShape = {
                1 + (paddedShape[0] - filterShape[0]) / stride[0],
                1 + (paddedShape[1] - filterShape[1]) / stride[1],
        };
        // Create a table such that each column is all the elements need for a convolution by the specified filter
        Matrix table = new Matrix(outputShape[0] * outputShape[1], filterShape[0] * filterShape[1] * filterShape[2]);
        // Compute area of dimensions to prevent repeat computations
        int paddedDimJump = paddedShape[0] * paddedShape[1];
        int filterDimJump = filterShape[0] * filterShape[1];
        // For each output cell
        for (int j = 0, y = 0, counter = 0; j < outputShape[1]; j++, y += stride[1])
            for (int i = 0, x = 0; i < outputShape[0]; i++, x += stride[0]) {
                // Compute a patch
                for (int l = 0; l < filterShape[0]; l++)
                    for (int m = 0; m < filterShape[1]; m++)
                        for (int n = 0; n < inputShape[2]; n++) {
                            int i_index = (x + l) + (y + m) * paddedShape[0] + n * paddedDimJump;
                            int f_index = l + m * filterShape[0] + (filterShape[2] == 1 ? 0 : n * filterDimJump);
                            table.set(counter, f_index, i_index);
                        }
                // Increment the convolution counter
                counter++;
            }
        // Return the new table
        return table;
    }

    /**
     * A method to build a table for quick pooling
     * @param inputShape the shape of the expected input
     * @param poolWidth the width of the pool
     * @param poolHeight the height of the pool
     * @return a pooling table with regard to the dimensions provided
     */
    public static Matrix generatePoolTable(int[] inputShape, int poolWidth, int poolHeight) {
        // Compute output shape
        int[] outputShape = new int[] {
                1 + (inputShape[0] - poolWidth) / poolWidth,
                1 + (inputShape[1] - poolHeight) / poolHeight,
                inputShape[2]
        };
        // Initialize the pool table
        Matrix output = new Matrix(outputShape[0] * outputShape[1] * outputShape[2], poolWidth * poolHeight);
        // Compute area of dimensions to prevent repeat computation later
        int output_dim_jump = outputShape[0] * outputShape[1];
        int input_dim_jump = inputShape[0] * inputShape[1];
        // Preform a fake pooling and build the table
        for (int k = 0, z = 0; k < outputShape[1]; k++, z++)
            for (int j = 0, y = 0; j < outputShape[1]; j++, y += poolHeight)
                for (int i = 0, x = 0; i < outputShape[0]; i++, x += poolWidth) {
                    int index = i + j * outputShape[0] + k * output_dim_jump;
                    for (int l = 0; l < poolWidth; l++)
                        for (int m = 0; m < poolHeight; m++)
                            output.set(index, l + m * poolWidth, (x + l) + (y + m) * inputShape[0] + z * input_dim_jump);
                }
        // Return the table
        return output;
    }

}
