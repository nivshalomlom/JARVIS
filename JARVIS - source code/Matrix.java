import java.util.function.Function;

public class Matrix {

    // Class specific variables
    private final double[][] data;
    private final int width;
    private final int height;

    /**
     * A constructor to build a empty matrix
     * @param width the matrix's width
     * @param height the matrix's height
     */
    public Matrix(int width, int height) {
        this.data = new double[width][height];
        this.width = width;
        this.height = height;
    }

    /**
     * A method to convert a 1 dimensional array into a vertical vector
     * @param values the array to be converted
     * @return a matrix representing a vertical vector storing the values from the array
     */
    public static Matrix makeVerticalVector(double ...values) {
        Matrix matrix = new Matrix(1, values.length);
        matrix.data[0] = values.clone();
        return matrix;
    }

    /**
     * A method to convert a 1 dimensional array into a horizontal vector
     * @param values the array to be converted
     * @return a matrix representing a horizontal vector storing the values from the array
     */
    public static Matrix makeHorizontalVector(double ...values) {
        Matrix matrix = new Matrix(values.length, 1);
        for (int i = 0; i < matrix.getWidth(); i++)
            matrix.set(i, 1, values[i]);
        return matrix;
    }

    /**
     * @return the width of the matrix
     */
    public int getWidth() {
        return width;
    }

    /**
     * @return the height of the matrix
     */
    public int getHeight() {
        return height;
    }

    /**
     * A method to set the (i,j) value in the matrix to a specified number
     * @param i the column index
     * @param j the row index
     * @param value the number to put in the (i, j) position
     */
    public void set(int i, int j, double value) {
        this.data[i][j] = value;
    }

    /**
     * A method to get the (i, j) value from the matrix
     * @param i the column index
     * @param j the row index
     * @return the value in the (i, j) position in the matrix
     */
    public double get(int i, int j) {
        return this.data[i][j];
    }

    /**
     * A method to get the i'th column
     * @param i the column index
     * @return the i'th column
     */
    public double[] getColumn(int i) {
        if (this.getWidth() <= i)
            throw new IndexOutOfBoundsException();
        return this.data[i];
    }

    /**
     * A method to preform the dot operation on this matrix and a given other matrix
     * @param matrix the other matrix for the dot operation
     * @return a new matrix that is the result of the dot operation
     * @throws Exception if the matrices size are invalid for preforming the dot operation (if matrix 1 width is different then matrix 2 height)
     */
    public Matrix dot(Matrix matrix) throws Exception {
        // Checking the other matrix is valid for multiplication
        if (this.getWidth() != matrix.getHeight())
            throw new Exception("The other matrix needs to have a height of " + this.getWidth() + "!");
        // Building result of the dot operation
        Matrix newMatrix = new Matrix(matrix.getWidth(), this.getHeight());
        for (int i = 0; i < newMatrix.getWidth(); i++)
            for (int j = 0; j < newMatrix.getHeight(); j++) {
                // Multiplying the i column vector with the j row vector
                double sum = 0;
                for (int k = 0; k < this.getWidth(); k++)
                    sum += matrix.get(i, k) * this.get(k, j);
                newMatrix.set(i, j, sum);
            }
        return newMatrix;
    }

    /**
     * Adds this matrix and a given one to a new matrix
     * @param matrix the matrix to be added to this one
     * @return a new matrix that is the result of the addition operation
     * @throws Exception if the matrices are different sizes
     */
    public Matrix addition(Matrix matrix) throws Exception {
        // Checking the other matrix is the same size as this
        if (this.getWidth() != matrix.getWidth() || this.getHeight() != matrix.getHeight())
            throw new Exception("Both matrices need to be the same size!");
        // Adding the two matrices together
        Matrix newMatrix = new Matrix(this.getWidth(), this.getHeight());
        for (int i = 0; i < newMatrix.getWidth(); i++)
            for (int j = 0; j < newMatrix.getHeight(); j++)
                newMatrix.set(i, j, this.get(i, j) + matrix.get(i, j));
        return newMatrix;
    }

    /**
     * Subtracts a given matrix from this one
     * @param matrix the matrix to be subtracted from this one
     * @return a new matrix that is the result of the subtraction operation
     * @throws Exception if the matrices are different sizes
     */
    public Matrix subtraction(Matrix matrix) throws Exception {
        // Checking the other matrix is the same size as this
        if (this.getWidth() != matrix.getWidth() || this.getHeight() != matrix.getHeight())
            throw new Exception("Both matrices need to be the same size!");
        // Subtracting the other matrix from this one
        Matrix newMatrix = new Matrix(this.getWidth(), this.getHeight());
        for (int i = 0; i < newMatrix.getWidth(); i++)
            for (int j = 0; j < newMatrix.getHeight(); j++)
                newMatrix.set(i, j, this.get(i, j) - matrix.get(i, j));
        return newMatrix;
    }

    /**
     * Multiples this matrix and a given one element wise to a new matrix
     * @param matrix the matrix to be multiplied to this one
     * @return a new matrix that is the result of the multiplication operation
     * @throws Exception if the matrices are different sizes
     */
    public Matrix multiplicationElementWise(Matrix matrix) throws Exception {
        // Checking the other matrix is the same size as this
        if (this.getWidth() != matrix.getWidth() || this.getHeight() != matrix.getHeight())
            throw new Exception("Both matrices need to be the same size!");
        // Multiplying the two matrices
        Matrix newMatrix = new Matrix(this.getWidth(), this.getHeight());
        for (int i = 0; i < newMatrix.getWidth(); i++)
            for (int j = 0; j < newMatrix.getHeight(); j++)
                newMatrix.set(i, j, this.get(i, j) * matrix.get(i, j));
        return newMatrix;
    }

    /**
     * Divides this matrix and a given one element wise to a new matrix
     * @param matrix the matrix to be used for divided
     * @return a new matrix that is the result of the division operation
     * @throws Exception if the matrices are different sizes
     */
    public Matrix divisionElementWise(Matrix matrix) throws Exception {
        // Checking the other matrix is the same size as this
        if (this.getWidth() != matrix.getWidth() || this.getHeight() != matrix.getHeight())
            throw new Exception("Both matrices need to be the same size!");
        // Multiplying the two matrices
        Matrix newMatrix = new Matrix(this.getWidth(), this.getHeight());
        for (int i = 0; i < newMatrix.getWidth(); i++)
            for (int j = 0; j < newMatrix.getHeight(); j++)
                newMatrix.set(i, j, this.get(i, j) / matrix.get(i, j));
        return newMatrix;
    }

    /**
     * Transposes this matrix
     * @return a new transposed matrix
     */
    public Matrix transpose() {
        Matrix newMatrix = new Matrix(this.getHeight(), this.getWidth());
        for (int i = 0; i < this.getWidth(); i++)
            for (int j = 0; j < this.getHeight(); j++)
                newMatrix.set(j, i, this.get(i, j));
        return newMatrix;
    }

    /**
     * Adds a given value to every number in the matrix
     * @param value the number to be added
     * @return a new matrix that is the result of the operation
     */
    public Matrix addition(double value) {
        return this.preformOnMatrix(x -> x + value);
    }

    /**
     * Subtracts a given value to every number in the matrix
     * @param value the number to be subtracted
     * @return a new matrix that is the result of the operation
     */
    public Matrix subtraction(double value) {
        return this.preformOnMatrix(x -> x - value);
    }

    /**
     * Multiplies every number in the matrix by a given value
     * @param value the number to be used for multiplication
     * @return a new matrix that is the result of the operation
     */
    public Matrix multiply(double value) {
        return this.preformOnMatrix(x -> x * value);
    }

    /**
     * Divides every number in the matrix by a given value
     * @param value the number to be used for division
     * @return a new matrix that is the result of the operation
     */
    public Matrix divide(double value) {
        return this.preformOnMatrix(x -> x / value);
    }

    /**
     * Preforms a given lambda expression on every matrix element
     * @param function the lambda expression
     * @return a new matrix that is the result of the operation
     */
    public Matrix preformOnMatrix(Function<Double, Double> function) {
        Matrix newMatrix = new Matrix(this.getWidth(), this.getHeight());
        for (int i = 0; i < this.getWidth(); i++)
            for (int j = 0; j < this.getHeight(); j++)
                newMatrix.set(i, j, function.apply(this.get(i, j)));
        return newMatrix;
    }

    /**
     * @return the sum of all elements in the matrix
     */
    public double sum() {
        double sum = 0;
        for (int i = 0; i < this.getWidth(); i++)
            for (int j = 0; j < this.getHeight(); j++)
                sum += this.get(i, j);
        return sum;
    }

    @Override
    public boolean equals(Object o) {
        // Check if o is instance of matrix
        if (o instanceof Matrix) {
            Matrix m = (Matrix)o;
            // Check both matrices are the same size
            if (m.getHeight() != this.getHeight())
                return false;
            if (m.getWidth() != this.getWidth())
                return false;
            // Check if the numbers in the matrices are the same
            for (int i = 0; i < this.getWidth(); i++)
                for (int j = 0; j < this.getHeight(); j++)
                    if (m.get(i, j) != this.get(i, j))
                        return false;
            // If all tests are passed the two matrices are equal
            return true;
        }
        return false;
    }

    @Override
    public String toString() {
        StringBuilder matrix_text = new StringBuilder();
        for (int j = 0; j < this.height; j++) {
            matrix_text.append("[");
            for (int i = 0; i < this.width; i++)
                matrix_text.append(this.get(i, j)).append(i == this.width - 1 ? "]" : ", ");
            if (j != this.height - 1)
                matrix_text.append("\n");
        }
        return matrix_text.toString();
    }

}
