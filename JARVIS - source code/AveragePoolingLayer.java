/**
 * A average pooling layer in java
 */
public class AveragePoolingLayer implements NeuronLayer {

    // The input/output shape
    private int[] inputShape;
    private int[] outputShape;

    // Pool size
    private int poolWidth;
    private int poolHeight;

    // The input mapping for pooling
    private Matrix convTable;

    // Saves the index if the max elements found for backpropagation
    private double[] meanMapping;

    /**
     * Creates a new average pooling layer
     * @param inputShape the shape of the expected input
     * @param poolWidth the width of the layer's pool
     * @param poolHeight the height of the layer's pool
     */
    public AveragePoolingLayer(int[] inputShape, int poolWidth, int poolHeight) {
        this.initLayer(inputShape, poolWidth, poolHeight, MLToolkit.generateConvTable(inputShape, new int[] {poolWidth, poolHeight, 1}, new int[] {0, 0}, new int[] {poolWidth, poolHeight}));
    }

    // A private constructor for quick cloning
    private AveragePoolingLayer(int[] inputShape, int poolWidth, int poolHeight, Matrix convTable) {
        this.initLayer(inputShape, poolWidth, poolHeight, convTable);
    }

    // A method to initialize the layer
    private void initLayer(int[] inputShape, int poolWidth, int poolHeight, Matrix convTable) {
        // Initialize the pool
        this.poolWidth = poolWidth;
        this.poolHeight = poolHeight;
        // Initialize IO shapes
        this.inputShape = inputShape;
        this.outputShape = new int[] {
                1 + (this.inputShape[0] - poolWidth) / poolWidth,
                1 + (this.inputShape[1] - poolHeight) / poolHeight,
                1
        };
        // Build the conv table
        this.convTable = convTable;
        // Array for storing max elements indices
        this.meanMapping = new double[this.convTable.getWidth()];
    }

    @Override
    public Matrix forward(Matrix input) throws Exception {
        return this.averagePool(input, this.convTable);
    }

    // A method to preform max pooling on a given input and a convolution table
    private Matrix averagePool(Matrix input, Matrix inputTables) {
        // Create the matrix to hold to result in a vector from, each row represents a dimension
        Matrix convolutionResult = new Matrix(1 ,inputTables.getWidth());
        // Compute filterTable dot inputTable
        for (int i = 0; i < inputTables.getWidth(); i++) {
            double mean = 0;
            for (int k = 0; k < inputTables.getHeight(); k++) {
                // Find max in column
                int index = (int) inputTables.get(i, k);
                double newNumber = input.get(0, index);
                mean += newNumber;
            }
            mean /= inputTables.getHeight();
            // Save the index for backpropagation later
            this.meanMapping[i] = mean;
            // Store in the result matrix
            convolutionResult.set(0, i, mean);
        }
        // Return the result
        return convolutionResult;
    }

    @Override
    public Matrix backpropagation(Matrix cost) throws Exception {
        // The result matrix
        Matrix result = new Matrix(1, this.inputShape[0] * this.inputShape[1] * this.inputShape[2]);
        // Fill each pool block with its average value
        for (int i = 0; i < this.convTable.getWidth(); i++) {
            double value = cost.get(0, i) * this.meanMapping[i];
            for (int j = 0; j < this.convTable.getHeight(); j++)
                result.set(0, (int) this.convTable.get(i, j), value);
        }
        // Return the result
        return result;
    }

    @Override
    public void commitGradientStep(int batchSize) throws Exception {

    }

    @Override
    public NeuronLayer breed(NeuronLayer other, double mutate_chance, NeuralNetwork newMaster) {
        return this.clone();
    }

    @Override
    public void setIsInTrainingMode(boolean isInTraining) {

    }

    @Override
    public int getInputDimensions() {
        return this.inputShape[0] * this.inputShape[1] * this.inputShape[2];
    }

    @Override
    public int getOutputDimensions() {
        return this.outputShape[0] * this.outputShape[1] * this.outputShape[2];
    }

    @Override
    protected AveragePoolingLayer clone()  {
        return new AveragePoolingLayer(this.inputShape, this.poolWidth, this.poolHeight, this.convTable);
    }

    /**
     * @return the expected output shape
     */
    public int[] getOutputShape() {
        return outputShape;
    }

    @Override
    public String toString() {
        // AveragePoolingLayer text = a|width, height, depth|pool width, pool height
        StringBuilder layerText = new StringBuilder("a|");
        // Append input shape
        layerText.append(this.inputShape[0]).append(",").append(this.inputShape[1]).append(",").append(this.inputShape[2]).append("|");
        // Append pool width and height
        layerText.append(this.poolWidth).append(",").append(poolHeight);
        // Return text
        return layerText.toString();
    }

    // A method to build a average pooling layer from string encoding
    public static AveragePoolingLayer readFromString(String encoding) {
        // If encoding is average pooling layer
        if (encoding.charAt(0) == 'a') {
            String[] split = encoding.replace(" ", "").split("\\|");
            // Read input shape
            int[] inputShape = new int[3];
            String[] shape = split[1].split(",");
            for (int i = 0; i < 3; i++)
                inputShape[i] = Integer.parseInt(shape[i]);
            // Read pool width and height
            String[] pool = split[2].split(",");
            int poolWidth = Integer.parseInt(pool[0]);
            int poolHeight = Integer.parseInt(pool[1]);;
            // Return the new layer
            return new AveragePoolingLayer(inputShape, poolWidth, poolHeight);
        }
        // Else return null
        else return null;
    }

}
