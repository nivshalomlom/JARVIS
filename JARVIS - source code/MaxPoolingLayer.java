/**
 * A max pooling layer in java
 */
public class MaxPoolingLayer implements NeuronLayer, Cloneable {

    // The input/output shape
    private int[] inputShape;
    private int[] outputShape;

    // Pool size
    private int poolWidth;
    private int poolHeight;

    // The input mapping for pooling
    private Matrix poolTable;

    // The activation to apply to layer output
    private ActivationFunction activationFunction;

    // Save last z for backpropagation
    private Matrix lastZ;

    // Saves the index if the max elements found for backpropagation
    private int[] maxMapping;

    /**
     * Creates a new max pooling layer
     * @param inputShape the shape of the expected input
     * @param poolWidth the width of the layer's pool
     * @param poolHeight the height of the layer's pool
     */
    public MaxPoolingLayer(int[] inputShape, int poolWidth, int poolHeight, ActivationFunction activationFunction) {
        this.initLayer(inputShape, poolWidth, poolHeight, MLToolkit.generatePoolTable(inputShape, poolWidth, poolHeight), activationFunction);
    }

    // A private constructor for breed / clone
    private MaxPoolingLayer(int[] inputShape, int poolWidth, int poolHeight, Matrix poolTable, ActivationFunction activationFunction) {
        this.initLayer(inputShape, poolWidth, poolHeight, poolTable, activationFunction);
    }

    // A method to initialize the layer
    private void initLayer(int[] inputShape, int poolWidth, int poolHeight, Matrix poolTable, ActivationFunction activationFunction) {
        // Initialize the pool
        this.poolWidth = poolWidth;
        this.poolHeight = poolHeight;
        // Initialize IO shapes
        this.inputShape = inputShape;
        this.outputShape = new int[] {
                1 + (this.inputShape[0] - poolWidth) / poolWidth,
                1 + (this.inputShape[1] - poolHeight) / poolHeight,
                this.inputShape[2]
        };
        // Build the conv table
        this.poolTable = poolTable;
        // Store activation
        this.activationFunction = activationFunction;
        // Array for storing max elements indices
        this.maxMapping = new int[this.poolTable.getWidth()];
    }

    @Override
    public Matrix forward(Matrix input) throws Exception {
        // Max pool
        this.lastZ = this.maxPool(input, this.poolTable);
        // Preform activation
        if (this.activationFunction.getName().equals("softmax"))
            return MLToolkit.softmax(this.lastZ);
        else return this.lastZ.preformOnMatrix(this.activationFunction::apply);
    }

    // A method to preform max pooling on a given input and a convolution table
    private Matrix maxPool(Matrix input, Matrix inputTables) {
        // Create the matrix to hold to result in a vector from, each row represents a dimension
        Matrix convolutionResult = new Matrix(1 ,inputTables.getWidth());
        // Compute filterTable dot inputTable
        for (int i = 0; i < inputTables.getWidth(); i++) {
            double max = Double.NEGATIVE_INFINITY;
            int maxIndex = -1;
            for (int k = 0; k < inputTables.getHeight(); k++) {
                // Find max in column
                int index = (int) inputTables.get(i, k);
                double newNumber = input.get(0, index);
                if (max < newNumber) {
                    max = newNumber;
                    maxIndex = index;
                }
            }
            // Save the index for backpropagation later
            this.maxMapping[i] = maxIndex;
            // Store in the result matrix
            convolutionResult.set(0, i, max);
        }
        // Return the result
        return convolutionResult;
    }

    @Override
    public Matrix backpropagation(Matrix cost) throws Exception {
        // Pre compute the second half of all equations, activation'(z) * cost
        Matrix second_Half = new Matrix(1, cost.getHeight());
        if (this.activationFunction.getName().equals("softmax"))
            second_Half = MLToolkit.softmaxDerivative(this.lastZ).multiplicationElementWise(cost);
        else for (int i = 0; i < second_Half.getHeight(); i++)
            second_Half.set(0, i, this.activationFunction.applyDerivative(this.lastZ.get(0, i)) * cost.get(0, i));
        // Rebuild matrix with cost in place of max values
        Matrix result = new Matrix(1, this.inputShape[0] * this.inputShape[1] * this.inputShape[2]);
        for (int i = 0; i < cost.getHeight(); i++)
            result.set(0, this.maxMapping[i], second_Half.get(0, i));
        // Return the cost of the previous layer
        return result;
    }

    @Override
    public void commitGradientStep(int batchSize) throws Exception {}

    @Override
    public NeuronLayer breed(NeuronLayer other, double mutate_chance, NeuralNetwork newMaster) {
        return this.clone();
    }

    @Override
    public void setIsInTrainingMode(boolean isInTraining) {}

    @Override
    public int getInputDimensions() {
        return this.inputShape[0] * this.inputShape[1] * this.inputShape[2];
    }

    @Override
    public int getOutputDimensions() {
        return this.outputShape[0] * this.outputShape[1] * this.outputShape[2];
    }

    @Override
    protected MaxPoolingLayer clone()  {
        return new MaxPoolingLayer(this.inputShape, this.poolWidth, this.poolHeight, this.poolTable, this.activationFunction);
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
        StringBuilder layerText = new StringBuilder("m|");
        // Append input shape
        layerText.append(this.inputShape[0]).append(",").append(this.inputShape[1]).append(",").append(this.inputShape[2]).append("|");
        // Append pool width and height
        layerText.append(this.poolWidth).append(",").append(poolHeight).append("|");
        // Append activation
        layerText.append(this.activationFunction.getName());
        // Return text
        return layerText.toString();
    }

    // A method to build a average pooling layer from string encoding
    public static MaxPoolingLayer readFromString(String encoding) {
        // If encoding is average pooling layer
        if (encoding.charAt(0) == 'm') {
            String[] split = encoding.replace(" ", "").split("\\|");
            // Read input shape
            int[] inputShape = new int[3];
            String[] shape = split[1].split(",");
            for (int i = 0; i < 3; i++)
                inputShape[i] = Integer.parseInt(shape[i]);
            // Read pool width and height
            String[] pool = split[2].split(",");
            int poolWidth = Integer.parseInt(pool[0]);
            int poolHeight = Integer.parseInt(pool[1]);
            // Read activation function
            ActivationFunction activationFunction = ActivationFunction.activationDictionary.get(split[3]);
            // Return the new layer
            return new MaxPoolingLayer(inputShape, poolWidth, poolHeight, activationFunction);
        }
        // Else return null
        else return null;
    }

}
