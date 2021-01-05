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
    private Matrix poolTable;

    // The activation to apply to layer output
    private ActivationFunction activationFunction;

    // Save last z for backpropagation
    private Matrix lastZ;

    // Saves the index if the max elements found for backpropagation
    private double[] meanMapping;

    /**
     * Creates a new average pooling layer
     * @param inputShape the shape of the expected input
     * @param poolWidth the width of the layer's pool
     * @param poolHeight the height of the layer's pool
     */
    public AveragePoolingLayer(int[] inputShape, int poolWidth, int poolHeight, ActivationFunction activationFunction) {
        this.initLayer(inputShape, poolWidth, poolHeight, MLToolkit.generatePoolTable(inputShape, poolWidth, poolHeight), activationFunction);
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
                inputShape[2]
        };
        // Save activation
        this.activationFunction = activationFunction;
        // Build the conv table
        this.poolTable = poolTable;
        // Array for storing max elements indices
        this.meanMapping = new double[this.poolTable.getWidth()];
    }

    @Override
    public Matrix forward(Matrix input) throws Exception {
        // Pool
       this.lastZ = this.averagePool(input, this.poolTable);
       // Preform activation
       if (this.activationFunction.getName().equals("softmax"))
           return MLToolkit.softmax(this.lastZ);
       else return this.lastZ.preformOnMatrix(this.activationFunction::apply);
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
        // Pre compute the second half of all equations, activation'(z) * cost
        Matrix second_Half = new Matrix(1, cost.getHeight());
        if (this.activationFunction.getName().equals("softmax"))
            second_Half = MLToolkit.softmaxDerivative(this.lastZ).multiplicationElementWise(cost);
        else for (int i = 0; i < second_Half.getHeight(); i++)
            second_Half.set(0, i, this.activationFunction.applyDerivative(this.lastZ.get(0, i)) * cost.get(0, i));
        // Fill each pool block with its average value
        for (int i = 0; i < this.poolTable.getWidth(); i++) {
            double value = second_Half.get(0, i) * this.meanMapping[i];
            for (int j = 0; j < this.poolTable.getHeight(); j++)
                result.set(0, (int) this.poolTable.get(i, j), value);
        }
        // Return the result
        return result;
    }

    @Override
    public void commitGradientStep(int batchSize) throws Exception {

    }

    @Override
    public NeuronLayer breed(NeuronLayer other, double mutate_chance, NeuralNetwork newMaster) {
        return new AveragePoolingLayer(this.inputShape, this.poolWidth, this.poolHeight, this.activationFunction);
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

    /**
     * @return the expected output shape
     */
    public int[] getOutputShape() {
        return outputShape;
    }

    @Override
    public String toString() {
        // AveragePoolingLayer text = a|width, height, depth|pool width, pool height|activation
        StringBuilder layerText = new StringBuilder("a|");
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
            int poolHeight = Integer.parseInt(pool[1]);
            // Read activation function
            ActivationFunction activationFunction = ActivationFunction.activationDictionary.get(split[3]);
            // Return the new layer
            return new AveragePoolingLayer(inputShape, poolWidth, poolHeight, activationFunction);
        }
        // Else return null
        else return null;
    }

}
