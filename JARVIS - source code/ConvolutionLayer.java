import java.util.Arrays;

/**
 * A java implementation of a convolution layer
 */
public class ConvolutionLayer implements NeuronLayer {

    // Shapes of relevant matrices
    private int[] inputShape;
    private int[] outputShape;
    private int[] filterShape;

    // The area of a dimension/layer in the output
    private int outputDimJump;

    // Learnable parameters
    private Matrix filters;
    private double[] biases;

    // Convolution tables
    private Matrix forwardTable;

    // Training parameters for momentum
    private Matrix lastFilterChange;
    private double[] lastBiasChange;
    private Matrix lastZVector;
    private Matrix lastInput;

    // Hyper parameters
    private int[] stride;
    private int[] padding;
    private ActivationFunction activationFunction;

    // A connection to the master network to get access to learning rate and momentum
    private NeuralNetwork master;

    /**
     * A constructor to build a new convolution layer
     * @param inputShape the shape of the input, a array containing (width, depth, height)
     * @param filterWidth the width of the layer's filter
     * @param filterHeight the height of the layer's filter
     * @param numberOfFilters the number of filters to apply
     * @param stride a array containing the amount of steps to take (horizontally after a patch, vertically after finishing a row)
     * @param padding a array containing the amount of padding (horizontally, vertically)
     * @param activationFunction the activation to apply to the layer's output
     * @param master a connection to the master network
     */
    public ConvolutionLayer(int[] inputShape, int filterWidth, int filterHeight, int numberOfFilters, int[] stride, int[] padding, ActivationFunction activationFunction, NeuralNetwork master) {
        // Initialize shapes
        this.inputShape = inputShape;
        this.filterShape = new int[] {filterWidth, filterHeight, 1};
        this.outputShape = new int[] {
                1 + (inputShape[0] + 2 * padding[0] - filterWidth) / stride[0],
                1 + (inputShape[1] + 2 * padding[1] - filterHeight) / stride[0],
                numberOfFilters
        };
        this.outputDimJump = this.outputShape[0] * this.outputShape[1];
        // Initialize learnable parameters
        this.filters = generateFilterTable(filterWidth, filterHeight, 1, numberOfFilters, activationFunction, this.getInputDimensions());
        this.biases = new double[numberOfFilters];
        // Initialize convolution tables
        this.forwardTable = MLToolkit.generateConvTable(this.inputShape, this.filterShape, padding, stride);
        // Initialize training parameters
        this.lastFilterChange = new Matrix(this.filters.getWidth(), this.filters.getHeight());
        this.lastBiasChange = new double[numberOfFilters];
        // Initialize hyper parameters
        this.stride = stride;
        this.padding = padding;
        this.activationFunction = activationFunction;
        // Initialize connection to master network
        this.master = master;
    }

    // A private constructor for reading from string
    private ConvolutionLayer(int[] inputShape,int filterWidth , int filterHeight, Matrix filterTable, double[] biases, int[] stride, int[] padding, ActivationFunction activationFunction, NeuralNetwork master) {
        // Initialize shapes
        this.inputShape = inputShape;
        this.filterShape = new int[] {filterWidth, filterHeight, 1};
        this.outputShape = new int[] {
                1 + (inputShape[0] + 2 * padding[0] - filterWidth) / stride[0],
                1 + (inputShape[1] + 2 * padding[1] - filterHeight) / stride[0],
                filterTable.getHeight()
        };
        this.outputDimJump = this.outputShape[0] * this.outputShape[1];
        // Initialize learnable parameters
        this.filters = filterTable;
        this.biases = biases;
        // Initialize convolution tables
        this.forwardTable = MLToolkit.generateConvTable(this.inputShape, this.filterShape, padding, stride);
        // Initialize training parameters
        this.lastFilterChange = new Matrix(this.filters.getWidth(), this.filters.getHeight());
        this.lastBiasChange = new double[filterTable.getHeight()];
        // Initialize hyper parameters
        this.stride = stride;
        this.padding = padding;
        this.activationFunction = activationFunction;
        // Initialize connection to master network
        this.master = master;
    }

    /**
     * A method used to build the matrix containing the layer's filters in a way it could be used for fast convolution O(n^2), with a input table
     * e.g: each row is a filter in vector form
     * @param filterWidth the height of the layer's filter
     * @param filterHeight the width of the layer's filter
     * @param numberOfFilters the number of filters to create
     * @return a filter table matrix
     */
    private static Matrix generateFilterTable(int filterWidth, int filterHeight, int filterDepth, int numberOfFilters, ActivationFunction activation, int inputDimensions) {
        // Create the table
        Matrix filterTable = new Matrix(filterWidth * filterHeight * filterDepth, numberOfFilters);
        // Return and fill with normally distributed values between [-1, 1] and multiply by desired variance (xavier initialization)
        double desiredVariance = (activation.equals(ActivationFunction.RELU) ? 2.0 : 1.0) / inputDimensions;
        return filterTable.preformOnMatrix(x -> (MLToolkit.RANDOM.nextGaussian() / 3) * desiredVariance);
    }

    /**
     * A method to preform a convolution of a given input with a given filterTable and inputTable
     * @param input the input to be convoluted
     * @param inputTables the input table generated with the generateFilterTable method from this class
     * @param filterTable a table where each row is a filter
     * @param biases the biases of each filter (if there's any)
     * @return the result of the convolution in a vector form
     */
    private static Matrix convolveWithTable(Matrix input, Matrix inputTables, Matrix filterTable, double[] biases) {
        // Create the matrix to hold to result in a vector from, each row represents a dimension
        Matrix convolutionResult = new Matrix(1 ,inputTables.getWidth() * filterTable.getHeight());
        // Compute filterTable dot inputTable
        for (int i = 0; i < inputTables.getWidth(); i++)
            for (int j = 0; j < filterTable.getHeight(); j++) {
                // Add the biases if provided ( saves time later, allows to skips a O(n^3) procedure of adding the biases after this)
                double sum = biases == null ? 0 : biases[j];
                for (int k = 0, z = 0; k < filterTable.getWidth(); k++, z++) {
                    // In case filter is a 1d repeating filter (e.g the layers filter) reseat index when needed
                    if (z >= filterTable.getWidth())
                        z = 0;
                    // Sum up the patch
                    sum += input.get(0, (int) inputTables.get(i, k)) * filterTable.get(z, j);
                }
                // Store in the result matrix
                convolutionResult.set(0, i + j * inputTables.getWidth(), sum);
            }
        // Return the result
        return convolutionResult;
    }

    @Override
    public Matrix forward(Matrix input) throws Exception {
        // Pad if needed
        if (this.padding[0] > 0 || this.padding[1] > 0)
            input = MLToolkit.pad(input, this.inputShape, this.padding);
        // Save last input for backpropagation
        this.lastInput = input;
        // Preform the convolution
        this.lastZVector = convolveWithTable(input, this.forwardTable, this.filters, this.biases);
        // Preform the activation function
        if (this.activationFunction.getName().equals("softmax"))
            return MLToolkit.softmax(this.lastZVector);
        else return this.lastZVector.preformOnMatrix(this.activationFunction::apply);
    }

    @Override
    public Matrix backpropagation(Matrix cost) throws Exception {
        // Get learning rate
        double learningRate = this.master.getLearningRate();
        // Compute dL/dz = cost * activationDerivative(z)
        Matrix secondPart = new Matrix(1, cost.getHeight());
        if (this.activationFunction.getName().equals("softmax")) {
            // If activation is softmax
            Matrix derivative = MLToolkit.softmaxDerivative(this.lastZVector);
            for (int i = 0; i < secondPart.getHeight(); i++)
                secondPart.set(0, i, cost.get(0, i) * derivative.get(0, i));
        }
        else
            // If activation isn't softmax
            for (int i = 0; i < secondPart.getHeight(); i++)
                secondPart.set(0, i, cost.get(0, i) * this.activationFunction.applyDerivative(this.lastZVector.get(0, i)));
        // Compute db
        for (int i = 0, k = 0; i < secondPart.getHeight(); i++) {
            if (i % this.forwardTable.getWidth() == 0 && i > 0)
                k++;
            this.lastBiasChange[k] -= learningRate * secondPart.get(0, i);
        }
        // Compute dw and dx
        Matrix dx = new Matrix(1, (this.inputShape[0] + 2 * this.padding[0]) * (this.inputShape[1] + 2 * padding[1]) * this.inputShape[2]);
        for (int i = 0; i < this.filters.getWidth(); i++)
            for (int j = 0; j < this.filters.getHeight(); j++) {
                double dw_sum = 0;
                // For each weight sum up all parts it has a effect on
                for (int k = 0; k < this.forwardTable.getWidth(); k++) {
                    dw_sum += this.lastInput.get(0, (int) this.forwardTable.get(k, i)) * secondPart.get(0, k + j * this.outputDimJump);
                    dx.set(0, (int) this.forwardTable.get(k, i),  dx.get(0, (int) this.forwardTable.get(k, i)) + this.filters.get(i, j) * secondPart.get(0, k + j * this.outputDimJump));
                }
                // Update dw
                this.lastFilterChange.set(i, j, this.lastFilterChange.get(i, j) - learningRate * dw_sum);
            }
        // Remove padding if needed
        if (this.padding[0] > 0 || this.padding[1] > 0)
            dx = MLToolkit.removePadding(dx, this.inputShape, this.padding);
        // Sends dx to previous layer
        return dx;
    }

    @Override
    public void commitGradientStep(int batchSize) throws Exception {
        // Get momentum
        double momentum = this.master.getMomentum();
        // Update weights
        this.lastFilterChange = this.lastFilterChange.divide(batchSize);
        this.filters = this.filters.addition(this.lastFilterChange);
        this.lastFilterChange = this.lastFilterChange.multiply(momentum);
        // Update biases
        for (int i = 0; i < this.biases.length; i++) {
            this.lastBiasChange[i] /= batchSize;
            this.biases[i] += this.lastBiasChange[i];
            this.lastBiasChange[i] *= momentum;
        }
    }

    @Override
    public NeuronLayer breed(NeuronLayer other, double mutate_chance, NeuralNetwork newMaster) {
        if (other instanceof ConvolutionLayer) {
            ConvolutionLayer otherLayer = (ConvolutionLayer)other;
            // Create new filters
            Matrix newFilters = new Matrix(this.filters.getWidth(), this.filters.getHeight());
            for (int i = 0; i < newFilters.getWidth(); i++)
                for (int j = 0; j < newFilters.getHeight(); j++)
                    newFilters.set(i, j, MLToolkit.breedAndMutate(this.filters.get(i, j), otherLayer.filters.get(i, j), mutate_chance));
            // Create new biases
            double[] newBiases = new double[this.biases.length];
            for (int i = 0; i < newBiases.length; i++)
                newBiases[i] = MLToolkit.breedAndMutate(this.biases[i], otherLayer.biases[i], mutate_chance);
            // Return new layer
            return new ConvolutionLayer(this.inputShape, this.filterShape[0], this.filterShape[1], newFilters, newBiases, this.stride, this.padding, MLToolkit.RANDOM.nextBoolean() ? this.activationFunction : otherLayer.activationFunction, newMaster);
        }
        else return null;
    }

    @Override
    public void setIsInTrainingMode(boolean isInTraining) { }

    @Override
    public int getInputDimensions() {
        return this.inputShape[0] * this.inputShape[1] * this.inputShape[2];
    }

    @Override
    public int getOutputDimensions() {
        return this.outputShape[0] * this.outputShape[1] * this.outputShape[2];
    }

    /**
     * @return the output shape of this layer
     */
    public int[] getOutputShape() {
        return outputShape;
    }

    @Override
    public String toString() {
        // ConvolutionLayer text = c|width, height, depth|filter matrix|f_width, f_height|biases|stride|padding|activation
        StringBuilder layerText = new StringBuilder("c|");
        // Append input shape
        layerText.append(this.inputShape[0]).append(",").append(this.inputShape[1]).append(",").append(this.inputShape[2]).append("|");
        // Append filter matrix
        layerText.append(this.filters.toString().replace("\n", "/")).append("|");
        // Append filter width, height
        layerText.append(this.filterShape[0]).append(",").append(this.filterShape[1]).append("|");
        // Append biases
        layerText.append(Arrays.toString(this.biases)).append("|");
        // Append stride
        layerText.append(this.stride[0]).append(",").append(this.stride[1]).append("|");
        // Append padding
        layerText.append(this.padding[0]).append(",").append(this.padding[1]).append("|");
        // Append activation
        layerText.append(this.activationFunction.getName());
        // Return encoding
        return layerText.toString();
    }

    // Reads a convolutional layer from a string encoding
    public static ConvolutionLayer readFromString(String encoding, NeuralNetwork master) {
        // Check if encoding is conv layer
        if (encoding.charAt(0) == 'c') {
            String[] split = encoding.replace(" ", "").split("\\|");
            // Read input shape
            int[] inputShape = new int[3];
            String[] shape = split[1].split(",");
            for (int i = 0; i < 3; i++)
                inputShape[i] = Integer.parseInt(shape[i]);
            // Read filter matrix
            String[] rows = split[2].split("/");
            double[][] matrixNumbers = new double[rows.length][];
            for (int i = 0; i < rows.length; i++) {
                String[] numbers = rows[i].substring(1, rows[i].length() - 1).split(",");
                matrixNumbers[i] = new double[numbers.length];
                for (int j = 0; j < numbers.length; j++)
                    matrixNumbers[i][j] = Double.parseDouble(numbers[j]);
            }
            Matrix filterTable = new Matrix(matrixNumbers[0].length, matrixNumbers.length);
            for (int i = 0; i < filterTable.getWidth(); i++)
                for (int j = 0; j < filterTable.getHeight(); j++)
                    filterTable.set(i, j, matrixNumbers[j][i]);
            // Read filter width, height
            String[] width_height = split[3].split(",");
            int filter_width = Integer.parseInt(width_height[0]);
            int filter_height = Integer.parseInt(width_height[1]);
            // Read biases
            String[] biasNumbers = split[4].substring(1, split[4].length() - 1).split(",");
            double[] biases = new double[biasNumbers.length];
            for (int i = 0; i < biases.length; i++)
                biases[i] = Double.parseDouble(biasNumbers[i]);
            // Read stride
            int[] stride = new int[2];
            String[] strideText = split[5].split(",");
            stride[0] = Integer.parseInt(strideText[0]);
            stride[1] = Integer.parseInt(strideText[1]);
            // Read padding
            int[] padding = new int[2];
            String[] paddingText = split[6].split(",");
            padding[0] = Integer.parseInt(paddingText[0]);
            padding[1] = Integer.parseInt(paddingText[1]);
            // Read activation
            ActivationFunction activationFunction = ActivationFunction.activationDictionary.get(split[7]);
            // Return new layer
            return new ConvolutionLayer(inputShape, filter_width, filter_height, filterTable, biases, stride, padding, activationFunction, master);
        }
        // If not return null
        return null;
    }

}
