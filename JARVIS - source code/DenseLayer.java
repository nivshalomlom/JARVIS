/**
 * A dense layer implementation in java
 */
public class DenseLayer implements NeuronLayer {

    // A connection to the main network controller
    private final NeuralNetwork master;

    private final int inputDimensions;
    private final int outputDimensions;

    // Forward propagation variables
    private final ActivationFunction activation;
    private Matrix weights;
    private Matrix biases;

    // Backwards propagation variables
    private Matrix lastWeightChange;
    private Matrix lastBiasChange;
    private Matrix lastInput;
    private Matrix lastZVector;

    /**
     * Creates a new dense layer
     * @param inputDimensions the number of inputs
     * @param outputDimensions the number of outputs
     * @param activation the activation function used in this layer
     */
    public DenseLayer(int inputDimensions, int outputDimensions, ActivationFunction activation, NeuralNetwork master) {
        // Setting up a connection to the main network controller
        this.master = master;
        // Save input and output dimensions
        this.inputDimensions = inputDimensions;
        this.outputDimensions = outputDimensions;
        // Initializing biases to zero
        this.biases = new Matrix(1, outputDimensions);
        // Initializing weights with xavier initialization
        double desiredVariance = (activation.equals(ActivationFunction.RELU) ? 2.0 : 1.0) / inputDimensions;
        this.weights = new Matrix(inputDimensions, outputDimensions);
        for (int i = 0; i < this.weights.getWidth(); i++)
            for (int j = 0; j < this.weights.getHeight(); j++)
                // Generate a random normally distributed number
                this.weights.set(i, j, (MLToolkit.RANDOM.nextGaussian() / 3) * desiredVariance);
        // Setting the activation function
        this.activation = activation;
        // Initializing the training variables
        this.lastWeightChange = new Matrix(inputDimensions, outputDimensions);
        this.lastBiasChange = new Matrix(1, outputDimensions);
    }

    // A private constructor for layer copying
    private DenseLayer(Matrix weights, Matrix biases, ActivationFunction activationFunction, NeuralNetwork master) {
        // Setting up a connection to the main network controller
        this.master = master;
        // Save input and output dimensions
        this.inputDimensions = weights.getWidth();
        this.outputDimensions = weights.getHeight();
        // Initializing biases to zero
        this.biases = biases;
        // Initializing weights
        this.weights = weights;
        // Setting the activation function
        this.activation = activationFunction;
        // Initializing the training variables
        this.lastWeightChange = new Matrix(inputDimensions, outputDimensions);
        this.lastBiasChange = new Matrix(1, outputDimensions);
    }

    @Override
    public Matrix forward(Matrix input) throws Exception {
        // Save the last input for later backwards propagation
        this.lastInput = input;
        // Compute z = w * x + b and save for later backwards propagation
        this.lastZVector = this.weights.dot(input).addition(this.biases);
        // Compute and return a = activation(z)
        if (this.activation.getName().equals("softmax"))
            return MLToolkit.softmax(this.lastZVector);
        return this.lastZVector.preformOnMatrix(this.activation::apply);

    }

    @Override
    public Matrix backpropagation(Matrix cost) throws Exception {
        // Get the learning rate of the network
        double learningRate = this.master.getLearningRate();
        // Pre compute the second half of all equations, activation'(z) * cost
        Matrix second_Half = new Matrix(1, this.outputDimensions);
        if (this.activation.getName().equals("softmax"))
            second_Half = MLToolkit.softmaxDerivative(this.lastZVector).multiplicationElementWise(cost);
        else for (int i = 0; i < second_Half.getHeight(); i++)
                second_Half.set(0, i, this.activation.applyDerivative(this.lastZVector.get(0, i)) * cost.get(0, i));
        // Compute bias changes
        for (int i = 0; i < this.biases.getHeight(); i++)
            this.lastBiasChange.set(0, i, this.lastBiasChange.get(0, i) - second_Half.get(0, i) * learningRate);
        // Compute previous layer activation cost and weight changes
        Matrix lastLayerCost = new Matrix(1, this.inputDimensions);
        for (int i = 0; i < this.weights.getWidth(); i++)
            for (int j = 0; j < this.weights.getHeight(); j++) {
                lastLayerCost.set(0, i, lastLayerCost.get(0, i) + this.weights.get(i, j) * second_Half.get(0, j));
                this.lastWeightChange.set(i, j, this.lastWeightChange.get(i, j) - this.lastInput.get(0, i) * second_Half.get(0, j) * learningRate);
            }
        // Send the new cost to the previous layer
        return lastLayerCost;
    }

    @Override
    public void commitGradientStep(int batchSize) throws Exception {
        // Get the momentum of the network
        double momentum = this.master.getMomentum();
        // Changing biases
        this.lastBiasChange = this.lastBiasChange.divide(batchSize);
        this.biases = this.biases.addition(this.lastBiasChange);
        this.lastBiasChange = this.lastBiasChange.multiply(momentum);
        // Changing weights
        this.lastWeightChange = this.lastWeightChange.divide(batchSize);
        this.weights = this.weights.addition(this.lastWeightChange);
        this.lastWeightChange = this.lastWeightChange.multiply(momentum);
    }

    @Override
    public NeuronLayer breed(NeuronLayer other, double mutate_chance, NeuralNetwork newMaster) {
        if (other instanceof DenseLayer) {
            // Breed only if the 2 layers are the same type!
            DenseLayer otherLayer = (DenseLayer)other;
            // Randomly choose layer parameters from the 2 parent layers
            int inputDimensions = MLToolkit.RANDOM.nextBoolean() ? this.inputDimensions : otherLayer.inputDimensions;
            int outputDimensions = MLToolkit.RANDOM.nextBoolean() ? this.outputDimensions : otherLayer.outputDimensions;
            ActivationFunction function = MLToolkit.RANDOM.nextBoolean() ? this.activation : otherLayer.activation;
            DenseLayer newLayer = new DenseLayer(inputDimensions, outputDimensions, function, newMaster);
            // Breed the weights and biases
            for (int j = 0; j < outputDimensions; j++) {
                newLayer.biases.set(0, j, MLToolkit.breedAndMutate(this.biases.get(0, j), otherLayer.biases.get(0, j), mutate_chance));
                for (int i = 0; i < inputDimensions; i++)
                    newLayer.weights.set(i, j, MLToolkit.breedAndMutate(this.weights.get(i, j), otherLayer.weights.get(i, j), mutate_chance));
            }
            return newLayer;
        }
        // If not the same type cant breed to return null
        else return null;
    }

    @Override
    public void setIsInTrainingMode(boolean isInTraining) {}

    @Override
    public int getInputDimensions() {
        return this.inputDimensions;
    }

    @Override
    public int getOutputDimensions() {
        return this.outputDimensions;
    }

    @Override
    public String toString() {
        // DenseLayer text = d|weights|biases|activation
        StringBuilder layerText = new StringBuilder("d|");
        // Append weights in rows, e.g weights = row1/row2/row3...
        layerText.append(this.weights.toString().replace("\n", "/")).append("|");
        // Append biases
        layerText.append(this.biases.transpose()).append("|");
        // Append activation
        layerText.append(this.activation.getName());
        // Return text
        return layerText.toString();
    }

    public static DenseLayer readFromString(String encoding, NeuralNetwork master) {
        // Check if encoding is of dense layer
        if (encoding.charAt(0) == 'd') {
            String[] split = encoding.replace(" ", "").split("\\|");
            // Read weights
            String[] rows = split[1].split("/");
            double[][] matrixNumbers = new double[rows.length][];
            for (int i = 0; i < rows.length; i++) {
                String[] numbers = rows[i].substring(1, rows[i].length() - 1).split(",");
                matrixNumbers[i] = new double[numbers.length];
                for (int j = 0; j < numbers.length; j++)
                    matrixNumbers[i][j] = Double.parseDouble(numbers[j]);
            }
            Matrix weights = new Matrix(matrixNumbers[0].length, matrixNumbers.length);
            for (int i = 0; i < weights.getWidth(); i++)
                for (int j = 0; j < weights.getHeight(); j++)
                    weights.set(i, j, matrixNumbers[j][i]);
            // Read biases
            String[] biasesString = split[2].substring(1, split[2].length() - 1).split(",");
            Matrix biases = new Matrix(1, biasesString.length);
            for (int i = 0; i < biases.getHeight(); i++)
                biases.set(0, i, Double.parseDouble(biasesString[i]));
            // Read activation
            ActivationFunction activationFunction = ActivationFunction.activationDictionary.get(split[3]);
            // Build new layer
            return new DenseLayer(weights, biases, activationFunction, master);
        }
        // Return null if not
        return null;
    }

}
