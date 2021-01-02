import java.io.*;
import java.util.*;

/**
 * A neural network implementation in java, with support for dense and batch normalization layers, supervised and unsupervised learning, dropout....
 */
public class NeuralNetwork {

    // The network layers
    private LinkedList<NeuronLayer> layers;

    // Network input shape and dimensions( = volume)
    private int inputDimensions;
    private int[] inputShape;

    // Network output shape and dimensions( = volume)
    private int outputDimensions;
    private int[] outputShape;

    // Training parameters, default values:
    private double learningRate = 0.5;
    private double momentum = 0.5;
    private int batchSize = -1;

    // Loss statistics parameters, default values:
    private LossFunctions lossFunctions = LossFunctions.MSE_2;
    private double averageLoss = Double.POSITIVE_INFINITY;

    // Flags
    private boolean printStatusMessages = false;

    /**
     * Builds a empty neural network
     * @param inputDimensions specifies the number of network inputs
     */
    public NeuralNetwork(int inputDimensions) {
        this.initNetwork(inputDimensions, new int[] {1, inputDimensions, 1});
    }

    /**
     * Builds a empty neural network
     * @param inputShape tells the network how to treat the vector inputs, a array containing (width, height, depth)
     */
    public NeuralNetwork(int[] inputShape) {
        this.initNetwork(inputShape[0] * inputShape[1] * inputShape[2], inputShape);
    }

    // A method to initialize the network's base parameters
    private void initNetwork(int inputDimensions, int[] inputShape) {
        this.layers = new LinkedList<>();
        this.inputDimensions = inputDimensions;
        this.outputDimensions = inputDimensions;
        this.inputShape = inputShape;
        this.outputShape = inputShape;
    }

    /**
     * A method to add a dense layer to the network
     * @param outputDimensions the number of outputs from the network
     * @param activationFunction the activation function to be used in the layer
     * @param batchNormalizeInputs if we should apply batch normalization to the layer inputs
     * @return return this instance of the class for chaining commands
     */
    public NeuralNetwork addDenseLayer(int outputDimensions, ActivationFunction activationFunction, boolean batchNormalizeInputs) {
        if (batchNormalizeInputs)
            // Create and add a new batch normalization layer if needed
            this.layers.add(new BatchNormalizationLayer(this.outputDimensions, this));
        // Create and add a new dense layer
        this.layers.add(new DenseLayer(this.outputDimensions, outputDimensions, activationFunction, this));
        // Updating the output dimensions and shape
        this.outputDimensions = outputDimensions;
        this.outputShape = new int[] {1, outputDimensions, 1};
        // Return this instance of the class for chaining commands
        return this;
    }

    /**
     * A method to add dense layers to the network
     * @param outputDimensions the number of outputs from the network
     * @param activationFunction the activation function to be used in the layer
     * @param batchNormalizeInputs if we should apply batch normalization to the layer inputs
     * @param amount the amount of layers to be added
     * @return return this instance of the class for chaining commands
     */
    public NeuralNetwork addDenseLayers(int outputDimensions, ActivationFunction activationFunction, boolean batchNormalizeInputs, int amount) {
        // Create and add the specified amount of layers
        for (int i = 0; i < amount; i++)
            this.addDenseLayer(outputDimensions, activationFunction, batchNormalizeInputs);
        return this;
    }

    /**
     * A method to add dense layers to the network
     * @param outputDimensions the number of outputs from the network
     * @param activationFunction the activation function to be used in the layer
     * @return return this instance of the class for chaining commands
     */
    public NeuralNetwork addDenseLayer(int outputDimensions, ActivationFunction activationFunction) {
        // Create and add the new layer
        this.addDenseLayer(outputDimensions, activationFunction, false);
        return this;
    }

    /**
     * A method to add dense layers to the network
     * @param outputDimensions the number of outputs from the network
     * @param activationFunction the activation function to be used in the layer
     * @param amount the amount of layers to be added
     * @return return this instance of the class for chaining commands
     */
    public NeuralNetwork addDenseLayers(int outputDimensions, ActivationFunction activationFunction, int amount) {
        // Create and add the specified amount of layers
        for (int i = 0; i < amount; i++)
            this.addDenseLayer(outputDimensions, activationFunction, false);
        return this;
    }

    /**
     * A method to add a dropout layer to the network
     * @param keepProbability the chance for inputs to be kept
     * @return return this instance of the class for chaining commands
     */
    public NeuralNetwork addDropOutLayer(double keepProbability) {
        // Add the new layer
        this.layers.add(new DropoutLayer(keepProbability, this.layers.getLast().getOutputDimensions()));
        // Return this instance of the class for chaining commands
        return this;
    }

    /**
     * Adds a convolution layer to the network
     * @param filterWidth the width of the layer's filter
     * @param filterHeight the height of the layer's filter
     * @param numberOfFilters the number of filters to apply
     * @param stride a array containing the amount of steps to take (horizontally after a patch, vertically after finishing a row)
     * @param padding a array containing the amount of padding (horizontally, vertically)
     * @param activationFunction the activation to apply to the layer's output
     * @param batchNormalizeInputs if we should apply batch normalization to the layer inputs
     * @return return this instance of the class for chaining commands
     */
    public NeuralNetwork addConvolutionLayer(int filterWidth, int filterHeight, int numberOfFilters, int[] stride, int[] padding, ActivationFunction activationFunction, boolean batchNormalizeInputs) {
        // Add batch norm if needed
        if (batchNormalizeInputs)
            this.layers.add(new BatchNormalizationLayer(this.outputDimensions, this));
        // Create the new layer
        ConvolutionLayer c = new ConvolutionLayer(this.outputShape, filterWidth, filterHeight, numberOfFilters, stride, padding, activationFunction, this);
        // Update network dimensions
        this.outputShape = c.getOutputShape();
        this.outputDimensions = c.getOutputDimensions();
        this.layers.add(c);
        // Return this instance of the class for chaining commands
        return this;
    }

    /**
     * Adds a convolution layer to the network
     * @param filterWidth the width of the layer's filter
     * @param filterHeight the height of the layer's filter
     * @param numberOfFilters the number of filters to apply
     * @param stride a array containing the amount of steps to take (horizontally after a patch, vertically after finishing a row)
     * @param padding a array containing the amount of padding (horizontally, vertically)
     * @param activationFunction the activation to apply to the layer's output
     * @return return this instance of the class for chaining commands
     */
    public NeuralNetwork addConvolutionLayer(int filterWidth, int filterHeight, int numberOfFilters, int[] stride, int[] padding, ActivationFunction activationFunction) {
        // Add the layer
        return this.addConvolutionLayer(filterWidth, filterHeight, numberOfFilters, stride, padding, activationFunction, false);
    }

    /**
     * Adds a specified amount of convolution layers to the network
     * @param filterWidth the width of the layer's filter
     * @param filterHeight the height of the layer's filter
     * @param numberOfFilters the number of filters to apply
     * @param stride a array containing the amount of steps to take (horizontally after a patch, vertically after finishing a row)
     * @param padding a array containing the amount of padding (horizontally, vertically)
     * @param activationFunction the activation to apply to the layer's output
     * @param batchNormalizeInputs if we should apply batch normalization to the layer inputs
     * @return return this instance of the class for chaining commands
     */
    public NeuralNetwork addConvolutionLayers(int filterWidth, int filterHeight, int numberOfFilters, int[] stride, int[] padding, ActivationFunction activationFunction, boolean batchNormalizeInputs, int amount) {
        // Add the specified amount of layers
        for (int i = 0; i < amount; i++)
            this.addConvolutionLayer(filterWidth, filterHeight, numberOfFilters, stride, padding, activationFunction, batchNormalizeInputs);
        // Return this instance of the class for chaining commands
        return this;
    }

    /**
     * Adds a specified amount of convolution layers to the network
     * @param filterWidth the width of the layer's filter
     * @param filterHeight the height of the layer's filter
     * @param numberOfFilters the number of filters to apply
     * @param stride a array containing the amount of steps to take (horizontally after a patch, vertically after finishing a row)
     * @param padding a array containing the amount of padding (horizontally, vertically)
     * @param activationFunction the activation to apply to the layer's output
     * @return return this instance of the class for chaining commands
     */
    public NeuralNetwork addConvolutionLayers(int filterWidth, int filterHeight, int numberOfFilters, int[] stride, int[] padding, ActivationFunction activationFunction, int amount) {
        // Add the specified amount of layers
        for (int i = 0; i < amount; i++)
            this.addConvolutionLayer(filterWidth, filterHeight, numberOfFilters, stride, padding, activationFunction, false);
        // Return this instance of the class for chaining commands
        return this;
    }

    /**
     * A method to add a dense layer that retains a specified 3d output shape (for use in convolutional neural networks)
     * @param outputWidth the width of the layer output
     * @param outputHeight the height of the layer output
     * @param outputDepth the depth of the layer output
     * @param activationFunction the activation to be applied to the output
     * @param batchNormalizeInputs if we should apply batch normalization to the layer inputs
     * @return return this instance of the class for chaining commands
     */
    public NeuralNetwork addFullyConnectedLayer(int outputWidth, int outputHeight, int outputDepth, ActivationFunction activationFunction, boolean batchNormalizeInputs) {
        // Add batch norm if needed
        if (batchNormalizeInputs)
            this.layers.add(new BatchNormalizationLayer(this.outputDimensions, this));
        // Compute the new output dimensions
        int newOut = outputWidth * outputHeight * outputDepth;
        // Create the new layer
        this.layers.add(new DenseLayer(this.outputDimensions, newOut, activationFunction, this));
        // Update input shape and dimensions
        this.outputShape[0] = outputWidth;
        this.outputShape[1] = outputHeight;
        this.outputShape[2] = outputDepth;
        this.outputDimensions = newOut;
        // Return this instance of the class for chaining commands
        return this;
    }

    /**
     * A method to add a dense layer that retains a specified 3d output shape (for use in convolutional neural networks)
     * @param outputWidth the width of the layer output
     * @param outputHeight the height of the layer output
     * @param outputDepth the depth of the layer output
     * @param activationFunction the activation to be applied to the output
     * @return return this instance of the class for chaining commands
     */
    public NeuralNetwork addFullyConnectedLayer(int outputWidth, int outputHeight, int outputDepth, ActivationFunction activationFunction) {
        // Add the new layer
        return this.addFullyConnectedLayer(outputWidth, outputHeight, outputDepth, activationFunction, false);
    }

    /**
     * A method to add a specified number of dense layers that retains a specified 3d output shape (for use in convolutional neural networks)
     * @param outputWidth the width of the layer output
     * @param outputHeight the height of the layer output
     * @param outputDepth the depth of the layer output
     * @param activationFunction the activation to be applied to the output
     * @param batchNormalizeInputs if we should apply batch normalization to the layer inputs
     * @param amount how many layers to add
     * @return return this instance of the class for chaining commands
     */
    public NeuralNetwork addFullyConnectedLayers(int outputWidth, int outputHeight, int outputDepth, ActivationFunction activationFunction, boolean batchNormalizeInputs, int amount) {
        // Adds the new layers
        for (int i = 0; i < amount; i++)
            this.addFullyConnectedLayer(outputWidth, outputHeight, outputDepth, activationFunction, batchNormalizeInputs);
        // Return this instance of the class for chaining commands
        return this;
    }

    /**
     * A method to add a specified number of dense layers that retains a specified 3d output shape (for use in convolutional neural networks)
     * @param outputWidth the width of the layer output
     * @param outputHeight the height of the layer output
     * @param outputDepth the depth of the layer output
     * @param activationFunction the activation to be applied to the output
     * @param amount how many layers to add
     * @return return this instance of the class for chaining commands
     */
    public NeuralNetwork addFullyConnectedLayers(int outputWidth, int outputHeight, int outputDepth, ActivationFunction activationFunction, int amount) {
        // Adds the new layers
        for (int i = 0; i < amount; i++)
            this.addFullyConnectedLayer(outputWidth, outputHeight, outputDepth, activationFunction, false);
        // Return this instance of the class for chaining commands
        return this;
    }

    /**
     * A method to add a max pooling layer to the network
     * @param poolWidth the width of the layer's pool
     * @param poolHeight the height of the layer's pool
     * @return return this instance of the class for chaining commands
     */
    public NeuralNetwork addMaxPoolingLayer(int poolWidth, int poolHeight) {
        // Create and add the new layer
        MaxPoolingLayer mpl = new MaxPoolingLayer(this.inputShape, poolWidth, poolHeight);
        this.layers.add(mpl);
        // Update output shape
        this.outputShape = mpl.getOutputShape();
        this.outputDimensions = mpl.getOutputDimensions();
        // Return this instance of the class for chaining commands
        return this;
    }

    /**
     * A method to add a average pooling layer to the network
     * @param poolWidth the width of the layer's pool
     * @param poolHeight the height of the layer's pool
     * @return return this instance of the class for chaining commands
     */
    public NeuralNetwork addAveragePoolingLayer(int poolWidth, int poolHeight) {
        // Create and add the new layer
        AveragePoolingLayer apl = new AveragePoolingLayer(this.inputShape, poolWidth, poolHeight);
        this.layers.add(apl);
        // Update output shape
        this.outputShape = apl.getOutputShape();
        this.outputDimensions = apl.getOutputDimensions();
        // Return this instance of the class for chaining commands
        return this;
    }

    /**
     * A method to send a data sample through the network
     * @param values the input values
     * @return the network's prediction
     * @throws Exception if the data is invalid or too big/small
     */
    public double[] input(double ...values) throws Exception {
        // Convert input to matrix
        Matrix input = Matrix.makeVerticalVector(values);
        // Run the input through the network
        for (NeuronLayer layer : layers)
            // Forward the input through the current layer
            input = layer.forward(input);
        // Return the prediction
        return input.getColumn(0);
    }

    /**
     * A method to train the neural network with given input output examples
     * @param inputs the sample inputs
     * @param outputs the outputs for the sample inputs
     * @param epochs the amount of cycles of training to do
     */
    public void train(List<double[]> inputs, List<double[]> outputs, int epochs) throws Exception {
        // Activate training mode
        this.setIsInTrainingMode(true);
        // If no batch size was specified it is set to the data set's size
        if (this.batchSize == -1)
            this.batchSize = inputs.size();
        // Convert all inputs and outputs to the matrix class for faster and easier processing
        LinkedList<Matrix> matrixInputs = new LinkedList<>();
        LinkedList<Matrix> matrixOutputs = new LinkedList<>();
        // Get the iterators so we would iterate over both simultaneously
        Iterator<double[]> inputIter = inputs.iterator();
        Iterator<double[]> outputIter = outputs.iterator();
        while (inputIter.hasNext() && outputIter.hasNext()) {
            matrixInputs.add(Matrix.makeVerticalVector(inputIter.next()));
            matrixOutputs.add(Matrix.makeVerticalVector(outputIter.next()));
        }
        // The training process
        for (int i = 0; i < epochs; i++) {
            // Keeping track of average loss each epoch
            double totalLoss = 0;
            // Create copies of the inputs and outputs for further epochs
            LinkedList<Matrix> inputCopy = new LinkedList<>(matrixInputs);
            LinkedList<Matrix> outputCopy = new LinkedList<>(matrixOutputs);
            while (!inputCopy.isEmpty() && !outputCopy.isEmpty()) {
                LinkedList<Matrix> inputBatch = new LinkedList<>();
                LinkedList<Matrix> outputBatch = new LinkedList<>();
                // Divide to batches randomly
                for (int j = 0; j < this.batchSize && !inputCopy.isEmpty() && !outputCopy.isEmpty(); j++) {
                    int randIndex = MLToolkit.RANDOM.nextInt(inputCopy.size());
                    inputBatch.add(inputCopy.remove(randIndex));
                    outputBatch.add(outputCopy.remove(randIndex));
                }
                totalLoss += this.processBatch(inputBatch, outputBatch);
            }
            // Save loss statistic
            this.averageLoss = totalLoss / inputs.size();
            // Print out status message if needed
            if (this.printStatusMessages)
                System.out.println("[ Epoch " + i + " done, average loss: " + this.averageLoss + " ]");
        }
        // Deactivate training mode
        this.setIsInTrainingMode(false);
    }

    // A method to process a given training batch
    private double processBatch(List<Matrix> inputs, List<Matrix> outputs) throws Exception {
        Iterator<Matrix> inputIter = inputs.iterator();
        Iterator<Matrix> outputIter = outputs.iterator();
        // Summing up the loss for diagnostics
        double batchLoss = 0;
        while (inputIter.hasNext() && outputIter.hasNext()) {
            // Get a iterator to iterate through the layers
            ListIterator<NeuronLayer> layerIter = this.layers.listIterator();
            // Forward the input through the network
            Matrix prediction = inputIter.next();
            while (layerIter.hasNext())
                prediction = layerIter.next().forward(prediction);
            // Backwards propagate the cost through the network
            Matrix target = outputIter.next();
            Matrix error = this.lossFunctions.computeCost(target, prediction);
            batchLoss += this.lossFunctions.computeLoss(target, prediction);
            while (layerIter.hasPrevious())
                error = layerIter.previous().backpropagation(error);
        }
        // Commit the changes to the network
        for (NeuronLayer layer : this.layers)
            layer.commitGradientStep(inputs.size());
        // Returning batch loss
        return batchLoss;
    }

    // A method to activate/deactivate training mode
    private void setIsInTrainingMode(boolean value) {
        for (NeuronLayer layer : this.layers)
            layer.setIsInTrainingMode(value);
    }

    /**
     * @return the networks learning rate
     */
    public double getLearningRate() {
        return learningRate;
    }

    /**
     * Sets the learning rate of the network
     * @param learningRate the new learning rate
     * @return return this instance of the class for chaining commands
     */
    public NeuralNetwork setLearningRate(double learningRate) {
        this.learningRate = learningRate;
        return this;
    }

    /**
     * @return the networks momentum
     */
    public double getMomentum() {
        return momentum;
    }

    /**
     * @return the average loss of the latest training session of the network, default value infinity
     */
    public double getAverageLoss() {
        return averageLoss;
    }

    /**
     * Sets the momentum of the network
     * @param momentum the new momentum
     * @return return this instance of the class for chaining commands
     */
    public NeuralNetwork setMomentum(double momentum) {
        this.momentum = momentum;
        return this;
    }

    /**
     * A method to set the batch size during training in stochastic gradient decent
     * @param batchSize the size of batches to be processed
     * @return return this instance of the class for chaining commands
     */
    public NeuralNetwork setBatchSize(int batchSize) {
        this.batchSize = batchSize;
        return this;
    }

    /**
     * A method to set the loss function the network will use
     * @param lossFunction the new loss function
     * @return return this instance of the class for chaining commands
     */
    public NeuralNetwork setLossFunction(LossFunctions lossFunction) {
        this.lossFunctions = lossFunction;
        return this;
    }

    /**
     * Tells the system if after each training epoch to print a message saying <br>
     * "[ Epoch i done, average loss: x ]"
     * @param value true/false whether to print or not
     */
    public void printStatusMessages(boolean value) {
        this.printStatusMessages = value;
    }

    /**
     * A method to breed to neural networks
     * @param other the network we breed with
     * @param mutate_chance the chance of mutations happening to the children
     * @param childrenCount the amount of children to create in the breeding process
     * @return a set including the new children
     */
    public Set<NeuralNetwork> breed(NeuralNetwork other, double mutate_chance, int childrenCount) {
        HashSet<NeuralNetwork> children = new HashSet<>();
        for (int i = 0; i < childrenCount; i++) {
            // Create a new child network
            NeuralNetwork child = new NeuralNetwork(this.layers.getFirst().getInputDimensions());
            // Breed every layer
            LinkedList<NeuronLayer> newLayers = new LinkedList<>();
            Iterator<NeuronLayer> otherIter = other.layers.iterator();
            Iterator<NeuronLayer> myIter = this.layers.iterator();
            while (otherIter.hasNext() && myIter.hasNext())
                newLayers.add(otherIter.next().breed(myIter.next(), mutate_chance, child));
            // Fill new network
            child.outputDimensions = newLayers.getLast().getOutputDimensions();
            child.layers = newLayers;
            child.learningRate = MLToolkit.RANDOM.nextBoolean() ? this.learningRate : other.learningRate;
            child.momentum = MLToolkit.RANDOM.nextBoolean() ? this.momentum : other.momentum;
            child.batchSize = MLToolkit.RANDOM.nextBoolean() ? this.batchSize : other.batchSize;
            // Add the child to the pool
            children.add(child);
        }
        return children;
    }

    /**
     * A method that writes this neural network to a file <br>
     * Note: this process saves everything expect what loss function you are using
     * @param filePath the path of the file to write to
     * @throws IOException if file patch is not valid
     */
    public void writeToFile(String filePath) throws IOException {
        // Write to file
        BufferedWriter writer = new BufferedWriter(new FileWriter(filePath));
        writer.write(this.toString());
        writer.close();
    }

    public static NeuralNetwork readFromFile(String filePath) throws IOException {
        // Attach reader to file
        BufferedReader reader = new BufferedReader(new FileReader(filePath));
        // Read IO shape and dimensions
        String[] IO = reader.readLine().split("\\|");
        String[] inShape = IO[1].split(",");
        String[] outShape = IO[3].split(",");
        int inputDimensions = Integer.parseInt(IO[0]);
        int[] inputShape = new int[3];
        int outputDimensions = Integer.parseInt(IO[2]);
        int[] outputShape = new int[3];
        for (int i = 0; i < 3; i++) {
            inputShape[i] = Integer.parseInt(inShape[i]);
            outputShape[i] = Integer.parseInt(outShape[i]);
        }
        // Read learningRate momentum batchSize
        String[] trainingParameters = reader.readLine().split("\\|");
        double learningRate = Double.parseDouble(trainingParameters[0]);
        double momentum = Double.parseDouble(trainingParameters[1]);
        int batchSize = Integer.parseInt(trainingParameters[2]);
        // The network being built
        NeuralNetwork output = new NeuralNetwork(inputShape);
        // Read all layers
        String encoding;
        LinkedList<NeuronLayer> layers = new LinkedList<>();
        while ((encoding = reader.readLine()) != null) {
            if (encoding.charAt(0) == 'b')
                layers.add(BatchNormalizationLayer.readFromString(encoding, output));
            else if (encoding.charAt(0) == 'd')
                layers.add(DenseLayer.readFromString(encoding, output));
            else if (encoding.charAt(0) == 'a')
                layers.add(AveragePoolingLayer.readFromString(encoding));
            else if (encoding.charAt(0) == 'm')
                layers.add(MaxPoolingLayer.readFromString(encoding));
            else if (encoding.charAt(0) == 'o')
                layers.add(ConvolutionLayer.readFromString(encoding, output));
            else if (encoding.charAt(0) == 'c')
                layers.add(DropoutLayer.readFromString(encoding));
        }
        // Build the network
        output.inputDimensions = inputDimensions;
        output.outputDimensions = outputDimensions;
        output.outputShape = outputShape;
        output.setBatchSize(batchSize).setLearningRate(learningRate).setMomentum(momentum);
        output.layers = layers;
        // Return the network
        return output;
    }

    @Override
    public String toString() {
        StringBuilder networkText = new StringBuilder();
        // Write: inputDimensions|width, height, depth|outputDimensions|width, height, depth
        networkText.append(this.inputDimensions).append("|").append(this.inputShape[0]).append(",").append(this.inputShape[1]).append(",").append(this.inputShape[2]).append("|");
        networkText.append(this.outputDimensions).append("|").append(this.outputShape[0]).append(",").append(this.outputShape[1]).append(",").append(this.outputShape[2]).append("|").append("\n");
        // Write: learningRate|momentum|batchSize
        networkText.append(this.learningRate).append("|").append(this.momentum).append("|").append(this.batchSize).append("\n");
        // Write all layer encodings
        for (NeuronLayer layer : this.layers)
            networkText.append(layer).append("\n");
        // Return the text
        return networkText.toString();
    }
}
