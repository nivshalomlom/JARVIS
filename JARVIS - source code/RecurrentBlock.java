import java.util.LinkedList;
import java.util.ListIterator;

/**
 * A simple recurrent block implementation in java <br>
 * output = activation(W * X + U * h + b)
 */
public class RecurrentBlock implements NeuronLayer {

    // The master network
    private NeuralNetwork master;

    // The activation function
    private ActivationFunction activationFunction;

    // Weights and biases
    private final Matrix W;
    private final Matrix U;
    private final Matrix b;

    // A list containing the values needed to back propagate through time
    // each element is a array containing:
    // [0] hidden state
    // [1] lastZ
    // [2] lastInput
    private LinkedList<Matrix[]> timeSteps;

    // Momentum variables
    private Matrix prevWChange;
    private Matrix prevUChange;
    private Matrix prevbChange;

    /**
     * A method to create a new recurrent block
     * @param inputDimensions the expected length of the input vector
     * @param outputDimensions the expected length of the output vector
     * @param activationFunction the activation function to apply to the output
     * @param master the master network
     */
    public RecurrentBlock(int inputDimensions, int outputDimensions, ActivationFunction activationFunction, NeuralNetwork master) {
        // Initialize weights and biases
        this.W = new Matrix(inputDimensions, outputDimensions);
        this.U = new Matrix(inputDimensions, outputDimensions);
        this.b = new Matrix(1, outputDimensions);
        // Fill weights
        for (int i = 0; i < inputDimensions; i++)
            for (int j = 0; j < outputDimensions; j++) {
                this.W.set(i, j, MLToolkit.RANDOM.nextGaussian() / 3);
                this.U.set(i, j, MLToolkit.RANDOM.nextGaussian() / 3);
            }
        // Initialize time step list
        this.timeSteps = new LinkedList<>();
        // Initialize training variables
        this.prevWChange = new Matrix(inputDimensions, outputDimensions);
        this.prevUChange = new Matrix(inputDimensions, outputDimensions);
        this.prevbChange = new Matrix(1, outputDimensions);
        // Initialize activation function
        this.activationFunction = activationFunction;
        // Initialize connection to master network
        this.master = master;
    }

    // Private constructor used for breeding
    private RecurrentBlock(Matrix W, Matrix U, Matrix b, ActivationFunction activationFunction, NeuralNetwork master) {
        // Initialize weights and biases
        this.W = W;
        this.U = U;
        this.b = b;
        // Initialize time step list
        this.timeSteps = new LinkedList<>();
        // Initialize training variables
        this.prevWChange = new Matrix(W.getWidth(), W.getHeight());
        this.prevUChange = new Matrix(W.getWidth(), W.getHeight());
        this.prevbChange = new Matrix(1, W.getHeight());
        // Initialize activation function
        this.activationFunction = activationFunction;
        // Initialize connection to master network
        this.master = master;
    }

    @Override
    public Matrix forward(Matrix input) throws Exception {
        Matrix[] newTimeStep = new Matrix[3];
        newTimeStep[1] = new Matrix(1, this.getOutputDimensions());
        // Cache last input
        newTimeStep[2] = input;
        // Create the output matrix
        Matrix output = new Matrix(1, this.W.getHeight());
        // Compute output = activation(W * X + U * h + b)
        for (int j = 0; j < output.getHeight(); j++) {
            for (int i = 0; i < this.W.getWidth(); i++) {
                // Compute W * X + U * h
                double hiddenStateValue = this.timeSteps.size() == 0 ? 0 : this.timeSteps.getLast()[0].get(0, j);
                output.set(0, j, output.get(0, j) + this.W.get(i, j) * input.get(0, j) + this.U.get(i, j) * hiddenStateValue);
            }
            // Add bias and preform activation
            newTimeStep[1].set(0, j, output.get(0, j) + this.b.get(0, j));
            if (!this.activationFunction.getName().equals("softmax"))
                output.set(0, j, this.activationFunction.apply(newTimeStep[1].get(0, j)));
        }
        // Return the output
        if (this.activationFunction.getName().equals("softmax"))
            output = MLToolkit.softmax(output);
        // Update hidden state
        newTimeStep[0] = output;
        // Return new value
        return output;
    }

    @Override
    public Matrix backpropagation(Matrix cost) throws Exception {
        // Get learning rate
        double learningRate = this.master.getLearningRate();
        // Compute partial derivatives
        Matrix dz = new Matrix(1, this.getOutputDimensions());
        Matrix dx = new Matrix(1, this.getInputDimensions());
        // Get time step iterator
        ListIterator<Matrix[]> timeStepIter = this.timeSteps.listIterator(this.timeSteps.size());
        // Flag to make sure we compute dx once
        boolean computedDx = false;
        while (timeStepIter.hasPrevious()) {
            Matrix[] timeStep = timeStepIter.previous();
            for (int j = 0; j < dz.getHeight(); j++) {
                dz.set(0, j, learningRate * cost.get(0, j) * this.activationFunction.applyDerivative(timeStep[1].get(0, j)));
                // Compute dw and du
                for (int i = 0; i < this.getInputDimensions(); i++) {
                    this.prevWChange.set(i, j, this.prevWChange.get(i, j) - learningRate * dz.get(0, j) * timeStep[2].get(0, i));
                    this.prevUChange.set(i, j, this.prevUChange.get(i, j) - learningRate * dz.get(0, j) * timeStep[0].get(0, i));
                    if (!computedDx)
                        dx.set(0, i, dx.get(0, i) + dz.get(0, j) * this.W.get(i, j));
                    cost.set(0, j, cost.get(0, i) + dz.get(0, j) * this.U.get(i, j));
                }
                // Compute db
                this.prevbChange.set(0, j, this.prevbChange.get(0, j) - dz.get(0, j));
            }
            // Toggle dx flag
            if (!computedDx)
                computedDx = true;
        }
        // Return the cost of the previous layer's activation
        return dx;
    }

    @Override
    public void commitGradientStep(int batchSize) throws Exception {
        // Get momentum
        double momentum = this.master.getMomentum();
        // Update parameters
        for (int j = 0; j < this.W.getHeight(); j++) {
            for (int i = 0; i < this.W.getWidth(); i++) {
                // W
                this.prevWChange.set(i, j, this.prevWChange.get(i, j) / batchSize);
                this.W.set(i, j, this.W.get(i, j) + this.prevWChange.get(i, j));
                this.prevWChange.set(i, j, this.prevWChange.get(i, j) * momentum);
                // U
                this.prevUChange.set(i, j, this.prevUChange.get(i, j) / batchSize);
                this.U.set(i, j, this.U.get(i, j) + this.prevUChange.get(i, j));
                this.prevUChange.set(i, j, this.prevUChange.get(i, j) * momentum);
            }
            // b
            this.prevbChange.set(0, j, this.prevbChange.get(0, j) / batchSize);
            this.W.set(0, j, this.W.get(0, j) + this.prevWChange.get(0, j));
            this.prevWChange.set(0, j, this.prevWChange.get(0, j) * momentum);
        }
        // Clear time steps
        this.timeSteps.clear();
    }

    @Override
    public NeuronLayer breed(NeuronLayer other, double mutate_chance, NeuralNetwork newMaster) {
        if (other instanceof RecurrentBlock) {
            RecurrentBlock otherBlock = (RecurrentBlock) other;
            Matrix newU = new Matrix(this.getInputDimensions(), this.getOutputDimensions());
            Matrix newW = new Matrix(this.getInputDimensions(), this.getOutputDimensions());
            Matrix newB = new Matrix(1, this.getOutputDimensions());
            for (int j = 0; j < this.getOutputDimensions(); j++) {
                for (int i = 0; i < this.getInputDimensions(); i++) {
                    newU.set(i, j, MLToolkit.breedAndMutate(this.U.get(i, j), otherBlock.U.get(i, j), mutate_chance));
                    newW.set(i, j, MLToolkit.breedAndMutate(this.W.get(i, j), otherBlock.W.get(i, j), mutate_chance));
                }
                newB.set(0, j, MLToolkit.breedAndMutate(this.b.get(0, j), otherBlock.b.get(0, j), mutate_chance));
            }
            return new RecurrentBlock(newW, newU, newB, MLToolkit.RANDOM.nextBoolean() ? this.activationFunction : otherBlock.activationFunction, newMaster);
        }
        return null;
    }

    @Override
    public void setIsInTrainingMode(boolean isInTraining) {

    }

    @Override
    public int getInputDimensions() {
        return this.W.getWidth();
    }

    @Override
    public int getOutputDimensions() {
        return this.W.getHeight();
    }
}
