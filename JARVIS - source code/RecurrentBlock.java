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
    private Matrix W;
    private Matrix U;
    private Matrix b;

    // The internal hidden state
    private Matrix hiddenState;

    // Backpropagation cache
    private Matrix lastZ;
    private Matrix lastInput;

    // Momentum variables
    private Matrix prevWChange;
    private Matrix prevUChange;
    private Matrix prevbChange;

    @Override
    public Matrix forward(Matrix input) throws Exception {
        // Cache last input
        this.lastInput = input;
        // Create the output matrix
        Matrix output = new Matrix(1, this.W.getHeight());
        // Compute output = activation(W * X + U * h + b)
        for (int j = 0; j < output.getHeight(); j++) {
            for (int i = 0; i < this.W.getWidth(); i++)
                // Compute W * X + U * h
                output.set(0, j, output.get(0, j) + this.W.get(i, j) * input.get(0, j) + this.U.get(i, j) * this.hiddenState.get(0, j));
            // Add bias and preform activation
            this.lastZ.set(0, j, output.get(0, j) + this.b.get(0, j));
            if (!this.activationFunction.getName().equals("softmax"))
                output.set(0, j, this.activationFunction.apply(lastZ.get(0, j)));
        }
        // Return the output
        if (this.activationFunction.getName().equals("softmax"))
            return MLToolkit.softmax(output);
        return output;
    }

    @Override
    public Matrix backpropagation(Matrix cost) throws Exception {
        // Get learning rate
        double learningRate = this.master.getLearningRate();
        // Compute partial derivatives
        Matrix dz = new Matrix(1, this.getOutputDimensions());
        Matrix dx = new Matrix(1, this.getInputDimensions());
        for (int j = 0; j < dz.getHeight(); j++) {
            dz.set(0, j, learningRate * cost.get(0, j) * this.activationFunction.applyDerivative(this.lastZ.get(0, j)));
            // Compute dw and du
            for (int i = 0; i < this.getInputDimensions(); i++) {
                this.prevWChange.set(i, j, this.prevWChange.get(i, j) - learningRate * dz.get(0, j) * this.lastInput.get(0, i));
                this.prevUChange.set(i, j, this.prevUChange.get(i, j) - learningRate * dz.get(0, j) * this.hiddenState.get(0, i));
                dx.set(0, i, dx.get(0, i) + dz.get(0, j) * this.W.get(i, j));
            }
            // Compute db
            this.prevbChange.set(0, j, this.prevbChange.get(0, j) - dz.get(0, j));
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
    }

    @Override
    public NeuronLayer breed(NeuronLayer other, double mutate_chance, NeuralNetwork newMaster) {
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
