import java.util.Random;

public class DropoutLayer implements NeuronLayer {

    // Constants
    private static final Random RANDOM = new Random();

    // Class specific variables
    private double keepProbability;
    private int inputDimensions;

    // Drop mapping for backpropagation
    private boolean[] dropMap;

    /**
     * Create a new dropout layer
     * @param keepProbability the chance of keeping a input
     * @param inputDimensions the number of inputs to the layer
     */
    public DropoutLayer(double keepProbability, int inputDimensions) {
        this.keepProbability = keepProbability;
        this.inputDimensions = inputDimensions;
        this.dropMap = new boolean[inputDimensions];
    }

    @Override
    public Matrix forward(Matrix input) throws Exception {
        // Apply drop out and save index's of dropped cells
        for (int i = 0; i < this.inputDimensions; i++) {
            if (MLToolkit.RANDOM.nextDouble() > this.keepProbability) {
                input.set(0, i, 0);
                this.dropMap[i] = true;
            }
            else {
                input.set(0, i, input.get(0, i) / (1 - this.keepProbability));
                dropMap[i] = false;
            }
        }
        // Return new input
        return input;
    }

    @Override
    public Matrix backpropagation(Matrix cost) throws Exception {
        // Apply the same drop that was applied forwards
        for (int i = 0; i < this.dropMap.length; i++) {
            if (this.dropMap[i])
                cost.set(0, i, 0);
            else cost.set(0, i, cost.get(0, i) / (1 - this.keepProbability));
        }
        // Return updated cost
        return cost;
    }

    @Override
    public void commitGradientStep(int batchSize) throws Exception {}

    @Override
    public NeuronLayer breed(NeuronLayer other, double mutate_chance, NeuralNetwork newMaster) {
        if (other instanceof DropoutLayer)
            return new DropoutLayer(MLToolkit.breedAndMutate(this.keepProbability, ((DropoutLayer)other).keepProbability, mutate_chance), this.inputDimensions);
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
        return this.inputDimensions;
    }

    @Override
    public String toString() {
        return "o|" + this.keepProbability + "|" + this.inputDimensions;
    }

    // A method to build a dropout layer from string encoding
    public static DropoutLayer readFromString(String encoding) {
        // Check if encoding is dropout
        if (encoding.charAt(0) == 'o') {
            String[] split = encoding.split("\\|");
            // Read keepProb
            double keepProbability = Double.parseDouble(split[1]);
            // Read input dimensions
            int inputDimensions = Integer.parseInt(split[2]);
            // Return new layer
            return new DropoutLayer(keepProbability, inputDimensions);
        }
        else return null;
    }

}
