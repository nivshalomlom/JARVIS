import java.util.HashSet;

/**
 * A batch normalization layer implementation in java
 */
public class BatchNormalizationLayer implements NeuronLayer {

    private final static double NOISE = 0.00001;

    // The number of layer inputs
    private final int inputDimensions;

    // A connection to the master network
    private final NeuralNetwork master;

    // The scaling variables
    private double beta;
    private double gamma;

    // The accumulated mean and variance
    private Matrix runningMean;
    private Matrix runningVariance;
    private int samplesProcessed;

    // Training variables
    private double lastBetaChange;
    private double lastGammaChange;
    private Matrix lastInput;
    private Matrix lastX_hat;

    // A cache to store batch inputs for processing during gradient step
    private final HashSet<Matrix> batchCache;

    // Flags to indicate current mode
    private boolean isInTraining;

    /**
     * Creates a new batch normalization layer
     * @param inputDimensions the number of inputs to the layer
     */
    public BatchNormalizationLayer(int inputDimensions, NeuralNetwork master) {
        // Setting up a connection to the main network controller
        this.master = master;
        // Setting up the scaling variables
        this.beta = 0;
        this.gamma = 1;
        // Setting up training variables
        this.lastBetaChange = 0;
        this.lastGammaChange = 0;
        this.lastX_hat = new Matrix(1, inputDimensions);
        // Resetting flags
        this.isInTraining = false;
        // Setting up the batch cache
        this.batchCache = new HashSet<>();
        // Setting up the input dimensions
        this.inputDimensions = inputDimensions;
        // Setting up the mean to 0 and variance to 1
        this.runningMean = new Matrix(1, inputDimensions);
        this.runningVariance = new Matrix(1, inputDimensions).preformOnMatrix(x -> 1.0);
        this.samplesProcessed = 0;
    }

    @Override
    public Matrix forward(Matrix input) throws Exception {
        // Saving the latest input
        this.lastInput = input;
        // If is in training keep the input for later computations
        if (this.isInTraining)
            this.batchCache.add(input);
        // Normalize and shift the incoming input
        for (int i = 0; i < this.lastX_hat.getHeight(); i++) {
            // normalizedInput = (input - mean) / sqrt(variance)
            double normalizedInput = (input.get(0, i) - this.runningMean.get(0, i)) / Math.sqrt(this.runningVariance.get(0, i) + NOISE);
            // shiftedInput = gamma * normalizedInput + beta
            this.lastX_hat.set(0, i, this.gamma * normalizedInput + this.beta);
        }
        // Return the new value
        return this.lastX_hat;
    }

    @Override
    public Matrix backpropagation(Matrix cost) throws Exception {
        double learningRate = master.getLearningRate();
        // Computing dx_hat, dgamma, dbeta, dvariance, std_inv
        Matrix dx_hat = new Matrix(1, this.inputDimensions);
        Matrix x_mean = new Matrix(1, this.inputDimensions);
        Matrix std_inv = new Matrix(1, this.inputDimensions);
        double d_variance = 0;
        for (int i = 0; i < this.inputDimensions; i++) {
            // dx_hat = cost * gamma
            dx_hat.set(0, i, cost.get(0, i) * this.gamma);
            // dbeta = sum(cost)
            this.lastBetaChange -= cost.get(0, i) * learningRate;
            // dgammma = sum(cost * dx_hat)
            this.lastGammaChange -= cost.get(0, i) * this.lastX_hat.get(0, i) * learningRate;
            // x_mean = x - mean
            x_mean.set(0, i, this.lastInput.get(0, i) - this.runningMean.get(0, i));
            // d_variance = dx_hat * (x - mean) * -0.5 * (variance + NOISE) ^ -1.5
            d_variance += dx_hat.get(0, i) * x_mean.get(0, i) * -0.5 * Math.pow(this.runningVariance.get(0, i) + NOISE, -1.5);
            // std_inv = 1 / sqrt(variance + NOISE)
            std_inv.set(0, i, 1 / Math.sqrt(this.runningVariance.get(0, i) + NOISE));
        }
        // Compute dx_mean
        double dx_mean = d_variance * (-2 * x_mean.sum()) / this.inputDimensions;
        for (int i = 0; i < this.inputDimensions; i++)
            dx_mean += dx_hat.get(0, i) * -std_inv.get(0, i);
        // Compute dx
        Matrix dx = new Matrix(1, this.inputDimensions);
        for (int i = 0; i < this.inputDimensions; i++)
            dx.set(0, i, dx_hat.get(0, i) * std_inv.get(0, i) + ((2 * d_variance) / this.inputDimensions) * x_mean.get(0, i) + dx_mean / this.inputDimensions);
        // Send the value to the previous layer
        return dx;
    }

    @Override
    public void commitGradientStep(int batchSize) throws Exception {
        // Get the momentum from the master network
        double momentum = this.master.getMomentum();
        // Compute batch mean
        Matrix batch_mean = new Matrix(1, this.inputDimensions);
        for (Matrix sample : this.batchCache)
            batch_mean = batch_mean.addition(sample);
        batch_mean = batch_mean.divide(this.batchCache.size());
        // Compute batch variance
        Matrix batch_variance = new Matrix(1, this.inputDimensions);
        for (Matrix sample : this.batchCache)
            batch_variance = batch_variance.addition(sample.subtraction(batch_mean).preformOnMatrix(x -> x * x));
        batch_variance = batch_variance.divide(this.batchCache.size());
        if (this.samplesProcessed > 0) {
            // define n and m and to make code shorter
            double m = this.samplesProcessed, n = this.batchCache.size();
            // Update the running variance
            for (int i = 0; i < this.runningVariance.getHeight(); i++) {
                double first_half = (this.runningVariance.get(0, i) * m + batch_variance.get(0, i) * n) / (m + n);
                this.runningVariance.set(0, i, first_half + m * n * Math.pow((this.runningMean.get(0, i) - batch_mean.get(0, i)) / (m + n), 2));
            }
            // Update the running mean
            for (int i = 0; i < this.runningMean.getHeight(); i++)
                this.runningMean.set(0, i, (this.runningMean.get(0, i) * m + batch_mean.get(0, i) * n) / (m + n));
        }
        // If its the first batch being processed set batch mean and variance to running mean and variance
        else {
            this.runningMean = batch_mean;
            this.runningVariance = batch_variance;
        }
        // Increment samples processed
        this.samplesProcessed += this.batchCache.size();
        // Update gamma
        this.lastGammaChange /= batchSize;
        this.gamma += this.lastGammaChange;
        this.lastGammaChange *= momentum;
        // Update beta
        this.lastBetaChange /= batchSize;
        this.beta += this.lastBetaChange;
        this.lastBetaChange *= momentum;
        // Reset for next batch
        this.batchCache.clear();
    }

    @Override
    public NeuronLayer breed(NeuronLayer other, double mutate_chance, NeuralNetwork newMaster) {
        if (other instanceof BatchNormalizationLayer) {
            // Breed only if the 2 layers are the same type!
            BatchNormalizationLayer otherLayer = ((BatchNormalizationLayer) other);
            BatchNormalizationLayer newLayer = new BatchNormalizationLayer(this.inputDimensions, newMaster);
            newLayer.runningMean = new Matrix(1, this.inputDimensions);
            newLayer.runningVariance = new Matrix(1, this.inputDimensions);
            // Breed the mean and variance
            for (int i = 0; i < this.inputDimensions; i++) {
                newLayer.runningMean.set(0, i, MLToolkit.breedAndMutate(this.runningMean.get(0, i), otherLayer.runningMean.get(0, i), mutate_chance));
                newLayer.runningVariance.set(0, i, MLToolkit.breedAndMutate(this.runningVariance.get(0, i), otherLayer.runningVariance.get(0, i), mutate_chance));
            }
            // Breed beta and gamma
            newLayer.beta = MLToolkit.breedAndMutate(this.beta, otherLayer.beta, mutate_chance);
            newLayer.gamma = MLToolkit.breedAndMutate(this.gamma, otherLayer.gamma, mutate_chance);
            // Return the new layer
            return newLayer;
        }
        // If not the same type cant breed to return null
        else return null;
    }

    @Override
    public void setIsInTrainingMode(boolean isInTraining) {
        this.isInTraining = isInTraining;
    }

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
        // BatchNormalizationLayer text = b|beta,gamma|runningMean|runningVariance|samplesProcessed
        StringBuilder layerText = new StringBuilder("b|");
        // Append beta, gamma
        layerText.append(this.beta).append(",").append(this.gamma).append("|");
        // Append mean, variance, samplesProcessed
        layerText.append(this.runningMean.transpose()).append("|").append(this.runningVariance.transpose()).append("|").append(this.samplesProcessed);
        // Return the string value
        return layerText.toString();
    }

    // Creates a new batch norm layer from string encoding
    public static BatchNormalizationLayer readFromString(String layerEncoding, NeuralNetwork master) {
        // Check if encoding is of a batch norm layer
        if (layerEncoding.charAt(0) == 'b') {
            // Split encoding by |
            String[] split = layerEncoding.replace(" ", "").split("\\|");
            // Read beta and gamma
            String[] secondPart = split[1].split(",");
            double beta = Double.parseDouble(secondPart[0]);
            double gamma = Double.parseDouble(secondPart[1]);
            // Read mean
            String[] mean = split[2].substring(1, split[2].length() - 1).split(",");
            Matrix meanMatrix = new Matrix(1, mean.length);
            for (int i = 0; i < mean.length; i++)
                meanMatrix.set(0, i, Double.parseDouble(mean[i]));
            // Read variance
            String[] variance = split[3].substring(1, split[3].length() - 1).split(",");
            Matrix varianceMatrix = new Matrix(1, variance.length);
            for (int i = 0; i < variance.length; i++)
                varianceMatrix.set(0, i, Double.parseDouble(variance[i]));
            // Read samples processed
            int samplesProcessed = Integer.parseInt(split[4]);
            // Build the new layer
            BatchNormalizationLayer newLayer = new BatchNormalizationLayer(varianceMatrix.getHeight(), master);
            newLayer.runningVariance = varianceMatrix;
            newLayer.runningMean = meanMatrix;
            newLayer.samplesProcessed = samplesProcessed;
            newLayer.beta = beta;
            newLayer.gamma = gamma;
            // Return the new layer
            return newLayer;
        }
        // If not return null
        return null;
    }

}
