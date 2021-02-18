import java.util.LinkedList;
import java.util.ListIterator;

/**
 * A lstm block implementation in java
 */
public class LSTMBlock implements NeuronLayer {

    // Connection to master network
    private NeuralNetwork master;

    // flags
    private boolean isInTraining = false;

    // Forget gate (f)
    private Matrix Wf;
    private Matrix Uf;
    private Matrix Bf;

    // Forget gate momentum (f)
    private Matrix prevWfChange;
    private Matrix prevUfChange;
    private Matrix prevBfChange;

    // Input/Update gate (i)
    private Matrix Wi;
    private Matrix Ui;
    private Matrix Bi;

    // Input/Update gate momentum (i)
    private Matrix prevWiChange;
    private Matrix prevUiChange;
    private Matrix prevBiChange;

    // Output gate (o)
    private Matrix Wo;
    private Matrix Uo;
    private Matrix Bo;

    // Output gate momentum (o)
    private Matrix prevWoChange;
    private Matrix prevUoChange;
    private Matrix prevBoChange;

    // Cell input gate (ct/c'/cTag...)
    private Matrix Wct;
    private Matrix Uct;
    private Matrix Bct;

    // Cell input gate momentum (ct/c'/cTag...)
    private Matrix prevWctChange;
    private Matrix prevUctChange;
    private Matrix prevBctChange;

    // The list of data needed per time step.
    // Every time step is a array built in this fashion:
    // [0] hiddenState
    // [1] memoryBus
    // [2] lastOt
    // [3] lastInput
    // [4] previousMemoryBus
    // [5] Zf
    // [6] Zi
    // [7] Zo
    // [8] Zct
    private LinkedList<Matrix[]> timeSteps;

    /**
     * A constructor to build a new LSTM block
     * @param inputDimensions the length of the expected input vector
     * @param outputDimensions the length of the expected output vector
     * @param master the controller network
     */
    public LSTMBlock(int inputDimensions, int outputDimensions, NeuralNetwork master) {
        // Initialize the layer
        this.initLayer(inputDimensions, outputDimensions);
        // Fill weight matrices
        for (int i = 0; i < inputDimensions; i++)
            for (int j = 0; j < inputDimensions; j++) {
                // f
                this.Wf.set(i, j, MLToolkit.RANDOM.nextGaussian() / 3);
                this.Uf.set(i, j, MLToolkit.RANDOM.nextGaussian() / 3);
                // i
                this.Wi.set(i, j, MLToolkit.RANDOM.nextGaussian() / 3);
                this.Ui.set(i, j, MLToolkit.RANDOM.nextGaussian() / 3);
                // o
                this.Wo.set(i, j, MLToolkit.RANDOM.nextGaussian() / 3);
                this.Uo.set(i, j, MLToolkit.RANDOM.nextGaussian() / 3);
                // ct
                this.Wct.set(i, j, MLToolkit.RANDOM.nextGaussian() / 3);
                this.Uct.set(i, j, MLToolkit.RANDOM.nextGaussian() / 3);
            }
        // Set up connection to master network
        this.master = master;
    }

    // A private empty default constructor for breeding
    private LSTMBlock(int inputDimensions, int outputDimensions) {
        // Initialize the layer
        this.initLayer(inputDimensions, outputDimensions);
    }

    // A method to initialize a empty layer
    private void initLayer(int inputDimensions, int outputDimensions) {
        // Initialize Forget gate and its momentum variables
        this.Wf = new Matrix(inputDimensions, outputDimensions);
        this.Uf = new Matrix(inputDimensions, outputDimensions);
        this.Bf = new Matrix(1, outputDimensions);
        this.prevWfChange = new Matrix(inputDimensions, outputDimensions);
        this.prevUfChange = new Matrix(inputDimensions, outputDimensions);
        this.prevBfChange = new Matrix(1, outputDimensions);
        // Initialize Input/Update gate and its momentum variables
        this.Wi = new Matrix(inputDimensions, outputDimensions);
        this.Ui = new Matrix(inputDimensions, outputDimensions);
        this.Bi = new Matrix(1, outputDimensions);
        this.prevWiChange = new Matrix(inputDimensions, outputDimensions);
        this.prevUiChange = new Matrix(inputDimensions, outputDimensions);
        this.prevBiChange = new Matrix(1, outputDimensions);
        // Initialize Output gate and its momentum variables
        this.Wo = new Matrix(inputDimensions, outputDimensions);
        this.Uo = new Matrix(inputDimensions, outputDimensions);
        this.Bo = new Matrix(1, outputDimensions);
        this.prevWoChange = new Matrix(inputDimensions, outputDimensions);
        this.prevUoChange = new Matrix(inputDimensions, outputDimensions);
        this.prevBoChange = new Matrix(1, outputDimensions);
        // Initialize Cell input gate and its momentum variables
        this.Wct = new Matrix(inputDimensions, outputDimensions);
        this.Uct = new Matrix(inputDimensions, outputDimensions);
        this.Bct = new Matrix(1, outputDimensions);
        this.prevWctChange = new Matrix(inputDimensions, outputDimensions);
        this.prevUctChange = new Matrix(inputDimensions, outputDimensions);
        this.prevBctChange = new Matrix(1, outputDimensions);
        // Initialize time step list
        this.timeSteps = new LinkedList<>();
        this.timeSteps.add(new Matrix[9]);
        // Initialize internal states
        this.timeSteps.getLast()[0] = new Matrix(1, outputDimensions); // memoryBus
        this.timeSteps.getLast()[1] = new Matrix(1, outputDimensions); // hiddenState
    }

    @Override
    public Matrix forward(Matrix input) throws CloneNotSupportedException {
        // Create a new time step
        Matrix[] nextTimeStep = new Matrix[9];
        // Cache variables for training
        nextTimeStep[4] = this.timeSteps.getLast()[1]; // cache previous memoryBus
        nextTimeStep[3] = input; // save last input
        // f
        Matrix Wfx = new Matrix(1, this.Wf.getHeight());
        Matrix Ufh = new Matrix(1, this.Uf.getHeight());
        Matrix f_t = new Matrix(1, this.Wf.getHeight());
        // i
        Matrix Wix = new Matrix(1, this.Wi.getHeight());
        Matrix Uih = new Matrix(1, this.Ui.getHeight());
        Matrix i_t = new Matrix(1, this.Wi.getHeight());
        // o
        Matrix Wox = new Matrix(1, this.Wo.getHeight());
        Matrix Uoh = new Matrix(1, this.Uo.getHeight());
        Matrix o_t = new Matrix(1, this.Wo.getHeight());
        // ct
        Matrix Wctx = new Matrix(1, this.Wct.getHeight());
        Matrix Ucth = new Matrix(1, this.Uct.getHeight());
        Matrix ct_t = new Matrix(1, this.Wct.getHeight());
        // The z vectors
        Matrix Zf = new Matrix(1, this.Wf.getWidth());
        Matrix Zi = new Matrix(1, this.Wi.getWidth());
        Matrix Zo = new Matrix(1, this.Wo.getWidth());
        Matrix Zct = new Matrix(1, this.Wct.getWidth());
        // The new internal states
        Matrix newMemBus = new Matrix(1, this.Wf.getWidth());
        Matrix newHidden = new Matrix(1, this.Wf.getWidth());
        // Compute all values needed for the forward pass
        for (int i = 0; i < this.Wf.getWidth(); i++) {
            // Compute all dot products for the hidden units W * X, U * H
            for (int j = 0; j < this.Wf.getHeight(); j++) {
                // f
                Wfx.set(0, i, Wfx.get(0, i) + input.get(0, j) * this.Wf.get(j, i));
                Ufh.set(0, i, Ufh.get(0, i) + this.timeSteps.getLast()[0].get(0, j) * this.Uf.get(j, i));
                // i
                Wix.set(0, i, Wix.get(0, i) + input.get(0, j) * this.Wi.get(j, i));
                Uih.set(0, i, Uih.get(0, i) + this.timeSteps.getLast()[0].get(0, j) * this.Ui.get(j, i));
                // o
                Wox.set(0, i, Wox.get(0, i) + input.get(0, j) * this.Wo.get(j, i));
                Uoh.set(0, i, Uoh.get(0, i) + this.timeSteps.getLast()[0].get(0, j) * this.Uo.get(j, i));
                // ct
                Wctx.set(0, i, Wctx.get(0, i) + input.get(0, j) * this.Wct.get(j, i));
                Ucth.set(0, i, Ucth.get(0, i) + this.timeSteps.getLast()[0].get(0, j) * this.Uct.get(j, i));
            }
            // Compute the activation vectors of the hidden units (ft, it....)
            Zf.set(0, i, Wfx.get(0, i) + Ufh.get(0, i) + this.Bf.get(0, i));
            f_t.set(0, i, ActivationFunction.SIGMOID.apply(Zf.get(0, i)));
            Zi.set(0, i, Wix.get(0, i) + Uih.get(0, i) + this.Bi.get(0, i));
            i_t.set(0, i, ActivationFunction.SIGMOID.apply(Zi.get(0, i)));
            Zo.set(0, i, Wox.get(0, i) + Uoh.get(0, i) + this.Bo.get(0, i));
            o_t.set(0, i, ActivationFunction.SIGMOID.apply(Zo.get(0, i)));
            Zct.set(0, i, Wctx.get(0, i) + Ucth.get(0, i) + this.Bct.get(0, i));
            ct_t.set(0, i, ActivationFunction.TANH.apply(Zct.get(0, i)));
            // Save z vectors for training
            nextTimeStep[5] = Zf;
            nextTimeStep[6] = Zi;
            nextTimeStep[7] = Zo;
            nextTimeStep[8] = Zct;
            // Compute the new hidden state and memory bus value
            newMemBus.set(0, i, f_t.get(0, i) * this.timeSteps.getLast()[1].get(0, i) + i_t.get(0, i) * ct_t.get(0, i));
            newHidden.set(0, i, o_t.get(0, i) * ActivationFunction.TANH.apply(newMemBus.get(0, i)));
        }
        // Save new internal states
        nextTimeStep[0] = newHidden;
        nextTimeStep[1] = newMemBus;
        // Cache the variables needed for training
        nextTimeStep[2] = o_t;
        // Save time step for backpropagation if needed
        if (!this.isInTraining)
            this.timeSteps.clear();
        this.timeSteps.add(nextTimeStep);
        // Return the new hidden state
        return newHidden;
    }

    @Override
    public Matrix backpropagation(Matrix cost) throws Exception {
        // Variables to hold dx and the computed dx, dh gradients
        Matrix dx = null;
        Matrix[] gradient = {null, cost};
        // The iterator over the time step list
        ListIterator<Matrix[]> timeStepIter = this.timeSteps.listIterator(this.timeSteps.size());
        for (int i = 0; i < this.timeSteps.size() - 1; i++) {
            // Compute dh, and dx (if needed)
            Matrix[] timeStep = timeStepIter.previous();
            gradient = this.computeGradient(gradient[1], timeStep[2], timeStep[1], timeStep[4], timeStep[0], timeStep[5], timeStep[6], timeStep[7], timeStep[8], timeStep[3], dx == null);
            if (dx == null)
                dx = gradient[0];
        }
        // Return dx
        return dx;
    }

    // A method to compute the error of a given time step parameters
    private Matrix[] computeGradient(Matrix cost, Matrix lastOt, Matrix memoryBus, Matrix previousMemoryBus, Matrix hiddenState, Matrix Zf, Matrix Zi, Matrix Zo, Matrix Zct, Matrix lastInput, boolean computeDx) {
        // Get learning rate
        double learningRate = this.master.getLearningRate();
        // The needed partial derivatives
        Matrix dCt = new Matrix(1, cost.getHeight());
        Matrix dZf = new Matrix(1, cost.getHeight());
        Matrix dZi = new Matrix(1, cost.getHeight());
        Matrix dZo = new Matrix(1, cost.getHeight());
        Matrix dZct = new Matrix(1, cost.getHeight());
        Matrix dx = computeDx ? new Matrix(1, cost.getHeight()) : null;
        Matrix dh = new Matrix(1, cost.getHeight());
        for (int i = 0; i < this.getOutputDimensions(); i++) {
            // Compute the needed partial derivatives
            dCt.set(0, i, cost.get(0, i) * lastOt.get(0, i) * ActivationFunction.TANH.applyDerivative(memoryBus.get(0, i)));
            dZf.set(0, i, dCt.get(0, i) * previousMemoryBus.get(0, i) * ActivationFunction.SIGMOID.applyDerivative(Zf.get(0, i)));
            dZi.set(0, i, dCt.get(0, i) * ActivationFunction.TANH.apply(Zct.get(0, i)) * ActivationFunction.SIGMOID.applyDerivative(Zi.get(0, i)));
            dZo.set(0, i, cost.get(0, i) * ActivationFunction.TANH.apply(memoryBus.get(0, i)) * ActivationFunction.SIGMOID.applyDerivative(Zo.get(0, i)));
            dZct.set(0, i, dCt.get(0, i) * ActivationFunction.SIGMOID.apply(Zi.get(0, i)) * ActivationFunction.TANH.applyDerivative(Zct.get(0, i)));
            // Compute weight changes
            for (int j = 0; j < this.getInputDimensions(); j++) {
                // f
                this.prevWfChange.set(j, i, this.prevWfChange.get(j, i) - learningRate * dZf.get(0, i) * lastInput.get(0, j));
                this.prevUfChange.set(j, i, this.prevUfChange.get(j, i) - learningRate * dZf.get(0, i) * hiddenState.get(0, j));
                dh.set(0, i, dh.get(0, j) + dZf.get(0, i) * this.Uf.get(j, i));
                // i
                this.prevWiChange.set(j, i, this.prevWiChange.get(j, i) - learningRate * dZi.get(0, i) * lastInput.get(0, j));
                this.prevUiChange.set(j, i, this.prevUiChange.get(j, i) - learningRate * dZi.get(0, i) * hiddenState.get(0, j));
                dh.set(0, i, dh.get(0, j) + dZi.get(0, i) * this.Ui.get(j, i));
                // o
                this.prevWoChange.set(j, i, this.prevWoChange.get(j, i) - learningRate * dZo.get(0, i) * lastInput.get(0, j));
                this.prevUoChange.set(j, i, this.prevUoChange.get(j, i) - learningRate * dZo.get(0, i) * hiddenState.get(0, j));
                dh.set(0, i, dh.get(0, j) + dZo.get(0, i) * this.Uo.get(j, i));
                // ct
                this.prevWctChange.set(j, i, this.prevWctChange.get(j, i) - learningRate * dZct.get(0, i) * lastInput.get(0, j));
                this.prevUctChange.set(j, i, this.prevUctChange.get(j, i) - learningRate * dZct.get(0, i) * hiddenState.get(0, j));
                dh.set(0, i, dh.get(0, j) + dZct.get(0, i) * this.Uct.get(j, i));
                // dx
                if (computeDx) {
                    dx.set(0, j, dx.get(0, j) + dZct.get(0, i) * this.Wct.get(j, i));
                    dx.set(0, j, dx.get(0, j) + dZo.get(0, i) * this.Wo.get(j, i));
                    dx.set(0, j, dx.get(0, j) + dZi.get(0, i) * this.Wi.get(j, i));
                    dx.set(0, j, dx.get(0, j) + dZf.get(0, i) * this.Wf.get(j, i));
                }
            }
            // Compute bias changes
            this.prevBfChange.set(0, i, this.prevBfChange.get(0, i) - learningRate * dZf.get(0, i));
            this.prevBiChange.set(0, i, this.prevBiChange.get(0, i) - learningRate * dZi.get(0, i));
            this.prevBoChange.set(0, i, this.prevBoChange.get(0, i) - learningRate * dZo.get(0, i));
            this.prevBctChange.set(0, i, this.prevBctChange.get(0, i) - learningRate * dZct.get(0, i));
        }
        // Return the cost of the last input
        return new Matrix[] {dx, dh};
    }

    @Override
    public void commitGradientStep(int batchSize) {
        // Get momentum
        double momentum = this.master.getMomentum();
        // Update all weights and biases
        for (int i = 0; i < this.getInputDimensions(); i++) {
            for (int j = 0; j < this.getInputDimensions(); j++) {
                // Wf
                this.prevWfChange.set(i, j, this.prevWfChange.get(i, j) / batchSize);
                this.Wf.set(i, j, this.Wf.get(i, j) + this.prevWfChange.get(i, j));
                this.prevWfChange.set(i, j, this.prevWfChange.get(i, j) * momentum);
                // Uf
                this.prevUfChange.set(i, j, this.prevUfChange.get(i, j) / batchSize);
                this.Uf.set(i, j, this.Uf.get(i, j) + this.prevUfChange.get(i, j));
                this.prevUfChange.set(i, j, this.prevUfChange.get(i, j) * momentum);
                // Wi
                this.prevWiChange.set(i, j, this.prevWiChange.get(i, j) / batchSize);
                this.Wi.set(i, j, this.Wi.get(i, j) + this.prevWiChange.get(i, j));
                this.prevWiChange.set(i, j, this.prevWiChange.get(i, j) * momentum);
                // Ui
                this.prevUiChange.set(i, j, this.prevUiChange.get(i, j) / batchSize);
                this.Ui.set(i, j, this.Ui.get(i, j) + this.prevUiChange.get(i, j));
                this.prevUiChange.set(i, j, this.prevUiChange.get(i, j) * momentum);
                // Wo
                this.prevWoChange.set(i, j, this.prevWoChange.get(i, j) / batchSize);
                this.Wo.set(i, j, this.Wo.get(i, j) + this.prevWoChange.get(i, j));
                this.prevWoChange.set(i, j, this.prevWoChange.get(i, j) * momentum);
                // Uo
                this.prevUoChange.set(i, j, this.prevUoChange.get(i, j) / batchSize);
                this.Uo.set(i, j, this.Uo.get(i, j) + this.prevUoChange.get(i, j));
                this.prevUoChange.set(i, j, this.prevUoChange.get(i, j) * momentum);
                // Wct
                this.prevWctChange.set(i, j, this.prevWctChange.get(i, j) / batchSize);
                this.Wct.set(i, j, this.Wct.get(i, j) + this.prevWctChange.get(i, j));
                this.prevWctChange.set(i, j, this.prevWctChange.get(i, j) * momentum);
                // Uct
                this.prevUctChange.set(i, j, this.prevUctChange.get(i, j) / batchSize);
                this.Uct.set(i, j, this.Uct.get(i, j) + this.prevUctChange.get(i, j));
                this.prevUctChange.set(i, j, this.prevUctChange.get(i, j) * momentum);
            }
            // Bf
            this.prevBfChange.set(0, i, this.prevBfChange.get(0, i) / batchSize);
            this.Bf.set(0, i, this.Bf.get(0, i) + this.prevBfChange.get(0, i));
            this.prevBfChange.set(0, i, this.prevBfChange.get(0, i) * momentum);
            // Bi
            this.prevBiChange.set(0, i, this.prevBiChange.get(0, i) / batchSize);
            this.Bi.set(0, i, this.Bi.get(0, i) + this.prevBiChange.get(0, i));
            this.prevBiChange.set(0, i, this.prevBiChange.get(0, i) * momentum);
            // Bo
            this.prevBoChange.set(0, i, this.prevBoChange.get(0, i) / batchSize);
            this.Bo.set(0, i, this.Bo.get(0, i) + this.prevBoChange.get(0, i));
            this.prevBoChange.set(0, i, this.prevBoChange.get(0, i) * momentum);
            // Bct
            this.prevBctChange.set(0, i, this.prevBctChange.get(0, i) / batchSize);
            this.Bct.set(0, i, this.Bct.get(0, i) + this.prevBctChange.get(0, i));
            this.prevBctChange.set(0, i, this.prevBctChange.get(0, i) * momentum);
        }
        // Reset training parameters and internal states
        this.timeSteps = new LinkedList<>();
        this.timeSteps.add(new Matrix[9]);
        this.timeSteps.getLast()[0] = new Matrix(1, this.getOutputDimensions()); // memoryBus
        this.timeSteps.getLast()[1] = new Matrix(1, this.getOutputDimensions()); // hiddenState
    }

    @Override
    public LSTMBlock breed(NeuronLayer other, double mutate_chance, NeuralNetwork newMaster) {
        if (other instanceof LSTMBlock) {
            // Convert other layer to lstm block
            LSTMBlock father = (LSTMBlock) other;
            // Create the new block
            LSTMBlock kid = new LSTMBlock(this.getInputDimensions(), this.getOutputDimensions());
            // Breed weights and biases
            kid.Wf = MLToolkit.breedAndMutate(this.Wf, father.Wf, mutate_chance);
            kid.Uf = MLToolkit.breedAndMutate(this.Uf, father.Uf, mutate_chance);
            kid.Bf = MLToolkit.breedAndMutate(this.Bf, father.Bf, mutate_chance);
            kid.Wi = MLToolkit.breedAndMutate(this.Wi, father.Wi, mutate_chance);
            kid.Ui = MLToolkit.breedAndMutate(this.Ui, father.Ui, mutate_chance);
            kid.Bi = MLToolkit.breedAndMutate(this.Bi, father.Bi, mutate_chance);
            kid.Wo = MLToolkit.breedAndMutate(this.Wo, father.Wo, mutate_chance);
            kid.Uo = MLToolkit.breedAndMutate(this.Uo, father.Uo, mutate_chance);
            kid.Bo = MLToolkit.breedAndMutate(this.Bo, father.Bo, mutate_chance);
            kid.Wct = MLToolkit.breedAndMutate(this.Wct, father.Wct, mutate_chance);
            kid.Uct = MLToolkit.breedAndMutate(this.Uct, father.Uct, mutate_chance);
            kid.Bct = MLToolkit.breedAndMutate(this.Bct, father.Bct, mutate_chance);
            kid.master = newMaster;
            // Return the new layer
            return kid;
        }
        // Cant breed if both layers arent the same type
        return null;
    }

    @Override
    public void setIsInTrainingMode(boolean isInTraining) {
        this.isInTraining = isInTraining;
        // Reset training parameters and internal states
        this.timeSteps = new LinkedList<>();
        this.timeSteps.add(new Matrix[9]);
        this.timeSteps.getLast()[0] = new Matrix(1, this.getOutputDimensions()); // memoryBus
        this.timeSteps.getLast()[1] = new Matrix(1, this.getOutputDimensions()); // hiddenState
    }

    @Override
    public int getInputDimensions() {
        return this.Wf.getWidth();
    }

    @Override
    public int getOutputDimensions() {
        return this.Wf.getHeight();
    }

    // TODO to string

}
