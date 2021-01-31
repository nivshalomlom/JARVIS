/**
 * A lstm block implementation in java
 */
public class LSTMBlock implements NeuronLayer {

    // Connection to master network
    private NeuralNetwork master;

    // Internal state variables
    private Matrix hiddenState; // Ht
    private Matrix memoryBus; // Ct

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

    // Training variables cache
    private Matrix lastOt;
    private Matrix lastInput;
    private Matrix previousMemoryBus;
    private Matrix Zf;
    private Matrix Zi;
    private Matrix Zo;
    private Matrix Zct;

    /**
     * A constructor to build a new LSTM block
     * @param inputDimensions the length of the expected input vector
     * @param master the controller network
     */
    public LSTMBlock(int inputDimensions, NeuralNetwork master) {
        // Initialize the layer
        this.initLayer(inputDimensions);
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
    private LSTMBlock(int inputDimensions) {
        // Initialize the layer
        this.initLayer(inputDimensions);
    }

    // A method to initialize a empty layer
    private void initLayer(int inputDimensions) {
        // Initialize internal states
        this.hiddenState = new Matrix(1, inputDimensions);
        this.memoryBus = new Matrix(1, inputDimensions);
        // Initialize Forget gate and its momentum variables
        this.Wf = new Matrix(inputDimensions, inputDimensions);
        this.Uf = new Matrix(inputDimensions, inputDimensions);
        this.Bf = new Matrix(1, inputDimensions);
        this.prevWfChange = new Matrix(inputDimensions, inputDimensions);
        this.prevUfChange = new Matrix(inputDimensions, inputDimensions);
        this.prevBfChange = new Matrix(1, inputDimensions);
        // Initialize Input/Update gate and its momentum variables
        this.Wi = new Matrix(inputDimensions, inputDimensions);
        this.Ui = new Matrix(inputDimensions, inputDimensions);
        this.Bi = new Matrix(1, inputDimensions);
        this.prevWiChange = new Matrix(inputDimensions, inputDimensions);
        this.prevUiChange = new Matrix(inputDimensions, inputDimensions);
        this.prevBiChange = new Matrix(1, inputDimensions);
        // Initialize Output gate and its momentum variables
        this.Wo = new Matrix(inputDimensions, inputDimensions);
        this.Uo = new Matrix(inputDimensions, inputDimensions);
        this.Bo = new Matrix(1, inputDimensions);
        this.prevWoChange = new Matrix(inputDimensions, inputDimensions);
        this.prevUoChange = new Matrix(inputDimensions, inputDimensions);
        this.prevBoChange = new Matrix(1, inputDimensions);
        // Initialize Cell input gate and its momentum variables
        this.Wct = new Matrix(inputDimensions, inputDimensions);
        this.Uct = new Matrix(inputDimensions, inputDimensions);
        this.Bct = new Matrix(1, inputDimensions);
        this.prevWctChange = new Matrix(inputDimensions, inputDimensions);
        this.prevUctChange = new Matrix(inputDimensions, inputDimensions);
        this.prevBctChange = new Matrix(1, inputDimensions);
        // Initialize training variables cache
        this.Zf = new Matrix(1, inputDimensions);
        this.Zi = new Matrix(1, inputDimensions);
        this.Zo = new Matrix(1, inputDimensions);
        this.Zct = new Matrix(1, inputDimensions);
    }

    @Override
    public Matrix forward(Matrix input) throws CloneNotSupportedException {
        // Cache variables for training
        this.previousMemoryBus = this.memoryBus.clone();
        this.lastInput = input;
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
        // Compute all values needed for the forward pass
        for (int i = 0; i < this.Wf.getWidth(); i++) {
            // Compute all dot products for the hidden units W * X, U * H
            for (int j = 0; j < this.Wf.getHeight(); j++) {
                // f
                Wfx.set(0, i, Wfx.get(0, i) + input.get(0, j) * this.Wf.get(j, i));
                Ufh.set(0, i, Ufh.get(0, i) + this.hiddenState.get(0, j) * this.Uf.get(j, i));
                // i
                Wix.set(0, i, Wix.get(0, i) + input.get(0, j) * this.Wi.get(j, i));
                Uih.set(0, i, Uih.get(0, i) + this.hiddenState.get(0, j) * this.Ui.get(j, i));
                // o
                Wox.set(0, i, Wox.get(0, i) + input.get(0, j) * this.Wo.get(j, i));
                Uoh.set(0, i, Uoh.get(0, i) + this.hiddenState.get(0, j) * this.Uo.get(j, i));
                // ct
                Wctx.set(0, i, Wctx.get(0, i) + input.get(0, j) * this.Wct.get(j, i));
                Ucth.set(0, i, Ucth.get(0, i) + this.hiddenState.get(0, j) * this.Uct.get(j, i));
            }
            // Compute the activation vectors of the hidden units (ft, it....)
            this.Zf.set(0, i, Wfx.get(0, i) + Ufh.get(0, i) + this.Bf.get(0, i));
            f_t.set(0, i, ActivationFunction.SIGMOID.apply(this.Zf.get(0, i)));
            this.Zi.set(0, i, Wix.get(0, i) + Uih.get(0, i) + this.Bi.get(0, i));
            i_t.set(0, i, ActivationFunction.SIGMOID.apply(this.Zi.get(0, i)));
            this.Zo.set(0, i, Wox.get(0, i) + Uoh.get(0, i) + this.Bo.get(0, i));
            o_t.set(0, i, ActivationFunction.SIGMOID.apply(this.Zo.get(0, i)));
            this.Zct.set(0, i, Wctx.get(0, i) + Ucth.get(0, i) + this.Bct.get(0, i));
            ct_t.set(0, i, ActivationFunction.TANH.apply(this.Zct.get(0, i)));
            // Compute the new hidden state and memory bus value
            this.memoryBus.set(0, i, f_t.get(0, i) * this.memoryBus.get(0, i) + i_t.get(0, i) * ct_t.get(0, i));
            this.hiddenState.set(0, i, o_t.get(0, i) * ActivationFunction.TANH.apply(this.memoryBus.get(0, i)));
        }
        // Cache the variables needed for training
        this.lastOt = o_t;
        // Return the new hidden state
        return this.hiddenState;
    }

    @Override
    public Matrix backpropagation(Matrix cost) throws Exception {
        // Check a forward was called and all needed values are cached
        if (this.lastOt == null)
            throw new Exception("You need to run a forward call before a backwards one!");
        // Get learning rate
        double learningRate = this.master.getLearningRate();
        // The needed partial derivatives
        Matrix dCt = new Matrix(1, cost.getHeight());
        Matrix dZf = new Matrix(1, cost.getHeight());
        Matrix dZi = new Matrix(1, cost.getHeight());
        Matrix dZo = new Matrix(1, cost.getHeight());
        Matrix dZct = new Matrix(1, cost.getHeight());
        Matrix dx = new Matrix(1, cost.getHeight());
        for (int i = 0; i < dCt.getHeight(); i++) {
            // Compute the needed partial derivatives
            dCt.set(0, i, cost.get(0, i) * this.lastOt.get(0, i) * ActivationFunction.TANH.applyDerivative(this.memoryBus.get(0, i)));
            dZf.set(0, i, dCt.get(0, i) * this.previousMemoryBus.get(0, i) * ActivationFunction.SIGMOID.applyDerivative(this.Zf.get(0, i)));
            dZi.set(0, i, dCt.get(0, i) * ActivationFunction.TANH.apply(this.Zct.get(0, i)) * ActivationFunction.SIGMOID.applyDerivative(this.Zi.get(0, i)));
            dZo.set(0, i, cost.get(0, i) * ActivationFunction.TANH.apply(this.memoryBus.get(0, i)) * ActivationFunction.SIGMOID.applyDerivative(this.Zo.get(0, i)));
            dZct.set(0, i, dCt.get(0, i) * ActivationFunction.SIGMOID.apply(this.Zi.get(0, i)) * ActivationFunction.TANH.applyDerivative(this.Zct.get(0, i)));
            // Compute weight changes
            for (int j = 0; j < this.getInputDimensions(); j++) {
                // f
                this.prevWfChange.set(j, i, this.prevWfChange.get(j, i) - learningRate * dZf.get(0, i) * this.lastInput.get(0, j));
                this.prevUfChange.set(j, i, this.prevUfChange.get(j, i) - learningRate * dZf.get(0, i) * this.hiddenState.get(0, j));
                dx.set(0, j, dx.get(0, j) + dZf.get(0, i) * this.Wf.get(j, i));
                // i
                this.prevWiChange.set(j, i, this.prevWiChange.get(j, i) - learningRate * dZi.get(0, i) * this.lastInput.get(0, j));
                this.prevUiChange.set(j, i, this.prevUiChange.get(j, i) - learningRate * dZi.get(0, i) * this.hiddenState.get(0, j));
                dx.set(0, j, dx.get(0, j) + dZi.get(0, i) * this.Wi.get(j, i));
                // o
                this.prevWoChange.set(j, i, this.prevWoChange.get(j, i) - learningRate * dZo.get(0, i) * this.lastInput.get(0, j));
                this.prevUoChange.set(j, i, this.prevUoChange.get(j, i) - learningRate * dZo.get(0, i) * this.hiddenState.get(0, j));
                dx.set(0, j, dx.get(0, j) + dZo.get(0, i) * this.Wo.get(j, i));
                // ct
                this.prevWctChange.set(j, i, this.prevWctChange.get(j, i) - learningRate * dZct.get(0, i) * this.lastInput.get(0, j));
                this.prevUctChange.set(j, i, this.prevUctChange.get(j, i) - learningRate * dZct.get(0, i) * this.hiddenState.get(0, j));
                dx.set(0, j, dx.get(0, j) + dZct.get(0, i) * this.Wct.get(j, i));
            }
            // Compute bias changes
            this.prevBfChange.set(0, i, this.prevBfChange.get(0, i) - learningRate * dZf.get(0, i));
            this.prevBiChange.set(0, i, this.prevBiChange.get(0, i) - learningRate * dZi.get(0, i));
            this.prevBoChange.set(0, i, this.prevBoChange.get(0, i) - learningRate * dZo.get(0, i));
            this.prevBctChange.set(0, i, this.prevBctChange.get(0, i) - learningRate * dZct.get(0, i));
        }
        // Return the cost of the last input
        return dx;
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
    }

    @Override
    public LSTMBlock breed(NeuronLayer other, double mutate_chance, NeuralNetwork newMaster) {
        if (other instanceof LSTMBlock) {
            // Convert other layer to lstm block
            LSTMBlock father = (LSTMBlock) other;
            // Create the new block
            LSTMBlock kid = new LSTMBlock(this.getInputDimensions());
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
    public void setIsInTrainingMode(boolean isInTraining) {}

    @Override
    public int getInputDimensions() {
        return this.Wf.getWidth();
    }

    @Override
    public int getOutputDimensions() {
        return this.Wf.getWidth();
    }

}
