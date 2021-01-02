/**
 * A general interface to describe a layer of neurons
 */
public interface NeuronLayer {

    /**
     * Forward a input through this layer
     * @param input the input to the layer
     * @return the output after this layer transform
     */
    Matrix forward(Matrix input) throws Exception;

    /**
     * Backwards propagates the cost of the activations of this layer
     * @param cost the cost of the example
     * @return the cost of the previous layer activations
     */
    Matrix backpropagation(Matrix cost) throws Exception;

    /**
     * Updates a network parameters according the gradients computed
     * during backwards propagation training cost
     */
    void commitGradientStep(int batchSize) throws Exception;

    /**
     * Breeds this layer with a second given layer
     * @param other the layer to breed with
     * @param newMaster the new master network
     * @return a new layer resulting from the breeding of both layers
     */
    NeuronLayer breed(NeuronLayer other, double mutate_chance, NeuralNetwork newMaster);

    /**
     * Activates/Deactivates the training mode flag in this layer
     * @param isInTraining true/false
     */
    void setIsInTrainingMode(boolean isInTraining);

    /**
     * @return returns the number of inputs the layer expects
     */
    int getInputDimensions();

    /**
     * @return returns the number of outputs the layer will produce
     */
    int getOutputDimensions();

}
