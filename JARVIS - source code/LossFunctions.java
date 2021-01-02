import java.util.function.BiFunction;

/**
 * A class to handle all implemented loss functions
 */
public class LossFunctions {

    /**
     * The MSE error function: <br>
     * loss = sum(0.5 * (prediction - target) ^ 2) <br>
     * cost = (prediction - target) <br>
     */
    public static final LossFunctions MSE = new LossFunctions(
            // Loss function
            (target, prediction) -> {
                Matrix result = new Matrix(1, target.getHeight());
                for (int i = 0; i < result.getHeight(); i++)
                    result.set(0, i, 0.5 * Math.pow(prediction.get(0, i) - target.get(0, i), 2));
                return result.sum();
            },
            // Cost function
            (target, prediction) -> {
                Matrix result = new Matrix(1, target.getHeight());
                for (int i = 0; i < result.getHeight(); i++)
                    result.set(0, i, prediction.get(0, i) - target.get(0, i));
                return result;
            },
            // Name
            "MSE");

    /**
     * The MSE_2 error function: <br>
     * loss = sum((prediction - target) ^ 2) <br>
     * cost = 2 * (prediction - target) <br>
     */
    public static final LossFunctions MSE_2 = new LossFunctions(
            // Loss function
            (target, prediction) -> {
                Matrix result = new Matrix(1, target.getHeight());
                for (int i = 0; i < result.getHeight(); i++)
                    result.set(0, i, Math.pow(prediction.get(0, i) - target.get(0, i), 2));
                return result.sum();
            },
            // Cost function
            (target, prediction) -> {
                Matrix result = new Matrix(1, target.getHeight());
                for (int i = 0; i < result.getHeight(); i++)
                    result.set(0, i, 2 * (prediction.get(0, i) - target.get(0, i)));
                return result;
            },
            // Name
            "MSE_2");

    /**
     * The cross entropy loss error function <br>
     * a = prediction, E = target <br>
     * loss = sum(E * ln(a) + (1 - E) * ln(1 - a) <br>
     * cost = (a - E) / ((1 - a) * a)
     */
    public static final LossFunctions CROSS_ENTROPY_LOSS = new LossFunctions(
            // Loss function
            (target, prediction) -> {
                Matrix result = new Matrix(1, target.getHeight());
                for (int i = 0; i < result.getHeight(); i++)
                    result.set(0, i, target.get(0, i) * Math.log(prediction.get(0, i)) + (1 -target.get(0, i)) * Math.log(1 - prediction.get(0, i)));
                return result.sum();
            },
            // Cost function
            (target, prediction) -> {
                Matrix result = new Matrix(1, target.getHeight());
                for (int i = 0; i < result.getHeight(); i++)
                    result.set(0, i, (prediction.get(0, i) - target.get(0, i)) / ((1 - prediction.get(0, i) * prediction.get(0, i))));
                return result;
            },
            // Name
            "cross entropy cost");

    /**
     * The exponential loss error function <br>
     * loss = tao * e ^ (sum((prediction - target) ^ 2) / tao) <br>
     * cost = (2 / tao) * (prediction - target) * loss(target, prediction, tao) <br>
     * @param tao a number to be used in the function, choose to fit target behavior
     * @return a instance of exponential loss error function
     */
    public static LossFunctions EXPONENTIAL_COST(double tao) {
        // Loss function
        BiFunction<Matrix, Matrix, Double> C_EXP = (target, prediction) -> {
            double sum = 0;
            for (int i = 0; i < target.getHeight(); i++)
                sum += Math.pow(prediction.get(0, i) - target.get(0, i), 2);
            return tao * Math.pow(Math.E, sum / tao);
        };
        // Cost function
        BiFunction<Matrix, Matrix, Matrix> cost = (target, prediction) -> {
            Matrix result = new Matrix(0, target.getHeight());
            for (int i = 0; i < target.getHeight(); i++)
                result.set(0, i, (2 * (prediction.get(0, i) - target.get(0, i)) * C_EXP.apply(target, prediction)) / tao);
            return result;
        };
        // Create the function with the given tao
        return new LossFunctions(C_EXP, cost, "exponential loss");
    }

    /**
     * The hellinger distance error function <br>
     * a = prediction, E = target <br>
     * loss = (1 / sqrt(2)) * sum((sqrt(a) - sqrt(E)) ^ 2) <br>
     * cost = (sqrt(a) - sqrt(E)) / (sqrt(2) * sqrt(a))
     */
    public static final LossFunctions HELLINGER_DISTANCE = new LossFunctions(
            // Loss function
            (target, prediction) -> {
                double sum = 0;
                for (int i = 0; i < target.getHeight(); i++)
                    sum += Math.pow(Math.sqrt(prediction.get(0, i)) - Math.sqrt(target.get(0, i)), 2);
                return sum / Math.sqrt(2);
            },
            // Cost function
            (target, prediction) -> {
                double sqrt2 = Math.sqrt(2);
                Matrix result = new Matrix(1, target.getHeight());
                for (int i = 0; i < result.getHeight(); i++) {
                    double sqrtPrediction = Math.sqrt(prediction.get(0, i));
                    result.set(0, i, (sqrtPrediction - Math.sqrt(target.get(0, i))) / (sqrt2 * sqrtPrediction));
                }
                return result;
            },
            // Name
            "hellinger distance");

    // Class parameters
    private final BiFunction<Matrix, Matrix, Double> lossFunction;
    private final BiFunction<Matrix, Matrix, Matrix> costFunction;
    private final String name;

    /**
     * A private constructor to build a new loss function
     * @param lossFunction the loss function (Matrix, Matrix) -> double
     * @param costFunction the cost function (Matrix, Matrix) -> matrix
     * @param name the name of this function
     */
    private LossFunctions(BiFunction<Matrix, Matrix, Double> lossFunction, BiFunction<Matrix, Matrix, Matrix> costFunction, String name) {
        this.lossFunction = lossFunction;
        this.costFunction = costFunction;
        this.name = name;
    }

    /**
     * @param target the target output
     * @param prediction the network's prediction
     * @return the loss of this prediction
     */
    public Double computeLoss(Matrix target, Matrix prediction) {
        return this.lossFunction.apply(target, prediction);
    }

    /**
     * @param target the target output
     * @param prediction the network's prediction
     * @return the cost of this prediction regrading the loss (dL/dC)
     */
    public Matrix computeCost(Matrix target, Matrix prediction) {
        return this.costFunction.apply(target, prediction);
    }

    /**
     * @return the name of this loss function
     */
    public String getName() {
        return name;
    }
}
