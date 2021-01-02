import java.util.Hashtable;
import java.util.function.Function;

public class ActivationFunction {

    // Constants to shorten code
    private static final Function<Double, Double> SIGMOID_FUNCTION = x -> 1 / (1 + Math.pow(Math.E, -x));
    private static final Function<Double, Double> TANH_FUNCTION = x -> (Math.pow(Math.E, x) - Math.pow(Math.E, -x)) / (Math.pow(Math.E, x) + Math.pow(Math.E, -x));

    // Constants to show available activation functions
    public static final ActivationFunction SIGMOID = new ActivationFunction(SIGMOID_FUNCTION, x -> SIGMOID_FUNCTION.apply(x) * (1 - SIGMOID_FUNCTION.apply(x)), "sigmoid");
    public static final ActivationFunction SOFT_PLUS = new ActivationFunction(x -> Math.log(1 + Math.pow(Math.E, x)), SIGMOID_FUNCTION, "soft_plus");
    public static final ActivationFunction TANH = new ActivationFunction(TANH_FUNCTION, x -> 1.0 - Math.pow(TANH_FUNCTION.apply(x), 2), "tanh");
    public static final ActivationFunction RELU = new ActivationFunction(x -> Math.max(0, x), x -> x <= 0 ? 0 : 1.0, "relu");
    public static final ActivationFunction BINARY_STEP = new ActivationFunction(x -> x < 0 ? 0 : 1.0, x -> x != 0 ? 0 : Double.NaN, "binary_step");
    public static final ActivationFunction LINEAR = new ActivationFunction(x -> x, x -> 1.0, "linear");
    public static final ActivationFunction SOFTMAX = new ActivationFunction(null, null, "softmax");

    // Create dictionary of all activation functions for quick access for reading from file
    protected static final Hashtable<String, ActivationFunction> activationDictionary = new Hashtable<>() {{
        put("sigmoid", SIGMOID);
        put("soft_plus", SOFT_PLUS);
        put("tanh", TANH);
        put("relu", RELU);
        put("binary_step", BINARY_STEP);
        put("linear", LINEAR);
        put("softmax", SOFTMAX);
    }};

    // Class specific variables
    private final Function<Double, Double> activation;
    private final Function<Double, Double> derivative;
    private final String name;

    // A private constructor
    private ActivationFunction(Function<Double, Double> activation, Function<Double, Double> derivative, String name) {
        this.activation = activation;
        this.derivative = derivative;
        this.name = name;
    }

    // A method to use the activation function
    public double apply(double value) {
        return this.activation.apply(value);
    }

    // A method to use the derivative of the activation function
    public double applyDerivative(double value) {
        return this.derivative.apply(value);
    }

    // A method to get the activation function's name
    public String getName() {
        return name;
    }
}
