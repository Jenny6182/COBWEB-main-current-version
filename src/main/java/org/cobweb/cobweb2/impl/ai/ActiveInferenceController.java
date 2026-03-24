package org.cobweb.cobweb2.impl.ai;

import org.cobweb.cobweb2.core.Agent;
import org.cobweb.cobweb2.core.Controller;
import org.cobweb.cobweb2.core.ControllerInput;
import org.cobweb.cobweb2.core.ControllerListener;
import org.cobweb.cobweb2.core.Environment;
import org.cobweb.cobweb2.core.SimulationInternals;
import org.cobweb.cobweb2.impl.ComplexAgent;
import org.cobweb.cobweb2.plugins.vision.SeeInfo;
import org.cobweb.cobweb2.plugins.vision.VisionState;

import java.util.Arrays;
import java.util.Random;

/**
 * A controller based on "Full" Active Inference principles.
 * 
 * Features:
 * - Internal Belief State (Hidden States).
 * - Generative Model (A, B, C, D matrices).
 * - Active Learning: Updates B-Matrix counts based on experience.
 * - Variational Inference: Minimizes Free Energy to infer states.
 * - Planning: Minimizes Expected Free Energy to select policies.
 *
 */
public class ActiveInferenceController implements Controller {

    // Define core model matrices
    // Generative model
    private double[][] A;              // Likelihood  (obs x state)
    private double[][][] B;            // Transition  (next x prev x action)
    private double[] C;                // Preferences (obs)
    private double[] D;                // Prior over states

    // Beliefs
    private double[] Qs;
    private double[] Qs_prev;
    private int action_prev = -1;

    // Learning
    private double[][][] b_concentration;

    // Other variables
    private final SimulationInternals simulation;
    private final ActiveInferenceAgentParams params;
    private final Random random;

    // --- Old State Space Definitions (Disregard this) ---
    // This doesn't work because doesn't reflect spatial structure
    // so will be replaced by other state encoding
    // Hidden States (S):
    // 0: Safe/Empty
    // 1: Food Ahead
    // 2: Wall/Agent Ahead (Obstacle)
    // Observations (O):
    // 0: See Nothing
    // 1: See Food
    // 2: See Obstacle (Stone/Agent/Drop)
    // Note: Energy is handled separately as a continuous preference in C

    // --- New Hidden States Encoding (S) ---
    static final int STATE_CLEAR = 0;
    static final int STATE_FOOD_AHEAD = 1;
    static final int STATE_FOOD_LEFT = 2;
    static final int STATE_FOOD_RIGHT = 3;
    static final int STATE_FOOD_BEHIND = 4;
    static final int STATE_OBSTACLE_AHEAD = 5;

    static final int NUM_STATES = 6;

    // --- New Observations (O) ---
    static final int O_CLEAR = 0;
    static final int O_FOOD_AHEAD = 1;
    static final int O_FOOD_LEFT = 2;
    static final int O_FOOD_RIGHT = 3;
    static final int O_FOOD_BEHIND = 4;
    static final int O_OBSTACLE_AHEAD = 5;

    static final int NUM_OBS = 6;

    // Actions (U):
    private static final int NUM_ACTIONS = 4;
    // Real actions mapped
    private static final int ACTION_MOVE = 0; // Forward
    private static final int ACTION_LEFT = 1;
    private static final int ACTION_RIGHT = 2;
    private static final int ACTION_REPRODUCE = 3;



    // Constructor of active inference controller
    public ActiveInferenceController(SimulationInternals simulation,
                                     ActiveInferenceAgentParams params) {
        this.simulation = simulation;
        this.params = params;
        this.random = new Random();

        initializeModel();

        System.out.println("Active Inference controller created!");
    }

    private void initializeModel() {
        // Likelihood matrix A (obs x state)
        A = new double[NUM_OBS][NUM_STATES];

        // Small base probability everywhere
        for (int o = 0; o < NUM_OBS; o++) {
            for (int s = 0; s < NUM_STATES; s++) {
                A[o][s] = 0.05;
            }
        }

        // Strong but non-deterministic mapping between hidden state and observation
        // this is to reflect "noise" in observations
        A[O_CLEAR][STATE_CLEAR] = 0.8;

        A[O_FOOD_AHEAD][STATE_FOOD_AHEAD] = 0.8;
        A[O_FOOD_LEFT][STATE_FOOD_LEFT] = 0.8;
        A[O_FOOD_RIGHT][STATE_FOOD_RIGHT] = 0.8;
        A[O_FOOD_BEHIND][STATE_FOOD_BEHIND] = 0.8;

        A[O_OBSTACLE_AHEAD][STATE_OBSTACLE_AHEAD] = 0.8;

        // Normalize each column
        for (int s = 0; s < NUM_STATES; s++) {
            double sum = 0.0;
            for (int o = 0; o < NUM_OBS; o++) sum += A[o][s];
            for (int o = 0; o < NUM_OBS; o++) A[o][s] /= sum;
        }

        // Transition matrix B (next x prev x action), this dictates belief
        B = new double[NUM_STATES][NUM_STATES][NUM_ACTIONS];

        initializeTransitionModel();

        // Preferences (prefer food)
        C = new double[]{
                0.0,   // CLEAR
                -20.0,  // FOOD_AHEAD (strongly preferred, this is when agent sees food in front of them)
                0.0,   // FOOD_LEFT
                0.0,   // FOOD_RIGHT
                0.0,   // FOOD_BEHIND
                10.0   // OBSTACLE_AHEAD (bad)
        };

        // Prior over states (uniform)
        D = new double[NUM_STATES];
        for (int i = 0; i < NUM_STATES; i++) {
            D[i] = 1.0 / NUM_STATES;
        }

        Qs = D.clone();

        // Learning concentration parameters
        b_concentration = new double[NUM_STATES][NUM_STATES][NUM_ACTIONS];
    }

    // Used to initialize matrix B (belief)
    private void initializeTransitionModel() {

        // Zero everything
        for (int u = 0; u < NUM_ACTIONS; u++) {
            for (int next = 0; next < NUM_STATES; next++) {
                for (int prev = 0; prev < NUM_STATES; prev++) {
                    B[next][prev][u] = 0.0;
                }
            }
        }

        // ------- ACTION_MOVE -------
        B[STATE_CLEAR][STATE_CLEAR][ACTION_MOVE] = 0.9;
        B[STATE_CLEAR][STATE_FOOD_AHEAD][ACTION_MOVE] = 1.0;
        B[STATE_OBSTACLE_AHEAD][STATE_OBSTACLE_AHEAD][ACTION_MOVE] = 1.0;

        B[STATE_FOOD_LEFT][STATE_FOOD_LEFT][ACTION_MOVE] = 1.0;
        B[STATE_FOOD_RIGHT][STATE_FOOD_RIGHT][ACTION_MOVE] = 1.0;
        B[STATE_FOOD_BEHIND][STATE_FOOD_BEHIND][ACTION_MOVE] = 1.0;

        // --- Add stochasticity to movement (smoothing) ---
        // this controls the amount of randomness
        for (int u = 0; u < NUM_ACTIONS; u++) {
            for (int prev = 0; prev < NUM_STATES; prev++) {
                double sum = 0.0;
                // Add a small amount of noise for stochasticity
                for (int next = 0; next < NUM_STATES; next++) {
                    B[next][prev][u] += random.nextDouble() * 0.02; // Control the noise (less noise)
                    sum += B[next][prev][u];
                }

                // Normalize so the probabilities sum to 1
                for (int next = 0; next < NUM_STATES; next++) {
                    B[next][prev][u] /= sum;
                }
            }
        }


        // ------- ACTION_LEFT (rotate CCW) -------
        B[STATE_CLEAR][STATE_CLEAR][ACTION_LEFT] = 1.0;
        B[STATE_OBSTACLE_AHEAD][STATE_OBSTACLE_AHEAD][ACTION_LEFT] = 1.0;

        B[STATE_FOOD_RIGHT][STATE_FOOD_AHEAD][ACTION_LEFT] = 1.0;
        B[STATE_FOOD_BEHIND][STATE_FOOD_RIGHT][ACTION_LEFT] = 1.0;
        B[STATE_FOOD_LEFT][STATE_FOOD_BEHIND][ACTION_LEFT] = 1.0;
        B[STATE_FOOD_AHEAD][STATE_FOOD_LEFT][ACTION_LEFT] = 1.0;


        // ------- ACTION_RIGHT (rotate CW) -------
        B[STATE_CLEAR][STATE_CLEAR][ACTION_RIGHT] = 1.0;
        B[STATE_OBSTACLE_AHEAD][STATE_OBSTACLE_AHEAD][ACTION_RIGHT] = 1.0;

        B[STATE_FOOD_LEFT][STATE_FOOD_AHEAD][ACTION_RIGHT] = 1.0;
        B[STATE_FOOD_BEHIND][STATE_FOOD_LEFT][ACTION_RIGHT] = 1.0;
        B[STATE_FOOD_RIGHT][STATE_FOOD_BEHIND][ACTION_RIGHT] = 1.0;
        B[STATE_FOOD_AHEAD][STATE_FOOD_RIGHT][ACTION_RIGHT] = 1.0;


        // ------- ACTION_REPRODUCE -------
        for (int s = 0; s < NUM_STATES; s++) {
            B[s][s][ACTION_REPRODUCE] = 1.0;
        }
    }


    public class AIInput implements ControllerInput {

        @Override
        public void mutate(float adjustmentStrength) {
            // For now, do nothing.
            // Later you could mutate A, B, C, etc.
        }
    }

    // Constructor for child (reproduction)
//    public ActiveInferenceController(ActiveInferenceController parent) {
//        this.simulation = parent.simulation;
//        this.params = parent.params;
//        this.random = new Random();
//
//        // Copy model state
//        this.A = parent.A;
//        this.B = parent.B;
//        this.C = parent.C;
//        this.D = parent.D;
//
//        this.b_concentration = parent.b_concentration;
//    }


    @Override
    public void controlAgent(Agent baseAgent, ControllerListener inputCallback) {

        System.out.println("AI tick");
        ComplexAgent agent = (ComplexAgent) baseAgent;

        // --- 1. Perception ---
        SeeInfo seeInfo = agent.getState(VisionState.class).distanceLook();
        int obsIdx = mapObservation(seeInfo);
        double energy = (double) agent.getEnergy();

        // Variational Inference (Belief Updating)
        double[] prior;
        if (Qs_prev != null && action_prev != -1) {
            prior = Matrices.multiply(BForAction(action_prev), Qs_prev);  // B * Qs_prev for prediction
        } else {
            prior = D;  // Use uniform prior if no previous belief exists
        }

        // Calculate the likelihood based on the observation
        double[] likelihood = new double[NUM_STATES];
        for (int s = 0; s < NUM_STATES; s++) {
            likelihood[s] = A[obsIdx][s];  // Likelihood of seeing the observation given the state
        }

        // Bayesian update: posterior = prior * likelihood
        double[] posterior = new double[NUM_STATES];
        for (int s = 0; s < NUM_STATES; s++) {
            posterior[s] = prior[s] * likelihood[s];  // Element-wise multiplication
        }

        // Normalize the posterior to ensure it sums to 1 (this makes it a valid probability distribution)
        Qs = Matrices.normalize(posterior);


        // --- 2. Dynamic Curiosity Adjustment (if curiosityFixed is 0) ---
        if (!params.curiosityFixed) { // do curiosity adjustment if curiosity is NOT fixed (so curiosityFixed = False)
            // Calculate entropy of the belief state (Qs)
            double uncertainty = Matrices.entropy(Qs);  // Entropy of the belief state

            // If uncertainty is high, increase curiosity to encourage exploration
            if (uncertainty > 0.7) {  // High entropy -> high uncertainty
                double newCuriosity = Math.min(params.curiosity + 0.05, 1.0);  // Increase curiosity but cap at 1.0
                params.curiosity = newCuriosity;  // Update the curiosity in the controller's params
            }

            // If uncertainty is low, decrease curiosity to focus more on exploiting known knowledge
            else if (uncertainty < 0.3) {  // Low entropy -> high certainty
                double newCuriosity = Math.max(params.curiosity - 0.05, 0.0);  // Decrease curiosity but cap at 0.0
                params.curiosity = newCuriosity;  // Update the curiosity in the controller's params
            }
        }

        System.out.println("Curiosity: " + params.curiosity);

        // --- 3. Learning (updating B) ---
        if (Qs_prev != null && action_prev != -1) {
            double learningRate = 1.0;
            for (int next = 0; next < NUM_STATES; next++) {
                for (int prev = 0; prev < NUM_STATES; prev++) {
                    b_concentration[next][prev][action_prev] += learningRate * Qs[next] * Qs_prev[prev];
                }
            }
        }

        inputCallback.beforeControl(agent, new AIInput());

        // --- 4. Planning (Action Selection) ---
        double[] G = new double[NUM_ACTIONS];

        for (int u = 0; u < NUM_ACTIONS; u++) {
            // Predict outcomes
            double[] predictedState = Matrices.multiply(BForAction(u), Qs);
            double[] predictedObs = Matrices.multiply(A, predictedState);

            // 1. Extrinsic Value (Preferences)
            double extrinsic = -Matrices.dot(predictedObs, C);

            // 2. Epistemic Value (Exploration)
            double curiosityFactor = params.curiosity;  // using this agent's params, we adjust curiosity
            double epistemic = -curiosityFactor * Matrices.entropy(predictedObs);

            System.out.println("extrinsic: " + extrinsic);
            System.out.println("epistemic: " + epistemic);

            // Heuristics for planning biases
            G[u] = extrinsic + epistemic;

            if (u == ACTION_LEFT || u == ACTION_RIGHT) {
                G[u] += 0.2; // Turn bonus;
                // changed from -= 0.5 to +=0.2
            }

            // Reproduction Drive:
            // If energy > 80 (implied max ~100 or params), reproduction is highly preferred
            // We model this as a "drive" or reduced Free Energy for ensuring survival of lineage
            if (u == ACTION_REPRODUCE) {
                // Fix: there is no agentParams in params
                if (energy > params.energyPreference * 40.0) { // Rough threshold based on preference
                    G[u] -= 5.0; // Big bonus to Reproduce if healthy
                } else {
                    G[u] += 10.0; // High cost if unhealthy (do not reproduce)
                }
                System.out.println("G[" + u + "] = " + G[u]);
            }
            System.out.println(Arrays.toString(predictedState));
        }


        // --- 4. Selection (Softmax) ---
        double temperature = 1.0;  // try 0.5 if too random

        double sum = 0.0;
        double[] probs = new double[NUM_ACTIONS];

        // Convert G (cost) to probabilities
        for (int u = 0; u < NUM_ACTIONS; u++) {
            probs[u] = Math.exp(-G[u] / temperature);
            sum += probs[u];
        }

        // Normalize
        for (int u = 0; u < NUM_ACTIONS; u++) {
            probs[u] /= sum;
        }

        // Sample
        double r = random.nextDouble();
        double cumulative = 0.0;
        int selectedAction = 0;

        for (int u = 0; u < NUM_ACTIONS; u++) {
            cumulative += probs[u];
            if (r < cumulative) {
                selectedAction = u;
                break;
            }
        }

        // for debugging
        System.out.println("Selected action: " + selectedAction);


        // --- 5. Execute ---
        // Reset flags
        agent.setShouldReproduceAsex(false);
        agent.setCommOutbox(0);

        Qs_prev = Qs.clone();
        action_prev = selectedAction;

        switch (selectedAction) {
            case ACTION_LEFT:
                agent.turnLeft();
                break;
            case ACTION_RIGHT:
                agent.turnRight();
                break;
            case ACTION_REPRODUCE:
                agent.setShouldReproduceAsex(true); // ENABLE REPRODUCTION
                break;
            case ACTION_MOVE:
            default:
                agent.step();
                break;
        }
    }


    private double[][] BForAction(int u) {
        double[][] Bu = new double[NUM_STATES][NUM_STATES];
        for (int next = 0; next < NUM_STATES; next++) {
            for (int prev = 0; prev < NUM_STATES; prev++) {
                Bu[next][prev] = B[next][prev][u];
            }
        }
        return Bu;
    }


    private int mapObservation(SeeInfo see) {

        int type = see.getType();
        int dx = see.getDx();
        int dy = see.getDy();

        if (type == Environment.FLAG_FOOD) {

            if (dx == 0 && dy < 0) return O_FOOD_AHEAD;
            if (dx < 0 && dy == 0) return O_FOOD_LEFT;
            if (dx > 0 && dy == 0) return O_FOOD_RIGHT;
            if (dx == 0 && dy > 0) return O_FOOD_BEHIND;

        }

        if (type == Environment.FLAG_STONE ||
                type == Environment.FLAG_DROP ||
                type == Environment.FLAG_AGENT) {

            if (dx == 0 && dy < 0) return O_OBSTACLE_AHEAD;
        }

        return O_CLEAR;
    }

    @Override
    public Controller createChildAsexual() {
        return new ActiveInferenceController(simulation, params);
    }

    @Override
    public Controller createChildSexual(Controller parent2) {
        return new ActiveInferenceController(simulation, params);
    }

    protected ActiveInferenceController(ActiveInferenceController parent) {
        this.simulation = parent.simulation;
        this.params = parent.params;
        this.random = new Random();

        initializeModel(); // or copy matrices if you want inheritance
    }

}