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
 * Reproduction is EXPLICITLY DISABLED.
 */
public class ActiveInferenceController implements Controller {

    // Define Core model matrices
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

    private final SimulationInternals simulation;
    private final ActiveInferenceAgentParams params;
    private final Random random;

    // --- Old State Space Definitions ---
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
    // This doesn't work because doesn't reflect spatial structure
    // so will be replaced by other state encoding
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

    // ... (Generative Model Parameters remain same) ...

    // ... (Constructor remains same) ...

    // ... (initializeModel remains same) ...

    // ... (updateExpectations remains same) ...

    // ... (AIInput class) ...

    // Constructor of active inference controller, cannot rely on default
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

        for (int o = 0; o < NUM_OBS; o++) {
            for (int s = 0; s < NUM_STATES; s++) {
                A[o][s] = 0.0;
            }
        }

        // Mapping between hidden state and observation
        A[O_CLEAR][STATE_CLEAR] = 1.0;

        A[O_FOOD_AHEAD][STATE_FOOD_AHEAD] = 1.0;
        A[O_FOOD_LEFT][STATE_FOOD_LEFT] = 1.0;
        A[O_FOOD_RIGHT][STATE_FOOD_RIGHT] = 1.0;
        A[O_FOOD_BEHIND][STATE_FOOD_BEHIND] = 1.0;

        A[O_OBSTACLE_AHEAD][STATE_OBSTACLE_AHEAD] = 1.0;

        // Transition matrix B (next x prev x action), this dictates belief
        B = new double[NUM_STATES][NUM_STATES][NUM_ACTIONS];

        initializeTransitionModel();

        // Preferences (prefer food)
        //changed from (0, -5, 5) to (0, -15, 10)
//        C = new double[] {0.0, -15.0, 10.0};
        C = new double[]{
                0.0,   // CLEAR
                -20.0,  // FOOD_AHEAD (strongly preferred)
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

        // ======================
        // ACTION_MOVE
        // ======================

        B[STATE_CLEAR][STATE_CLEAR][ACTION_MOVE] = 1.0;
        B[STATE_CLEAR][STATE_FOOD_AHEAD][ACTION_MOVE] = 1.0;
        B[STATE_OBSTACLE_AHEAD][STATE_OBSTACLE_AHEAD][ACTION_MOVE] = 1.0;

        B[STATE_FOOD_LEFT][STATE_FOOD_LEFT][ACTION_MOVE] = 1.0;
        B[STATE_FOOD_RIGHT][STATE_FOOD_RIGHT][ACTION_MOVE] = 1.0;
        B[STATE_FOOD_BEHIND][STATE_FOOD_BEHIND][ACTION_MOVE] = 1.0;

        // ======================
        // ACTION_LEFT (rotate CCW)
        // ======================

        B[STATE_CLEAR][STATE_CLEAR][ACTION_LEFT] = 1.0;
        B[STATE_OBSTACLE_AHEAD][STATE_OBSTACLE_AHEAD][ACTION_LEFT] = 1.0;

        B[STATE_FOOD_RIGHT][STATE_FOOD_AHEAD][ACTION_LEFT] = 1.0;
        B[STATE_FOOD_BEHIND][STATE_FOOD_RIGHT][ACTION_LEFT] = 1.0;
        B[STATE_FOOD_LEFT][STATE_FOOD_BEHIND][ACTION_LEFT] = 1.0;
        B[STATE_FOOD_AHEAD][STATE_FOOD_LEFT][ACTION_LEFT] = 1.0;

        // ======================
        // ACTION_RIGHT (rotate CW)
        // ======================

        B[STATE_CLEAR][STATE_CLEAR][ACTION_RIGHT] = 1.0;
        B[STATE_OBSTACLE_AHEAD][STATE_OBSTACLE_AHEAD][ACTION_RIGHT] = 1.0;

        B[STATE_FOOD_LEFT][STATE_FOOD_AHEAD][ACTION_RIGHT] = 1.0;
        B[STATE_FOOD_BEHIND][STATE_FOOD_LEFT][ACTION_RIGHT] = 1.0;
        B[STATE_FOOD_RIGHT][STATE_FOOD_BEHIND][ACTION_RIGHT] = 1.0;
        B[STATE_FOOD_AHEAD][STATE_FOOD_RIGHT][ACTION_RIGHT] = 1.0;

        // ======================
        // ACTION_REPRODUCE
        // ======================

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



        // --- 1. Persevere (Perception) ---
        SeeInfo seeInfo = agent.getState(VisionState.class).distanceLook();
        int obsIdx = mapObservation(seeInfo);
        double energy = (double) agent.getEnergy();

        // Variational Inference (Belief Updating)
        double[] prior;
        if (Qs_prev != null && action_prev != -1) {
            prior = Matrices.multiply(BForAction(action_prev), Qs_prev);
        } else {
            prior = D;
        }

        double[] likelihood = new double[NUM_STATES];
        for (int s = 0; s < NUM_STATES; s++) {
            likelihood[s] = A[obsIdx][s];
        }

        double[] posterior = Matrices.multiplyElementwise(prior, likelihood);
        Qs = Matrices.normalize(posterior);

        // --- 2. Learning (Update B) ---
        if (Qs_prev != null && action_prev != -1) {
            double learningRate = 1.0;
            for (int next = 0; next < NUM_STATES; next++) {
                for (int prev = 0; prev < NUM_STATES; prev++) {
                    b_concentration[next][prev][action_prev] += learningRate * Qs[next] * Qs_prev[prev];
                }
            }
            // Re-normalize B occasionally
            // For performance, we might skip this every tick, but here we do it for
            // correctness
//            updateExpectations(); (was not defined) this needs to be defined
        }

        inputCallback.beforeControl(agent, new AIInput());

        // --- 3. Planning (Action Selection) ---
        double[] G = new double[NUM_ACTIONS];

        for (int u = 0; u < NUM_ACTIONS; u++) {
            // Predict outcomes
            double[] predictedState = Matrices.multiply(BForAction(u), Qs);
            double[] predictedObs = Matrices.multiply(A, predictedState);

            // 1. Extrinsic Value (Preferences)
            double extrinsic = -Matrices.dot(predictedObs, C);

            // 2. Epistemic Value (Exploration) (changed from -0.5 to -0.1 to try to discourage curiosity)
//            double epistemic = -0.1 * Matrices.entropy(predictedObs); // this was fixed, change to adjustable
            // new:
            double curiosityFactor = params.curiosity;  // from this agent's params
            double epistemic = -curiosityFactor * Matrices.entropy(predictedObs);
            // TODO: add a row in the ActiveInferencePanel so this parameter is adjustable

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
            // We model this as a "drive" or reduced Free Energy for ensuring survival of
            // lineage
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