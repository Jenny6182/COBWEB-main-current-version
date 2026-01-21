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

    private final SimulationInternals simulation;
    private final ActiveInferenceAgentParams params;
    private final Random random;

    // --- State Space Definitions ---
    // Hidden States (S):
    // 0: Safe/Empty
    // 1: Food Ahead
    // 2: Wall/Agent Ahead (Obstacle)
    private static final int NUM_STATES = 3;
    private static final int S_SAFE = 0;
    private static final int S_FOOD = 1;
    private static final int S_OBSTACLE = 2;

    // Observations (O):
    // 0: See Nothing
    // 1: See Food
    // 2: See Obstacle (Stone/Agent/Drop)
    // Note: Energy is handled separately as a continuous preference in C
    private static final int NUM_OBS = 3;
    private static final int O_NULL = 0;
    private static final int O_FOOD = 1;
    private static final int O_OBSTACLE = 2;

    // Actions (U):
    private static final int NUM_ACTIONS = 3;
    // Real actions mapped
    private static final int ACTION_MOVE = 0; // Forward
    private static final int ACTION_LEFT = 1;
    private static final int ACTION_RIGHT = 2;

    // --- Generative Model Parameters (The "Brain") ---

    // A: Likelihood P(o|s) [NUM_OBS][NUM_STATES]
    // Mapping Logic: S_SAFE->O_NULL, S_FOOD->O_FOOD, S_OBSTACLE->O_OBSTACLE
    private double[][] A;

    // B: Transitions P(s'|s, u) [NUM_STATES][NUM_STATES][NUM_ACTIONS]
    // This is LEARNING. We store counts (dirichlet parameters).
    private double[][][] b_concentration;
    private double[][][] B; // Expected B

    // C: Preferences P(o) (Log probabilities)
    private double[] C;

    // D: Prior P(s_0)
    private double[] D;

    // Current Belief Q(s)
    private double[] Qs;

    // Previous State/Action (for Learning)
    private double[] Qs_prev;
    private int action_prev = -1;

    public ActiveInferenceController(SimulationInternals sim, ActiveInferenceAgentParams params) {
        this.simulation = sim;
        this.params = params;
        this.random = sim.getRandom();

        initializeModel();
    }

    // Copy Constructor
    protected ActiveInferenceController(ActiveInferenceController parent) {
        this.simulation = parent.simulation;
        this.params = parent.params;
        this.random = simulation.getRandom();

        // Deep copy model state
        this.A = Matrices.addScalar(parent.A, 0); // Copy via math op
        this.C = parent.C.clone();
        this.D = parent.D.clone();

        // Deep copy learning params
        this.b_concentration = new double[NUM_STATES][NUM_STATES][NUM_ACTIONS];
        for (int s = 0; s < NUM_STATES; s++) {
            for (int sp = 0; sp < NUM_STATES; sp++) {
                this.b_concentration[s][sp] = parent.b_concentration[s][sp].clone();
            }
        }
        updateExpectations(); // Re-compute B from b_concentration

        this.Qs = parent.Qs.clone();
        this.Qs_prev = parent.Qs_prev != null ? parent.Qs_prev.clone() : null;
        this.action_prev = parent.action_prev;
    }

    private void initializeModel() {
        // 1. Initialize A (Likelihood) - Fixed "Identity" mapping for now, assuming
        // good vision
        // We could make this learnable too, but B-learning is the requested focus.
        A = new double[NUM_OBS][NUM_STATES];
        // S_SAFE -> O_NULL mostly
        A[O_NULL][S_SAFE] = 10.0;
        A[O_FOOD][S_SAFE] = 0.1;
        A[O_OBSTACLE][S_SAFE] = 0.1;
        // S_FOOD -> O_FOOD
        A[O_NULL][S_FOOD] = 0.1;
        A[O_FOOD][S_FOOD] = 10.0;
        A[O_OBSTACLE][S_FOOD] = 0.1;
        // S_OBSTACLE -> O_OBSTACLE
        A[O_NULL][S_OBSTACLE] = 0.1;
        A[O_FOOD][S_OBSTACLE] = 0.1;
        A[O_OBSTACLE][S_OBSTACLE] = 10.0;

        A = Matrices.dirichletExpectation(A); // Normalize

        // 2. Initialize B (Transitions) - Learning
        // Start with flat priors (add 1.0 everywhere) or slight bias towards stability
        b_concentration = new double[NUM_STATES][NUM_STATES][NUM_ACTIONS];
        for (int i = 0; i < NUM_STATES; i++) {
            for (int j = 0; j < NUM_STATES; j++) {
                for (int u = 0; u < NUM_ACTIONS; u++) {
                    b_concentration[i][j][u] = 1.0;
                    // Bias: predict state stays same by default (S_i -> S_i)
                    if (i == j)
                        b_concentration[i][j][u] += 2.0;
                }
            }
        }
        updateExpectations();

        // 3. Initialize C (Preferences)
        // We prefer Food observations!
        C = new double[NUM_OBS];
        C[O_NULL] = 0.0;
        C[O_FOOD] = params.foodPreference; // High preference
        C[O_OBSTACLE] = -2.0; // Dislike obstacles
        // Note: energy is handled as a separate 'modulator' or bias in planning

        // 4. Initialize D (Prior) and Qs
        D = Matrices.createUniform(NUM_STATES);
        Qs = D.clone();
    }

    private void updateExpectations() {
        B = new double[NUM_STATES][NUM_STATES][NUM_ACTIONS];
        for (int u = 0; u < NUM_ACTIONS; u++) {
            // Each B[...][u] is a transition matrix for action u
            // B[next][curr]
            for (int prev = 0; prev < NUM_STATES; prev++) {
                double[] col = new double[NUM_STATES];
                for (int next = 0; next < NUM_STATES; next++) {
                    col[next] = b_concentration[next][prev][u];
                }
                col = Matrices.dirichletExpectation(col); // Normalize column
                for (int next = 0; next < NUM_STATES; next++) {
                    B[next][prev][u] = col[next];
                }
            }
        }
    }

    public class AIInput implements ControllerInput {
        @Override
        public void mutate(float adjustmentStrength) {
        }
    }

    @Override
    public void controlAgent(Agent baseAgent, ControllerListener inputCallback) {
        ComplexAgent agent = (ComplexAgent) baseAgent;

        // --- 1. Persevere (Perception) ---
        // Get Observation
        SeeInfo seeInfo = agent.getState(VisionState.class).distanceLook();
        int obsIdx = mapObservation(seeInfo);

        // Variational Inference: Find Q(s_t) that explains o_t given prior D/B
        // For simplicity in this loop, we map Qs = A * o_t (approximate posterior)
        // Or technically: Q(s) \propto P(o|s) * P(s_prior)
        // P(s_prior) = B(action_prev) * Qs_prev

        double[] prior;
        if (Qs_prev != null && action_prev != -1) {
            // Predicted state from previous step
            prior = Matrices.multiply(BForAction(action_prev), Qs_prev);
        } else {
            prior = D;
        }

        // Likelihood from A
        double[] likelihood = new double[NUM_STATES];
        for (int s = 0; s < NUM_STATES; s++) {
            likelihood[s] = A[obsIdx][s]; // P(o|s)
        }

        // Posterior update
        double[] posterior = Matrices.multiplyElementwise(prior, likelihood);
        Qs = Matrices.normalize(posterior);

        // --- 2. Learning (Update B) ---
        if (Qs_prev != null && action_prev != -1) {
            // Update B counts: b[next][prev][u] += Q(next) * Q(prev)
            // Active Learning!
            double learningRate = 1.0;
            for (int next = 0; next < NUM_STATES; next++) {
                for (int prev = 0; prev < NUM_STATES; prev++) {
                    b_concentration[next][prev][action_prev] += learningRate * Qs[next] * Qs_prev[prev];
                }
            }
            // Re-normalize B occasionally or every step
            updateExpectations();
        }

        inputCallback.beforeControl(agent, new AIInput());

        // --- 3. Planning (Action Selection) ---
        // Calculate Expected Free Energy (G) for each policy (action)
        double[] G = new double[NUM_ACTIONS];

        for (int u = 0; u < NUM_ACTIONS; u++) {
            // Predict next state: Q(s'|u) = B(u) * Q(s)
            double[] predictedState = Matrices.multiply(BForAction(u), Qs);

            // Predict observation: Q(o'|u) = A * Q(s'|u)
            double[] predictedObs = Matrices.multiply(A, predictedState);

            // 1. Extrinsic (Preference realization): D_KL(Q(o') || P(o')) => - E[ln P(o)]
            // We want Q(o') to match C (preferred obs)
            // Simplified: G_extrinsic = - sum(Q(o') * C)
            double extrinsic = -Matrices.dot(predictedObs, C);

            // Add Energy consideration to preferences dynamically
            // If predicted state is SAFE, we might step more?
            // Actually, let's keep it simple: C includes food preference.
            // We can modulate C based on hunger here if we wanted complex homeostasis.

            // 2. Epistemic (Exploration): H(Q(s'|u)) or similar parameter uncertainty
            // "Active Learning" bonus: Actions that lead to unvisited B-states?
            // For now, simple entropy of predicted observations (Curiosity)
            double epistemic = -0.5 * Matrices.entropy(predictedObs); // -H means we want LOW entropy (prediction)?
            // Actually, for exploration we want param info gain.
            // Let's use a simplified exploration bonus for turns like before, or implicit
            // in B uncertainty.

            // Re-use simple "Turn" bonus from before as a baseline epistemic proxy
            if (u == ACTION_LEFT || u == ACTION_RIGHT) {
                G[u] -= 0.5; // Artificial epistemic bonus for looking around
            }

            G[u] = extrinsic + epistemic;
        }

        // --- 4. Selection ---
        // Select min G (most negative is best value) -> Softmax on -G
        // Or just ArgMin
        int selectedAction = 0;
        double minG = Double.MAX_VALUE;
        for (int u = 0; u < NUM_ACTIONS; u++) {
            // Add small noise for tie-breaking
            double val = G[u] + (random.nextDouble() * 0.1);
            if (val < minG) {
                minG = val;
                selectedAction = u;
            }
        }

        // --- 5. Execute ---
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
        // Dist needed?
        // Ideally state would include distance. For this simple 3-state model,
        // we only care if it's "Ahead" (Dist < X?).
        // Let's assume if we see it, it's relevant.

        switch (type) {
            case Environment.FLAG_FOOD:
                return O_FOOD;
            case Environment.FLAG_STONE:
            case Environment.FLAG_DROP:
            case Environment.FLAG_AGENT:
                return O_OBSTACLE;
            default:
                return O_NULL;
        }
    }

    @Override
    public Controller createChildAsexual() {
        return new ActiveInferenceController(this);
    }

    @Override
    public Controller createChildSexual(Controller parent2) {
        return new ActiveInferenceController(this);
    }
}
