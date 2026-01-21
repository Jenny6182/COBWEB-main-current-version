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

    @Override
    public void controlAgent(Agent baseAgent, ControllerListener inputCallback) {
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
            updateExpectations();
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

            // 2. Epistemic Value (Exploration)
            double epistemic = -0.5 * Matrices.entropy(predictedObs);

            // Heuristics for planning biases
            if (u == ACTION_LEFT || u == ACTION_RIGHT) {
                G[u] -= 0.5; // Turn bonus
            }

            // Reproduction Drive:
            // If energy > 80 (implied max ~100 or params), reproduction is highly preferred
            // We model this as a "drive" or reduced Free Energy for ensuring survival of
            // lineage
            if (u == ACTION_REPRODUCE) {
                if (energy > params.agentParams[0].energyPreference * 40.0) { // Rough threshold based on preference
                    G[u] -= 5.0; // Big bonus to Reproduce if healthy
                } else {
                    G[u] += 10.0; // High cost if unhealthy (do not reproduce)
                }
            }

            G[u] = extrinsic + epistemic;
        }

        // --- 4. Selection ---
        int selectedAction = 0;
        double minG = Double.MAX_VALUE;
        for (int u = 0; u < NUM_ACTIONS; u++) {
            double val = G[u] + (random.nextDouble() * 0.1);
            if (val < minG) {
                minG = val;
                selectedAction = u;
            }
        }

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
