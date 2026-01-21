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
 * A controller based on Active Inference principles.
 * It minimizes Expected Free Energy to make decisions.
 * 
 * Simplified Model:
 * - Observations: Energy Level, Visual Input
 * - Actions: Step, Turn Left, Turn Right
 * - Preferences: High Energy, Seeing Food
 * 
 * Reproduction is EXPLICITLY DISABLED.
 */
public class ActiveInferenceController implements Controller {

    private final SimulationInternals simulation;
    private final ActiveInferenceAgentParams params;
    private final Random random;

    // Action Constants
    private static final int ACTION_STAY = 0;
    private static final int ACTION_LEFT = 1;
    private static final int ACTION_RIGHT = 2;
    private static final int ACTION_STEP = 3;

    public ActiveInferenceController(SimulationInternals sim, ActiveInferenceAgentParams params) {
        this.simulation = sim;
        this.params = params;
        this.random = sim.getRandom();
    }

    // Copy Constructor
    protected ActiveInferenceController(ActiveInferenceController parent) {
        this.simulation = parent.simulation;
        this.params = parent.params;
        this.random = simulation.getRandom();
    }

    public class AIInput implements ControllerInput {
        @Override
        public void mutate(float adjustmentStrength) {
            // No mutation for this fixed active inference model
        }
    }

    @Override
    public void controlAgent(Agent baseAgent, ControllerListener inputCallback) {
        ComplexAgent agent = (ComplexAgent) baseAgent;

        // 1. Gather Observations (State)
        SeeInfo seeInfo = agent.getState(VisionState.class).distanceLook();
        int visualType = seeInfo.getType();
        int visualDist = seeInfo.getDist();
        double energyLevel = (double) agent.getEnergy() / 100.0; // Normalized approx (soft cap around 100 often)

        // Callback for logging/visualization (required by interface)
        inputCallback.beforeControl(agent, new AIInput());

        // 2. Calculate Expected Free Energy (G) for each action
        // G(u) = - E_q[ln P(o|s) + ln P(o)] (Preferred observations)
        // Simplified: Value = Predicted Preference Match

        double valueStep = calculateStepValue(visualType, visualDist, energyLevel);
        double valueLeft = calculateTurnValue(energyLevel); // Exploring
        double valueRight = calculateTurnValue(energyLevel); // Exploring

        // 3. Action Selection (Softmax or ArgMax)
        // Using ArgMax with simple noise for exploration if values are close

        int bestAction = ACTION_STAY;
        double bestValue = -Double.MAX_VALUE;

        // Evaluate Step
        if (valueStep > bestValue) {
            bestValue = valueStep;
            bestAction = ACTION_STEP;
        }

        // Evaluate Left
        if (valueLeft > bestValue) {
            bestValue = valueLeft;
            bestAction = ACTION_LEFT;
        }

        // Evaluate Right
        // Add small random tie-breaker for turns if they are equal
        if (valueRight > bestValue || (valueRight == bestValue && random.nextBoolean())) {
            bestValue = valueRight;
            bestAction = ACTION_RIGHT;
        }

        // 4. Execute Action & Disable Reproduction
        agent.setShouldReproduceAsex(false); // EXPLICITLY DISABLED
        // Also clear communication outputs just in case
        agent.setCommOutbox(0);

        switch (bestAction) {
            case ACTION_LEFT:
                agent.turnLeft();
                break;
            case ACTION_RIGHT:
                agent.turnRight();
                break;
            case ACTION_STEP:
                agent.step();
                break;
            default:
                // Do nothing
                break;
        }
    }

    /**
     * Estimates value of Stepping forward.
     * High value if Food is seen. Low value if Wall or Agent (collision).
     */
    private double calculateStepValue(int visualType, int visualDist, double currentEnergy) {
        double value = 0.0;

        // Transition Model & Preferences
        switch (visualType) {
            case Environment.FLAG_FOOD:
                // Expectation: Will eat food -> High Energy
                // Closer food is better (less steps to get there)
                value += params.foodPreference * 2.0;
                if (visualDist == 0) { // Right in front
                    value += params.energyPreference; // Massive reward for eating
                }
                break;

            case Environment.FLAG_STONE:
            case Environment.FLAG_DROP: // Waste also acts like a barrier usually
            case Environment.FLAG_AGENT: // Assuming bump agent is bad or neutral, but usually blocks
                // Expectation: Collision -> Energy Loss
                value -= 5.0; // Heavy penalty for bumping
                break;

            case 0: // Empty space (Environment.FLAG_VOID usually assumed 0 in SeeInfo if not
                    // defined)
            default:
                // Expectation: Small energy loss for stepping, but moving might find food
                // Epistemic value: moving reveals new observations usually
                value -= 0.1; // Cost of movement
                break;
        }

        return value;
    }

    /**
     * Estimates value of Turning.
     * Turning is an "Epistemic" action - it changes the field of view.
     * Simple heuristic: If we don't see food, turning is good.
     */
    private double calculateTurnValue(double currentEnergy) {
        // Base cost of turning
        double value = -0.05;

        // Epistemic Value:
        // If we assumed a uniform prior over what we might see after turning,
        // and we currently see nothing useful, turning has high potential information
        // gain.
        // For this simple agent, we represent this as a small constant "curiosity"
        // bonus.

        value += 0.2; // Exploration bonus

        return value;
    }

    @Override
    public Controller createChildAsexual() {
        return new ActiveInferenceController(this);
    }

    @Override
    public Controller createChildSexual(Controller parent2) {
        // Sexual reproduction disabled in logic, but standard factory method must
        // return valid object
        return new ActiveInferenceController(this);
    }
}
