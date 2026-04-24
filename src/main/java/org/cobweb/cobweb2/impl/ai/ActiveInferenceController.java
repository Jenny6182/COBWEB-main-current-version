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
import java.util.ArrayList;   // ADD
import java.util.List;        // ADD
import java.util.Random;

public class ActiveInferenceController implements Controller {

    private double[][] A;
    private double[][][] B;
    private double[] C;
    private double[] D;

    private double[] Qs;
    private double[] Qs_prev;
    private int action_prev = -1;

    private double[][][] b_concentration;

    private final SimulationInternals simulation;
    private final ActiveInferenceAgentParams params;
    private final Random random;

    // ----------------------------------------------------------------
    // ADD: shared registry of all agents' snapshots this tick.
    // ----------------------------------------------------------------

    private static final Object registryLock = new Object();
    private static List<ClusterUtils.AgentSnapshot> agentRegistry =
            new ArrayList<ClusterUtils.AgentSnapshot>();
    private static int registryTimestep = -1;

    /** Cluster radius used for excursion detection. */
    private static final double CLUSTER_RADIUS = 10.0;

    // ----------------------------------------------------------------
    // Original state space constants — unchanged
    // ----------------------------------------------------------------

    static final int STATE_CLEAR = 0;
    static final int STATE_FOOD_AHEAD = 1;
    static final int STATE_FOOD_LEFT = 2;
    static final int STATE_FOOD_RIGHT = 3;
    static final int STATE_FOOD_BEHIND = 4;
    static final int STATE_OBSTACLE_AHEAD = 5;
    static final int NUM_STATES = 6;

    static final int O_CLEAR = 0;
    static final int O_FOOD_AHEAD = 1;
    static final int O_FOOD_LEFT = 2;
    static final int O_FOOD_RIGHT = 3;
    static final int O_FOOD_BEHIND = 4;
    static final int O_OBSTACLE_AHEAD = 5;
    static final int NUM_OBS = 6;

    private static final int NUM_ACTIONS = 4;
    private static final int ACTION_MOVE = 0;
    private static final int ACTION_LEFT = 1;
    private static final int ACTION_RIGHT = 2;
    private static final int ACTION_REPRODUCE = 3;

    // ADD:
    public static boolean loggerInitialized = false;

    // ----------------------------------------------------------------
    // Constructors — unchanged
    // ----------------------------------------------------------------

    public ActiveInferenceController(SimulationInternals simulation,
                                     ActiveInferenceAgentParams params) {
        this.simulation = simulation;
        this.params = params;
        this.random = new Random();
        initializeModel();

        // ADD: initialize logger with params from this agent type
        if (!loggerInitialized) {
            String runLabel =
//                    "curiosity=" + params.curiosity
//                    "curiosityFixed=" + params.curiosityFixed
                    "_population=" + simulation.getInitialAgentCount(0)
                    // this gets the initial agent count for each type
                    // this is specifically used for the experiment ran with same initial count
                    // for every type of agent, so in the output dir, we'll see the pop for each agent type
                    // for that specific run
                    + "_seed=" + params.randomSeed;
//                    + "_t=" + System.currentTimeMillis(); // to ensure uniqueness
            ActiveInferenceLogger.initRun(runLabel);
            loggerInitialized = true;
        }
        //
        System.out.println("Active Inference controller created!");
    }

    protected ActiveInferenceController(ActiveInferenceController parent) {
        this.simulation = parent.simulation;
        this.params = parent.params;
        this.random = new Random();
        initializeModel();
    }

    // ----------------------------------------------------------------
    // initializeModel — unchanged
    // ----------------------------------------------------------------

    private void initializeModel() {
        A = new double[NUM_OBS][NUM_STATES];
        for (int o = 0; o < NUM_OBS; o++) {
            for (int s = 0; s < NUM_STATES; s++) {
                A[o][s] = 0.05;
            }
        }

        A[O_CLEAR][STATE_CLEAR] = 0.8;
        A[O_FOOD_AHEAD][STATE_FOOD_AHEAD] = 0.8;
        A[O_FOOD_LEFT][STATE_FOOD_LEFT] = 0.8;
        A[O_FOOD_RIGHT][STATE_FOOD_RIGHT] = 0.8;
        A[O_FOOD_BEHIND][STATE_FOOD_BEHIND] = 0.8;
        A[O_OBSTACLE_AHEAD][STATE_OBSTACLE_AHEAD] = 0.8;

        for (int s = 0; s < NUM_STATES; s++) {
            double sum = 0.0;
            for (int o = 0; o < NUM_OBS; o++) {
                sum += A[o][s];
            }
            for (int o = 0; o < NUM_OBS; o++) {
                A[o][s] /= sum;
            }
        }

        B = new double[NUM_STATES][NUM_STATES][NUM_ACTIONS];
        initializeTransitionModel();

        C = new double[]{0.0, -20.0, 0.0, 0.0, 0.0, 10.0};

        D = new double[NUM_STATES];
        for (int i = 0; i < NUM_STATES; i++) {
            D[i] = 1.0 / NUM_STATES;
        }

        Qs = D.clone();
        b_concentration = new double[NUM_STATES][NUM_STATES][NUM_ACTIONS];
    }

    private void initializeTransitionModel() {
        for (int u = 0; u < NUM_ACTIONS; u++) {
            for (int next = 0; next < NUM_STATES; next++) {
                for (int prev = 0; prev < NUM_STATES; prev++) {
                    B[next][prev][u] = 0.0;
                }
            }
        }

        B[STATE_CLEAR][STATE_CLEAR][ACTION_MOVE] = 0.9;
        B[STATE_CLEAR][STATE_FOOD_AHEAD][ACTION_MOVE] = 1.0;
        B[STATE_OBSTACLE_AHEAD][STATE_OBSTACLE_AHEAD][ACTION_MOVE] = 1.0;
        B[STATE_FOOD_LEFT][STATE_FOOD_LEFT][ACTION_MOVE] = 1.0;
        B[STATE_FOOD_RIGHT][STATE_FOOD_RIGHT][ACTION_MOVE] = 1.0;
        B[STATE_FOOD_BEHIND][STATE_FOOD_BEHIND][ACTION_MOVE] = 1.0;

        for (int u = 0; u < NUM_ACTIONS; u++) {
            for (int prev = 0; prev < NUM_STATES; prev++) {
                double sum = 0.0;
                for (int next = 0; next < NUM_STATES; next++) {
                    B[next][prev][u] += random.nextDouble() * 0.02;
                    sum += B[next][prev][u];
                }
                for (int next = 0; next < NUM_STATES; next++) {
                    B[next][prev][u] /= sum;
                }
            }
        }

        B[STATE_CLEAR][STATE_CLEAR][ACTION_LEFT] = 1.0;
        B[STATE_OBSTACLE_AHEAD][STATE_OBSTACLE_AHEAD][ACTION_LEFT] = 1.0;
        B[STATE_FOOD_RIGHT][STATE_FOOD_AHEAD][ACTION_LEFT] = 1.0;
        B[STATE_FOOD_BEHIND][STATE_FOOD_RIGHT][ACTION_LEFT] = 1.0;
        B[STATE_FOOD_LEFT][STATE_FOOD_BEHIND][ACTION_LEFT] = 1.0;
        B[STATE_FOOD_AHEAD][STATE_FOOD_LEFT][ACTION_LEFT] = 1.0;

        B[STATE_CLEAR][STATE_CLEAR][ACTION_RIGHT] = 1.0;
        B[STATE_OBSTACLE_AHEAD][STATE_OBSTACLE_AHEAD][ACTION_RIGHT] = 1.0;
        B[STATE_FOOD_LEFT][STATE_FOOD_AHEAD][ACTION_RIGHT] = 1.0;
        B[STATE_FOOD_BEHIND][STATE_FOOD_LEFT][ACTION_RIGHT] = 1.0;
        B[STATE_FOOD_RIGHT][STATE_FOOD_BEHIND][ACTION_RIGHT] = 1.0;
        B[STATE_FOOD_AHEAD][STATE_FOOD_RIGHT][ACTION_RIGHT] = 1.0;

        for (int s = 0; s < NUM_STATES; s++) {
            B[s][s][ACTION_REPRODUCE] = 1.0;
        }
    }

    public class AIInput implements ControllerInput {
        @Override
        public void mutate(float adjustmentStrength) {}
    }

    // ================================================================
    // controlAgent
    // ================================================================

    @Override
    public void controlAgent(Agent baseAgent, ControllerListener inputCallback) {
        System.out.println("Agent curiosity: " + params.curiosity);
        System.out.println("AI tick");
        ComplexAgent agent = (ComplexAgent) baseAgent;

        // --- 1. Perception (unchanged) ---
        SeeInfo seeInfo = agent.getState(VisionState.class).distanceLook();
        int obsIdx = mapObservation(seeInfo);
        double energy = (double) agent.getEnergy();

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

        double[] posterior = new double[NUM_STATES];
        for (int s = 0; s < NUM_STATES; s++) {
            posterior[s] = prior[s] * likelihood[s];
        }
        Qs = Matrices.normalize(posterior);

        // --- 2. Curiosity adjustment (unchanged) ---
        if (!params.curiosityFixed) {
            double uncertainty = Matrices.entropy(Qs);
            if (uncertainty > 0.7) {
                params.curiosity = Math.min(params.curiosity + 0.05, 1.0);
            } else if (uncertainty < 0.3) {
                params.curiosity = Math.max(params.curiosity - 0.05, 0.0);
            }
        }

        // --- 3. Learning (unchanged) ---
        if (Qs_prev != null && action_prev != -1) {
            double learningRate = 1.0;
            for (int next = 0; next < NUM_STATES; next++) {
                for (int prev = 0; prev < NUM_STATES; prev++) {
                    b_concentration[next][prev][action_prev] += learningRate * Qs[next] * Qs_prev[prev];
                }
            }
        }

        inputCallback.beforeControl(agent, new AIInput());

        // --- 4. Planning ---
        double[] G = new double[NUM_ACTIONS];
        double[] epistemicPerAction = new double[NUM_ACTIONS];
        double[] extrinsicPerAction = new double[NUM_ACTIONS];

        for (int u = 0; u < NUM_ACTIONS; u++) {
            double[] predictedState = Matrices.multiply(BForAction(u), Qs);
            double[] predictedObs = Matrices.multiply(A, predictedState);

            double extrinsic = -Matrices.dot(predictedObs, C);
            double curiosityFactor = params.curiosity;
            double epistemic = -curiosityFactor * Matrices.entropy(predictedObs);

            epistemicPerAction[u] = epistemic;
            extrinsicPerAction[u] = extrinsic;

            G[u] = extrinsic + epistemic;

            if (u == ACTION_LEFT || u == ACTION_RIGHT) {
                G[u] += 0.2;
            }

            if (u == ACTION_REPRODUCE) {
                if (energy > params.energyPreference * 40.0) {
                    G[u] -= 5.0;
                } else {
                    G[u] += 10.0;
                }
            }
        }

        // --- 4. Softmax selection ---
        double temperature = 1.0;
        double sum = 0.0;
        double[] probs = new double[NUM_ACTIONS];
        for (int u = 0; u < NUM_ACTIONS; u++) {
            probs[u] = Math.exp(-G[u] / temperature);
            sum += probs[u];
        }
        for (int u = 0; u < NUM_ACTIONS; u++) {
            probs[u] /= sum;
        }

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

        // --- 5. Execute ---
        agent.setShouldReproduceAsex(false);
        agent.setCommOutbox(0);

        Qs_prev = Qs.clone();
        action_prev = selectedAction;

        switch (selectedAction) {
            case ACTION_LEFT:      agent.turnLeft(); break;
            case ACTION_RIGHT:     agent.turnRight(); break;
            case ACTION_REPRODUCE: agent.setShouldReproduceAsex(true); break;
            case ACTION_MOVE:
            default:               agent.step(); break;
        }

        // ================================================================
        // ADD: Data collection
        // ================================================================

        try {
            double complexity = ActiveInferenceLogger.computeKL(Qs, prior);
            double inaccuracy = ActiveInferenceLogger.computeInaccuracy(Qs, A, obsIdx);
            double freeEnergy = complexity + inaccuracy;

            double epistemicValue = epistemicPerAction[selectedAction];
            double extrinsicValue = extrinsicPerAction[selectedAction];

            int agentId = System.identityHashCode(agent);
            int agentType = agent.getType();
            double ax = agent.getPosition().x;
            double ay = agent.getPosition().y;

            int timestep = (int) simulation.getTime();
            synchronized (registryLock) {
                if (timestep != registryTimestep) {
                    if (registryTimestep >= 0) {
                        flushSystemLog(registryTimestep);
                    }
                    agentRegistry.clear();
                    registryTimestep = timestep;
                }
                agentRegistry.add(new ClusterUtils.AgentSnapshot(
                        agentId, agentType, ax, ay, freeEnergy
                ));
            }

            List<ClusterUtils.AgentSnapshot> registrySnapshot;
            synchronized (registryLock) {
                registrySnapshot = new ArrayList<ClusterUtils.AgentSnapshot>(agentRegistry);
            }

            List<ClusterUtils.Cluster> clusters =
                    ClusterUtils.detectClusters(registrySnapshot, CLUSTER_RADIUS);

            ClusterUtils.Cluster myCluster = ClusterUtils.findClusterOf(agentId, clusters);

            double centroidX = myCluster != null ? myCluster.centroidX : ax;
            double centroidY = myCluster != null ? myCluster.centroidY : ay;
            double clusterF = myCluster != null ? myCluster.meanFreeEnergy() : freeEnergy;
            double clusterPurity = myCluster != null ? myCluster.purity() : 1.0;

            double nnDist = ClusterUtils.nearestNeighborDist(agentId, ax, ay, registrySnapshot);

            ActiveInferenceLogger logger = ActiveInferenceLogger.getInstance();
            logger.logAgentStep(
                    timestep, agentId, agentType,
                    ax, ay,
                    freeEnergy, complexity, inaccuracy,
                    epistemicValue, extrinsicValue,
                    params.curiosity,
                    nnDist == Double.MAX_VALUE ? -1.0 : nnDist,
                    centroidX, centroidY,
                    clusterF, clusterPurity
            );

        } catch (Exception e) {
            System.err.println("[ActiveInferenceLogger] Logging error: " + e.getMessage());
        }
    }

    private static void flushSystemLog(int completedTimestep) {
        List<ClusterUtils.AgentSnapshot> snapshot;
        synchronized (registryLock) {
            snapshot = new ArrayList<ClusterUtils.AgentSnapshot>(agentRegistry);
        }

        List<ClusterUtils.Cluster> clusters =
                ClusterUtils.detectClusters(snapshot, CLUSTER_RADIUS);

        double orderParam = ClusterUtils.systemOrderParameter(clusters);
        double[] positions = ClusterUtils.toPositionArray(snapshot);

        int gridW = 100;
        int gridH = 100;

        ActiveInferenceLogger.getInstance().logSystemStep(
                completedTimestep, orderParam, positions, gridW, gridH
        );
    }

    // ----------------------------------------------------------------
    // Original helpers — unchanged
    // ----------------------------------------------------------------

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
}