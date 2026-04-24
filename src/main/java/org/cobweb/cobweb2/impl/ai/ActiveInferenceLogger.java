package org.cobweb.cobweb2.impl.ai;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Map;
import org.cobweb.cobweb2.core.Agent;

/**
 * One data logger for Active Inference simulation.
 *
 * Writes three CSV files simultaneously:
 * system_log.csv    - one row per timestep, macro/thermodynamic quantities
 * agent_log.csv     - one row per agent per sampled timestep
 * excursion_log.csv - one row per messenger excursion event (departure + return)
 *
 * - No changes to original COBWEB code
 * - Controller calls logger; logger owns all I/O.
 * - Compatible with Java 1.7 (no lambdas, no streams).
 * - Excursion detection is fully internal to this class.
 */
public class ActiveInferenceLogger {

    // The data logger
    // -----------------------------------------------------------------------

    private static ActiveInferenceLogger instance;

    public static ActiveInferenceLogger getInstance() {
        if (instance == null) {
            instance = new ActiveInferenceLogger();
        }
        return instance;
    }

    // Configuration
    // -----------------------------------------------------------------------

    /**
     * How often (in timesteps) to write full agent rows.
     */
    public int agentLogInterval = 10;

    /**
     * Minimum change in system-level order parameter to trigger
     * an extra full-agent snapshot.
     */
    public double orderParameterDeltaTrigger = 0.05;

    /**
     * Distance (in grid cells) an agent must travel from its cluster
     * centroid to be considered "in excursion" (messenger) mode.
     */
    public double excursionRadiusThreshold = 8.0;

    /**
     * Output directory. Trailing slash optional.
     */
    public String outputDir = "ai_logs/";

    // Internal state
    // -----------------------------------------------------------------------
    private PrintWriter systemWriter;
    private PrintWriter agentWriter;
    private PrintWriter excursionWriter;

    private boolean initialized = false;

    // Tracks last known order parameter for delta-triggered snapshots
    private double lastOrderParameter = -1.0;

    // Per-agent excursion state: agentId -> ExcursionRecord
    private final Map<Integer, ExcursionRecord> excursionState =
            new HashMap<Integer, ExcursionRecord>();

    // Accumulator for system-level aggregates, reset each timestep
    private final Map<Integer, TypeAccumulator> typeAccumulators =
            new HashMap<Integer, TypeAccumulator>();

    private int lastSystemTimestep = -1;

    // helper classes
    // -----------------------------------------------------------------------

    /** Tracks one agent's excursion (messenger) state across timesteps. */
    private static class ExcursionRecord {
        boolean inExcursion = false;
        int departureTimestep = -1;
        double clusterFAtDeparture = Double.NaN;
        double clusterPurityAtDeparture = Double.NaN;
        double maxDistanceDuringExcursion = 0.0;
    }

    /** Accumulates per-type statistics within a single timestep. */
    private static class TypeAccumulator {
        int count = 0;
        double sumF = 0.0;
        double sumFSq = 0.0;   // for variance: E[F^2] - E[F]^2
        double sumEpistemic = 0.0;
        double sumExtrinsic = 0.0;
        int excursionCount = 0;
    }

    // Initialization
    // -----------------------------------------------------------------------

    private ActiveInferenceLogger() {
        // private: use getInstance()
    }

    public static void initRun(String runLabel) {
        // TRYING to fix folder issue
//        if (instance != null) {
//            instance.close();
//        }
//
//        instance = new ActiveInferenceLogger();
//
//        instance.outputDir = "ai_logs/" + runLabel + "/";
//        instance.initialized = false;
//
//        ActiveInferenceController.loggerInitialized = false;
        ActiveInferenceLogger logger = getInstance();
        logger.outputDir = "ai_logs/" + runLabel + "/";
        logger.initialized = false;  // force re-init with new directory
        ActiveInferenceController.loggerInitialized = false; // reset for next run
    }

    /**
     * Opens all output files.
     */
    private void ensureInitialized() {
        if (initialized) {
            return;
        }
        try {
            // Create output directory if needed
            java.io.File dir = new java.io.File(outputDir);
            if (!dir.exists()) {
                dir.mkdirs();
            }

            systemWriter = new PrintWriter(new FileWriter(outputDir + "system_log.csv", false));
            agentWriter = new PrintWriter(new FileWriter(outputDir + "agent_log.csv", false));
            excursionWriter = new PrintWriter(new FileWriter(outputDir + "excursion_log.csv", false));

            writeSystemHeader();
            writeAgentHeader();
            writeExcursionHeader();

            initialized = true;
        } catch (IOException e) {
            System.err.println("[ActiveInferenceLogger] Failed to open log files: " + e.getMessage());
        }
    }

    private void writeSystemHeader() {
        systemWriter.println(
                "timestep," +
                        "total_F," +
                        "mean_F," +
                        "F_variance," +
                        "order_parameter," +
                        "num_agents," +
                        "num_excursions," +
                        "mean_epistemic," +
                        "mean_extrinsic," +
                        "entropy_spatial"
        );
        systemWriter.flush();
    }

    private void writeAgentHeader() {
        agentWriter.println(
                "timestep," +
                        "agent_id," +
                        "agent_type," +
                        "x," +
                        "y," +
                        "free_energy," +
                        "complexity," +
                        "inaccuracy," +
                        "epistemic_value," +
                        "extrinsic_value," +
                        "curiosity," +
                        "nearest_neighbor_dist," +
                        "is_excursion"
        );
        agentWriter.flush();
    }

    private void writeExcursionHeader() {
        excursionWriter.println(
                "agent_id," +
                        "agent_type," +
                        "timestep_departure," +
                        "timestep_return," +
                        "duration," +
                        "max_distance," +
                        "cluster_F_before," +
                        "cluster_F_after," +
                        "cluster_purity_before," +
                        "cluster_purity_after"
        );
        excursionWriter.flush();
    }

    // Public API
    // -----------------------------------------------------------------------

    public void logAgentStep(
            int timestep,
            int agentId,
            int agentType,
            double x,
            double y,
            double freeEnergy,
            double complexity,
            double inaccuracy,
            double epistemicValue,
            double extrinsicValue,
            double curiosity,
            double nearestNeighborDist,
            double clusterCentroidX,
            double clusterCentroidY,
            double currentClusterF,
            double currentClusterPurity
    ) {
        ensureInitialized();

        // --- Excursion detection ---
        double distFromCentroid = Math.sqrt(
                Math.pow(x - clusterCentroidX, 2) + Math.pow(y - clusterCentroidY, 2)
        );
        boolean isExcursion = distFromCentroid > excursionRadiusThreshold;
        updateExcursionState(agentId, agentType, timestep, isExcursion,
                distFromCentroid, currentClusterF, currentClusterPurity);

        // --- Accumulate type statistics for system log ---
        TypeAccumulator acc = typeAccumulators.get(agentType);
        if (acc == null) {
            acc = new TypeAccumulator();
            typeAccumulators.put(agentType, acc);
        }
        acc.count++;
        acc.sumF += freeEnergy;
        acc.sumFSq += freeEnergy * freeEnergy;
        acc.sumEpistemic += epistemicValue;
        acc.sumExtrinsic += extrinsicValue;
        if (isExcursion) {
            acc.excursionCount++;
        }

        // --- Write agent row (interval or event-triggered) ---
        boolean shouldWrite = (timestep % agentLogInterval == 0);
        if (shouldWrite && agentWriter != null) {
            agentWriter.printf(
                    "%d,%d,%d,%.2f,%.2f,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%d%n",
                    timestep, agentId, agentType,
                    x, y,
                    freeEnergy, complexity, inaccuracy,
                    epistemicValue, extrinsicValue,
                    curiosity, nearestNeighborDist,
                    isExcursion ? 1 : 0
            );
        }
    }

    public void logSystemStep(
            int timestep,
            double orderParameter,
            double[] agentPositions,
            int gridWidth,
            int gridHeight
    ) {
        ensureInitialized();

        if (timestep == lastSystemTimestep) {
            return;
        }
        lastSystemTimestep = timestep;

        // Aggregate across all types
        double totalF = 0.0;
        double totalFSq = 0.0;
        int totalAgents = 0;
        int totalExcursions = 0;
        double totalEpistemic = 0.0;
        double totalExtrinsic = 0.0;

        for (TypeAccumulator acc : typeAccumulators.values()) {
            totalF += acc.sumF;
            totalFSq += acc.sumFSq;
            totalAgents += acc.count;
            totalExcursions += acc.excursionCount;
            totalEpistemic += acc.sumEpistemic;
            totalExtrinsic += acc.sumExtrinsic;
        }

        double meanF = totalAgents > 0 ? totalF / totalAgents : 0.0;
        double varianceF = totalAgents > 0
                ? (totalFSq / totalAgents) - (meanF * meanF)
                : 0.0;
        double meanEpistemic = totalAgents > 0 ? totalEpistemic / totalAgents : 0.0;
        double meanExtrinsic = totalAgents > 0 ? totalExtrinsic / totalAgents : 0.0;

        double spatialEntropy = computeSpatialEntropy(agentPositions, gridWidth, gridHeight);

        if (systemWriter != null) {
            systemWriter.printf(
                    "%d,%.6f,%.6f,%.6f,%.6f,%d,%d,%.6f,%.6f,%.6f%n",
                    timestep,
                    totalF, meanF, varianceF,
                    orderParameter,
                    totalAgents, totalExcursions,
                    meanEpistemic, meanExtrinsic,
                    spatialEntropy
            );
        }

        // --- Event-triggered agent snapshot ---
        boolean orderParamJumped = Math.abs(orderParameter - lastOrderParameter) > orderParameterDeltaTrigger;
        if (orderParamJumped && lastOrderParameter >= 0) {
            if (agentWriter != null) {
                agentWriter.flush();
            }
        }
        lastOrderParameter = orderParameter;

        if (systemWriter != null) {
            systemWriter.flush();
        }
        if (agentWriter != null) {
            agentWriter.flush();
        }

        typeAccumulators.clear();
    }

    public void close() {
        if (systemWriter != null) {
            systemWriter.flush();
            systemWriter.close();
        }
        if (agentWriter != null) {
            agentWriter.flush();
            agentWriter.close();
        }
        if (excursionWriter != null) {
            excursionWriter.flush();
            excursionWriter.close();
        }
        initialized = false;
        instance = null;
    }

    // Excursion tracking (internal)
    // -----------------------------------------------------------------------

    private void updateExcursionState(
            int agentId,
            int agentType,
            int timestep,
            boolean isExcursion,
            double distFromCentroid,
            double currentClusterF,
            double currentClusterPurity
    ) {
        ExcursionRecord rec = excursionState.get(agentId);
        if (rec == null) {
            rec = new ExcursionRecord();
            excursionState.put(agentId, rec);
        }

        if (!rec.inExcursion && isExcursion) {
            rec.inExcursion = true;
            rec.departureTimestep = timestep;
            rec.clusterFAtDeparture = currentClusterF;
            rec.clusterPurityAtDeparture = currentClusterPurity;
            rec.maxDistanceDuringExcursion = distFromCentroid;

        } else if (rec.inExcursion && isExcursion) {
            if (distFromCentroid > rec.maxDistanceDuringExcursion) {
                rec.maxDistanceDuringExcursion = distFromCentroid;
            }

        } else if (rec.inExcursion && !isExcursion) {
            int duration = timestep - rec.departureTimestep;
            if (excursionWriter != null) {
                excursionWriter.printf(
                        "%d,%d,%d,%d,%d,%.4f,%.6f,%.6f,%.6f,%.6f%n",
                        agentId, agentType, rec.departureTimestep, timestep,
                        duration, rec.maxDistanceDuringExcursion,
                        rec.clusterFAtDeparture, currentClusterF,
                        rec.clusterPurityAtDeparture, currentClusterPurity
                );
                excursionWriter.flush();
            }
            rec.inExcursion = false;
            rec.maxDistanceDuringExcursion = 0.0;
        }
    }

    // Derived quantities (computed internally)
    // -----------------------------------------------------------------------

    private double computeSpatialEntropy(double[] agentPositions, int gridWidth, int gridHeight) {
        if (agentPositions == null || agentPositions.length < 2) {
            return 0.0;
        }

        int bins = 4; // 4x4 grid = 16 cells
        int[] counts = new int[bins * bins];
        int totalAgents = agentPositions.length / 2;

        for (int i = 0; i < agentPositions.length; i += 2) {
            double x = agentPositions[i];
            double y = agentPositions[i + 1];
            int bx = Math.min((int) (x / gridWidth * bins), bins - 1);
            int by = Math.min((int) (y / gridHeight * bins), bins - 1);
            counts[by * bins + bx]++;
        }

        double entropy = 0.0;
        for (int c : counts) {
            if (c > 0) {
                double p = (double) c / totalAgents;
                entropy -= p * Math.log(p);
            }
        }
        return entropy;
    }

    // Utility
    // -----------------------------------------------------------------------

    public static double computeFreeEnergy(double[] Qs, double[] prior,
                                           double[][] A, int obsIdx) {
        double complexity = computeKL(Qs, prior);
        double inaccuracy = computeInaccuracy(Qs, A, obsIdx);
        return complexity + inaccuracy;
    }

    public static double computeKL(double[] Q, double[] P) {
        double kl = 0.0;
        for (int s = 0; s < Q.length; s++) {
            if (Q[s] > 1e-12 && P[s] > 1e-12) {
                kl += Q[s] * Math.log(Q[s] / P[s]);
            }
        }
        return kl;
    }

    public static double computeInaccuracy(double[] Qs, double[][] A, int obsIdx) {
        double inaccuracy = 0.0;
        for (int s = 0; s < Qs.length; s++) {
            if (A[obsIdx][s] > 1e-12) {
                inaccuracy -= Qs[s] * Math.log(A[obsIdx][s]);
            }
        }
        return inaccuracy;
    }

    public static double computeEpistemicValue(double[] predictedObs) {
        double entropy = 0.0;
        for (int o = 0; o < predictedObs.length; o++) {
            if (predictedObs[o] > 1e-12) {
                entropy -= predictedObs[o] * Math.log(predictedObs[o]);
            }
        }
        return entropy;
    }
}