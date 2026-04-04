package org.cobweb.cobweb2.impl.ai;

import java.util.ArrayList;
import java.util.List;

/**
 * Lightweight spatial utilities for cluster detection and measurement.
 *
 * Designed for Java 1.7 compatibility (no streams, no lambdas).
 *
 * These are called from ActiveInferenceController.controlAgent() to
 * provide spatial context needed by ActiveInferenceLogger.
 *
 * All methods are static — no instantiation needed.
 */
public class ClusterUtils {

    // Prevent instantiation
    private ClusterUtils() {}

    // -----------------------------------------------------------------------
    // Agent snapshot: a lightweight struct passed around instead of
    // reaching back into ComplexAgent repeatedly
    // -----------------------------------------------------------------------

    public static class AgentSnapshot {
        public final int id;
        public final int type;
        public final double x;
        public final double y;
        public final double freeEnergy;

        public AgentSnapshot(int id, int type, double x, double y, double freeEnergy) {
            this.id = id;
            this.type = type;
            this.x = x;
            this.y = y;
            this.freeEnergy = freeEnergy;
        }
    }

    // -----------------------------------------------------------------------
    // Nearest-neighbor distance
    // -----------------------------------------------------------------------

    /**
     * Euclidean distance from (x, y) to the nearest other agent in the list.
     * Returns Double.MAX_VALUE if the list has fewer than 2 agents.
     *
     * @param selfId    ID of the agent to exclude from the search
     * @param x         agent's x position
     * @param y         agent's y position
     * @param allAgents snapshot list of all agents this timestep
     */
    public static double nearestNeighborDist(int selfId, double x, double y,
                                             List<AgentSnapshot> allAgents) {
        double minDist = Double.MAX_VALUE;
        for (int i = 0; i < allAgents.size(); i++) {
            AgentSnapshot a = allAgents.get(i);
            if (a.id == selfId) {
                continue;
            }
            double d = euclidean(x, y, a.x, a.y);
            if (d < minDist) {
                minDist = d;
            }
        }
        return minDist;
    }

    // -----------------------------------------------------------------------
    // Cluster assignment: simple radius-based clustering
    // -----------------------------------------------------------------------

    /**
     * A detected cluster.
     */
    public static class Cluster {
        public final List<AgentSnapshot> members;
        public double centroidX;
        public double centroidY;

        public Cluster() {
            this.members = new ArrayList<AgentSnapshot>();
        }

        /** Recompute centroid from current members list. */
        public void recomputeCentroid() {
            double sumX = 0, sumY = 0;
            for (int i = 0; i < members.size(); i++) {
                sumX += members.get(i).x;
                sumY += members.get(i).y;
            }
            centroidX = members.isEmpty() ? 0 : sumX / members.size();
            centroidY = members.isEmpty() ? 0 : sumY / members.size();
        }

        /**
         * Purity = fraction of agents belonging to the most common type.
         * 1.0 = all same type, lower = mixed.
         */
        public double purity() {
            if (members.isEmpty()) {
                return 0.0;
            }
            // Count by type (using a simple parallel array approach for Java 1.7)
            // Assumes agent types are small non-negative integers
            int maxType = 0;
            for (int i = 0; i < members.size(); i++) {
                if (members.get(i).type > maxType) {
                    maxType = members.get(i).type;
                }
            }
            int[] typeCounts = new int[maxType + 1];
            for (int i = 0; i < members.size(); i++) {
                typeCounts[members.get(i).type]++;
            }
            int maxCount = 0;
            for (int c : typeCounts) {
                if (c > maxCount) {
                    maxCount = c;
                }
            }
            return (double) maxCount / members.size();
        }

        /**
         * Mean free energy of all members in this cluster.
         */
        public double meanFreeEnergy() {
            if (members.isEmpty()) {
                return 0.0;
            }
            double sum = 0;
            for (int i = 0; i < members.size(); i++) {
                sum += members.get(i).freeEnergy;
            }
            return sum / members.size();
        }

        public int size() {
            return members.size();
        }
    }

    /**
     * Greedy single-linkage clustering by radius.
     *
     * Each agent joins the nearest existing cluster if within clusterRadius.
     * Otherwise it starts a new cluster.
     *
     * O(n^2) — fine for typical COBWEB population sizes (< 1000 agents).
     * Centroids are recomputed after all agents are assigned.
     *
     * @param agents        all agents this timestep
     * @param clusterRadius max distance to join an existing cluster
     * @return              list of detected clusters
     */
    public static List<Cluster> detectClusters(List<AgentSnapshot> agents,
                                               double clusterRadius) {
        List<Cluster> clusters = new ArrayList<Cluster>();

        for (int i = 0; i < agents.size(); i++) {
            AgentSnapshot a = agents.get(i);
            Cluster nearest = null;
            double nearestDist = Double.MAX_VALUE;

            // Check against current cluster centroids
            for (int j = 0; j < clusters.size(); j++) {
                Cluster c = clusters.get(j);
                double d = euclidean(a.x, a.y, c.centroidX, c.centroidY);
                if (d < clusterRadius && d < nearestDist) {
                    nearest = c;
                    nearestDist = d;
                }
            }

            if (nearest != null) {
                nearest.members.add(a);
                nearest.recomputeCentroid(); // incremental update
            } else {
                Cluster newCluster = new Cluster();
                newCluster.members.add(a);
                newCluster.centroidX = a.x;
                newCluster.centroidY = a.y;
                clusters.add(newCluster);
            }
        }

        return clusters;
    }

    /**
     * Find which cluster an agent belongs to (by agent ID).
     * Returns null if not found (agent may be isolated).
     */
    public static Cluster findClusterOf(int agentId, List<Cluster> clusters) {
        for (int i = 0; i < clusters.size(); i++) {
            Cluster c = clusters.get(i);
            for (int j = 0; j < c.members.size(); j++) {
                if (c.members.get(j).id == agentId) {
                    return c;
                }
            }
        }
        return null;
    }

    // -----------------------------------------------------------------------
    // System-level order parameter
    // -----------------------------------------------------------------------

    /**
     * System-wide order parameter: mean cluster purity across all clusters,
     * weighted by cluster size.
     *
     * 0.0 = all clusters perfectly mixed
     * 1.0 = all clusters perfectly sorted by type
     */
    public static double systemOrderParameter(List<Cluster> clusters) {
        if (clusters.isEmpty()) {
            return 0.0;
        }
        double weightedPuritySum = 0.0;
        int totalAgents = 0;
        for (int i = 0; i < clusters.size(); i++) {
            Cluster c = clusters.get(i);
            weightedPuritySum += c.purity() * c.size();
            totalAgents += c.size();
        }
        return totalAgents > 0 ? weightedPuritySum / totalAgents : 0.0;
    }

    // -----------------------------------------------------------------------
    // Flat position array for spatial entropy calculation
    // -----------------------------------------------------------------------

    /**
     * Convert agent list to flat [x0,y0,x1,y1,...] array
     * for use with ActiveInferenceLogger.logSystemStep().
     */
    public static double[] toPositionArray(List<AgentSnapshot> agents) {
        double[] arr = new double[agents.size() * 2];
        for (int i = 0; i < agents.size(); i++) {
            arr[i * 2] = agents.get(i).x;
            arr[i * 2 + 1] = agents.get(i).y;
        }
        return arr;
    }

    // -----------------------------------------------------------------------
    // Internal
    // -----------------------------------------------------------------------

    private static double euclidean(double x1, double y1, double x2, double y2) {
        double dx = x1 - x2;
        double dy = y1 - y2;
        return Math.sqrt(dx * dx + dy * dy);
    }
}