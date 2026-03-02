package org.cobweb.cobweb2.plugins.vision;

import org.cobweb.cobweb2.plugins.AgentState;

/**
 * This class provides the information of what an agent sees.
 *
 */
public class SeeInfo implements AgentState {
	private final int dist;

	private final int type;

	private final int maxDistance;

    // New fields: relative coordinates used for Active Inference
    // and other potentially useful models
    private final int dx;  // relative x from agent
    private final int dy;  // relative y from agent

	/**
	 * Contains the information of what the agent sees.
	 *
	 * @param d Distance to t.
	 * @param t Type of object seen.
	 * @param maxd Maximum distance the agent is able to see.
     * @param dx Relative x coordinate from agent
     * @param dy Relative y coordinate from agent
	 */
	public SeeInfo(int d, int t, int maxd, int dx, int dy) {
		maxDistance = maxd;
		dist = d;
		type = t;
        // modified to store extra information used for agent models
        this.dx = dx;
        this.dy = dy;
	}

    // Old Constructor
    public SeeInfo(int d, int t, int maxd) {
        this(d, t, maxd, 0, 0); // default dx, dy = 0
    }

	/**
	 * Agent sees nothing.
	 * @param maxd Maximum distance the agent can see
	 */
    public SeeInfo(int maxd) {
        this(maxd, 0, maxd, 0, 0);
    }

	/**
	 * @return How far away the object is.
	 */
	public int getDist() {
		return dist;
	}

	/**
	 * @return What the agent sees (rock, food, etc.)
	 */
	public int getType() {
		return type;
	}

	/**
	 * @return Maximum distance the agent can see
	 */
	public int getMaxDistance() {
		return maxDistance;
	}

    // New getters that is used for active inference agent, and potentially
    // might be useful for future agent types / models
    // class VisionState was modified to be consistent with this implementation,
    // it has a modified distanceLook() that computes dx dy and store this in SeeInfo
    public int getDx() {
        return dx;
    }

    public int getDy() {
        return dy;
    }


    @Override
	public boolean isTransient() {
		return true;
	}
	private static final long serialVersionUID = 1L;
}