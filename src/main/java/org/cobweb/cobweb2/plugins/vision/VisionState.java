package org.cobweb.cobweb2.plugins.vision;

import org.cobweb.cobweb2.core.Agent;
import org.cobweb.cobweb2.core.Direction;
import org.cobweb.cobweb2.core.Environment;
import org.cobweb.cobweb2.core.LocationDirection;
import org.cobweb.cobweb2.plugins.AgentState;


public class VisionState implements AgentState {

	private final Environment environment;
	private final Agent agent;

	public VisionState(Environment environment, Agent agent) {
		this.environment = environment;
		this.agent = agent;
	}

	public static final int LOOK_DISTANCE = 4;

	/**
	 * This method allows the agent to see what is in front of it.
	 *
	 * @return What the agent sees and at what distance.
	 */
	public SeeInfo distanceLook() {
        LocationDirection agentPos = agent.getPosition();
		LocationDirection destPos = environment.topology.getAdjacent(agentPos);

		for (int dist = 1; dist <= LOOK_DISTANCE; ++dist) {

            // Turn global position into relative position to pass into SeeInfo,
            // because dx (distance x) dy (distance y) should be relative to the agents
            int dxGlobal = 0;
            int dyGlobal = 0;

            if (destPos != null) {
                dxGlobal = destPos.x - agentPos.x;
                dyGlobal = destPos.y - agentPos.y;
            }

            // Call the helper to transfer global to relative position
            int[] rel = toRelative(dxGlobal, dyGlobal, agentPos.direction);

			// We are looking at the wall
			if (destPos == null) {
				return new SeeInfo(dist, Environment.FLAG_STONE, LOOK_DISTANCE, rel[0], rel[1]);
            }

			// Check for stone...
			if (environment.hasStone(destPos))
                return new SeeInfo(dist, Environment.FLAG_STONE, LOOK_DISTANCE, rel[0], rel[1]);

			// If there's another agent there, then return that it's a stone...
			if (environment.hasAgent(destPos) && environment.getAgent(destPos) != agent)
				return new SeeInfo(dist, Environment.FLAG_AGENT, LOOK_DISTANCE, rel[0], rel[1]);

			// If there's food there, return the food...
			if (environment.hasFood(destPos))
                return new SeeInfo(dist, Environment.FLAG_FOOD, LOOK_DISTANCE, rel[0], rel[1]);

			if (environment.hasDrop(destPos))
                return new SeeInfo(dist, Environment.FLAG_DROP, LOOK_DISTANCE, rel[0], rel[1]);

			destPos = environment.topology.getAdjacent(destPos);
		}
        // See nothing
		return new SeeInfo(LOOK_DISTANCE, 0, LOOK_DISTANCE, 0, 0);
	}

    /**
     * Converts global dx, dy to coordinates relative to the agent's facing direction.
     */
    private int[] toRelative(int dxGlobal, int dyGlobal, Direction facing) {
        int fx = facing.x;
        int fy = facing.y;

        // 90-degree rotation based on facing
        // If facing up (-y), relX = dx, relY = dy
        // If facing right (+x), rotate 90° clockwise: relX = dy, relY = -dx
        // If facing down (+y), rotate 180°: relX = -dx, relY = -dy
        // If facing left (-x), rotate 90° counterclockwise: relX = -dy, relY = dx
        int relX = 0;
        int relY = 0;

        if (fx == 0 && fy == -1) { // up
            relX = dxGlobal;
            relY = dyGlobal;
        } else if (fx == 1 && fy == 0) { // right
            relX = dyGlobal;
            relY = -dxGlobal;
        } else if (fx == 0 && fy == 1) { // down
            relX = -dxGlobal;
            relY = -dyGlobal;
        } else if (fx == -1 && fy == 0) { // left
            relX = -dyGlobal;
            relY = dxGlobal;
        } else { // fallback, should not happen
            relX = dxGlobal;
            relY = dyGlobal;
        }

        return new int[]{relX, relY};
    }


	@Override
	public boolean isTransient() {
		return true;
	}
	private static final long serialVersionUID = 1L;
}
