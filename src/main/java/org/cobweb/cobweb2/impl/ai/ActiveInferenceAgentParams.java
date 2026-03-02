package org.cobweb.cobweb2.impl.ai;

import org.cobweb.cobweb2.impl.SimulationParams;
import org.cobweb.io.ConfDisplayName;
import org.cobweb.io.ConfXMLTag;
import org.cobweb.io.ParameterSerializable;

public class ActiveInferenceAgentParams implements ParameterSerializable {

    private static final long serialVersionUID = 1L;

    @ConfDisplayName("Random Seed")
    @ConfXMLTag("RandomSeed")
    public long randomSeed = 42;

    /**
     * How strongly the agent prefers high energy.
     * Higher values mean clearer preference for being energetic.
     */
    @ConfDisplayName("Energy Preference")
    @ConfXMLTag("EnergyPreference")
    public double energyPreference = 2.0;

    /**
     * How strongly the agent prefers seeing food.
     */
    @ConfDisplayName("Food Preference")
    @ConfXMLTag("FoodPreference")
    public double foodPreference = 1.0;
    public double curiosity = 0.5;  // default 0.5, range 0-1

    public ActiveInferenceAgentParams(SimulationParams simParam) {
        // Initialize with sim params if needed in future
    }
}
