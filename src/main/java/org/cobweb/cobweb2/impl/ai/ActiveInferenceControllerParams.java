package org.cobweb.cobweb2.impl.ai;

import org.cobweb.cobweb2.core.Controller;
import org.cobweb.cobweb2.core.SimulationInternals;
import org.cobweb.cobweb2.impl.ControllerParams;
import org.cobweb.cobweb2.impl.SimulationParams;
import org.cobweb.cobweb2.plugins.PerAgentParams;

public class ActiveInferenceControllerParams extends PerAgentParams<ActiveInferenceAgentParams> implements ControllerParams {
    private static final long serialVersionUID = 1L;
    private final transient SimulationParams simParam;

    public ActiveInferenceControllerParams(SimulationParams simParams) {
        super(ActiveInferenceAgentParams.class);
        this.simParam = simParams;

        resize(simParams);

        // FORCE each agent type to have its own instance
        for (int i = 0; i < agentParams.length; i++) {
            agentParams[i] = new ActiveInferenceAgentParams(simParam);
        }
    }

    @Override
    protected ActiveInferenceAgentParams newAgentParam() {
        return new ActiveInferenceAgentParams(simParam);
    }

    @Override
    public Controller createController(SimulationInternals sim, int type) {
        return new ActiveInferenceController(sim, agentParams[type]);
    }
}
