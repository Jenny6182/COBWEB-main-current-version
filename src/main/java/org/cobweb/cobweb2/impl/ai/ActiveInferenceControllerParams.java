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
//        System.out.println("ActiveInferenceControllerParams constructor called!");  // ADD
//        System.out.println("ActiveInferenceControllerParams constructor called from:");
//        new Exception("stack trace").printStackTrace(System.out);

        resize(simParams);
        // FORCE each agent type to have its own instance
        for (int i = 0; i < agentParams.length; i++) {
//            if (agentParams[i] == null) {
                agentParams[i] = new ActiveInferenceAgentParams(simParam);
//            }
        }
    }

    @Override
    protected ActiveInferenceAgentParams newAgentParam() {
        return new ActiveInferenceAgentParams(simParam);
    }

    @Override
    public Controller createController(SimulationInternals sim, int type) {
        System.out.println("CREATE CONTROLLER seed[" + type + "] = " + agentParams[type].randomSeed);
        return new ActiveInferenceController(sim, agentParams[type]);
    }
}
