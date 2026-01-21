package org.cobweb.cobweb2.ui.swing.config;

import java.awt.BorderLayout;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.table.TableColumnModel;

import org.cobweb.cobweb2.SimulationConfig;
import org.cobweb.cobweb2.impl.ai.ActiveInferenceController;
import org.cobweb.cobweb2.impl.ai.ActiveInferenceControllerParams;
import org.cobweb.swingutil.ColorLookup;

public class ActiveInferencePanel extends SettingsPanel {

    private static final long serialVersionUID = 1L;
    private ActiveInferenceControllerParams params;
    private ColorLookup agentColors;

    public ActiveInferencePanel(ColorLookup agentColors) {
        this.agentColors = agentColors;
    }

    @Override
    public void bindToParser(SimulationConfig p) {
        if (!(p.controllerParams instanceof ActiveInferenceControllerParams)) {
            p.setControllerName(ActiveInferenceController.class.getName());
            // If existing params match type, keep them, else SimulationConfig created new
            // defaults
            if (params != null)
                p.controllerParams = params;
        }
        params = (ActiveInferenceControllerParams) p.controllerParams;
        updateBoxes();
    }

    private void updateBoxes() {
        setLayout(new BorderLayout());
        this.removeAll();

        JPanel agentPanel = new JPanel();
        agentPanel.setLayout(new BorderLayout());
        Util.makeGroupPanel(agentPanel, "Active Inference Parameters");

        final MixedValueJTable agentParamTable = new MixedValueJTable(
                new ConfigTableModel(params.agentParams, "Agent "));

        TableColumnModel agParamColModel = agentParamTable.getColumnModel();
        agParamColModel.getColumn(0).setPreferredWidth(200);

        Util.colorHeaders(agentParamTable, true, agentColors);
        JScrollPane agentScroll = new JScrollPane(agentParamTable);

        agentPanel.add(agentScroll, BorderLayout.CENTER);

        this.add(agentPanel, BorderLayout.CENTER);
    }
}
