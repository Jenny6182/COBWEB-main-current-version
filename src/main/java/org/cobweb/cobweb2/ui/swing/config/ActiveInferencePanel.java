package org.cobweb.cobweb2.ui.swing.config;

import java.awt.*;
import javax.swing.*;
import javax.swing.table.TableColumnModel;

import org.cobweb.cobweb2.SimulationConfig;
import org.cobweb.cobweb2.impl.ai.ActiveInferenceController;
import org.cobweb.cobweb2.impl.ai.ActiveInferenceControllerParams;
import org.cobweb.swingutil.ColorLookup;

import java.awt.event.ActionEvent;
import java.util.Random;
import javax.swing.AbstractAction;

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
//
//    private void updateBoxes() {
//        setLayout(new BorderLayout());
//        this.removeAll();
//
//        JPanel agentPanel = new JPanel();
//        agentPanel.setLayout(new BorderLayout());
//        Util.makeGroupPanel(agentPanel, "Active Inference Parameters");
//
//        final MixedValueJTable agentParamTable = new MixedValueJTable(
//                new ConfigTableModel(params.agentParams, "Agent "));
//
//        TableColumnModel agParamColModel = agentParamTable.getColumnModel();
//        agParamColModel.getColumn(0).setPreferredWidth(200);
//
//        // Buttons panel
//        JPanel buttons = new JPanel(new GridLayout(1, params.agentParams.length));
//
//        for (int i = 0; i < params.agentParams.length; i++) {
//            JButton newSeedButton = new JButton(new NewSeedAction(i));
//            buttons.add(newSeedButton);
//        }
//
//        this.add(buttons, BorderLayout.SOUTH);
//
//        Util.colorHeaders(agentParamTable, true, agentColors);
//        JScrollPane agentScroll = new JScrollPane(agentParamTable);
//
//        agentPanel.add(agentScroll, BorderLayout.CENTER);
//
//        this.add(agentPanel, BorderLayout.CENTER);
//    }


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

        // BUTTONS PANEL (matches GeneticAIPanel structure)
        JPanel buttons = new JPanel(new GridLayout(1, params.agentParams.length));

        for (int i = 0; i < params.agentParams.length; i++) {
            JButton randomizeSeed = new JButton(new NewSeedAction(i, agentParamTable));

            JPanel typePanel = new JPanel(new GridLayout(1, 1));
            typePanel.add(randomizeSeed);

            buttons.add(typePanel);
        }

        this.add(buttons, BorderLayout.SOUTH);

        revalidate();
        repaint();
    }

    private final class NewSeedAction extends AbstractAction {

        private final int type;
        private final MixedValueJTable agentParamTable;
        private static final long serialVersionUID = 1L;

        private NewSeedAction(int type, MixedValueJTable agentParamTable) {
            super("New Seed");
            this.type = type;
            this.agentParamTable = agentParamTable;
        }

        @Override
        public void actionPerformed(ActionEvent e) {
            Random r = new Random();
            params.agentParams[type].randomSeed =
                    Math.abs(r.nextLong() % 100000L);

            agentParamTable.repaint();
        }
    }

}
