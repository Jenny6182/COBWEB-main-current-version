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
        }

        // ALWAYS use the params from SimulationConfig (source of truth)
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

        // BUTTONS PANEL (matches GeneticAIPanel structure)
        JPanel buttons = new JPanel(new GridLayout(1, params.agentParams.length));

        for (int i = 0; i < params.agentParams.length; i++) {
            JButton randomizeSeed = new JButton(new NewSeedAction(i, agentParamTable));

            JPanel typePanel = new JPanel(new GridLayout(1, 1));
            typePanel.add(randomizeSeed);

            buttons.add(typePanel);
        }

        this.add(buttons, BorderLayout.SOUTH);

        // Add the curiosity checkbox panel
        JPanel curiosityPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        JCheckBox curiosityCheckBox = new JCheckBox("Fixed Curiosity Mode");

        // Set checkbox based on FIRST agent (assume all same initially)
        curiosityCheckBox.setSelected(params.agentParams[0].curiosityFixed);

        // When toggled → update ALL agents
        curiosityCheckBox.addActionListener(e -> {
            boolean isChecked = curiosityCheckBox.isSelected();

            for (int i = 0; i < params.agentParams.length; i++) {
                params.agentParams[i].curiosityFixed = isChecked;
            }

            System.out.println("Curiosity Fixed (all agents): " + isChecked);
        });

        curiosityPanel.add(curiosityCheckBox);
        this.add(curiosityPanel, BorderLayout.NORTH); // Add the panel to the top of your UI

        revalidate();
        repaint();
    }

    private final class NewSeedAction extends AbstractAction {

        private final int type; // indicate which agent type's seed is being changed
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
            long newSeed = Math.abs(r.nextLong() % 100000L); // get random number for new seed

            // Update the params object
            params.agentParams[type].randomSeed = newSeed;

            agentParamTable.revalidate();
            agentParamTable.repaint();
        }
    }

}
