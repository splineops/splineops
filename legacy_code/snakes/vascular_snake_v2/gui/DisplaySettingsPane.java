/*******************************************************************************
 * Copyright (c) 2012-2013 Biomedical Image Group (BIG), EPFL, Switzerland.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Public License v3.0
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/gpl.html
 * 
 * Contributors:
 *     Ricard Delgado-Gonzalo (ricard.delgado@gmail.com)
 *     Nicolas Chenouard (nicolas.chenouard@gmail.com)
 *     Philippe Th&#233;venaz (philippe.thevenaz@epfl.ch)
 *     Emrah Bostan (emrah.bostan@gmail.com)
 *     Ulugbek S. Kamilov (kamilov@gmail.com)
 *     Ramtin Madani (ramtin_madani@yahoo.com)
 *     Masih Nilchian (masih_n85@yahoo.com)
 *     C&#233;dric Vonesch (cedric.vonesch@epfl.ch)
 *     Virginie Uhlmann (virginie.uhlmann@epfl.ch)
 *     Cl&#233;ment Marti (clement.marti@epfl.ch)
 *     Julien Jacquemot (julien.jacquemot@epfl.ch)
 ******************************************************************************/
package plugins.big.vascular.gui;

import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.GridLayout;
import java.awt.Insets;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;

import javax.swing.DefaultComboBoxModel;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSpinner;
import javax.swing.SpinnerNumberModel;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import plugins.big.bigsnakeutils.icy.gui.CollapsiblePanel;
import plugins.big.bigsnakeutils.icy.gui.DetailPanelMode;
import plugins.big.vascular.BIGVascular;
import plugins.big.vascular.core.DisplaySettings;
import plugins.big.vascular.core.Settings;

/**
 * Panel in which the user specifies the display parameters.
 * 
 * @version May 6, 2013
 * 
 * @author Ricard Delgado-Gonzalo (ricard.delgado@gmail.com)
 */
public class DisplaySettingsPane extends CollapsiblePanel implements
		ChangeListener, ItemListener {

	/**
	 * Determines if a de-serialized file is compatible with this class.
	 * 
	 * Maintainers must change this value if and only if the new version of this
	 * class is not compatible with old versions. See Sun docs for <a
	 * href=http://java.sun.com/products/jdk/1.1/docs/guide
	 * /serialization/spec/version.doc.html> details. </a>
	 * 
	 * Not necessary to include in first version of the class, but included here
	 * as a reminder of its importance.
	 */
	private static final long serialVersionUID = -2078162621447160540L;

	// ----------------------------------------------------------------------------
	// INPUT OBJECTS

	private final JComboBox meshResolutionComboBox_ = new JComboBox();
	private final JSpinner meshStrokeThicknessSpinner_ = new JSpinner();
	private final JSpinner meshDepthTransparencySpinner_ = new JSpinner();
	private final JCheckBox refreshDuringOptimizationCheckbox_ = new JCheckBox(
			"Show evolution");
	private final JCheckBox refreshSnakeDrag_ = new JCheckBox(
			"Show drag");

	// ----------------------------------------------------------------------------
	// LABELS

	private final JLabel meshResolutionLabel_ = new JLabel("Mesh resolution");
	private final JLabel meshStrokeThicknessLabel_ = new JLabel(
			"Mesh thickness");
	private final JLabel meshDepthTransparencyLabel_ = new JLabel(
			"Mesh transparency");

	// ----------------------------------------------------------------------------
	// OTHER

	/** Reference to the main plug-in. */
	private BIGVascular bigSnake_ = null;

	// ============================================================================
	// PUBLIC METHODS

	/** Default constructor. */
	public DisplaySettingsPane(String title, BIGVascular bigSnake) {
		super(title);
		bigSnake_ = bigSnake;

		GridBagLayout gridBagLayout = new GridBagLayout();
		gridBagLayout.columnWidths = new int[] { 120, 150, 0 };
		gridBagLayout.rowHeights = new int[] { 27, 28, 27, 56, 23 };
		gridBagLayout.columnWeights = new double[] { 0.0, 1.0, Double.MIN_VALUE };
		gridBagLayout.rowWeights = new double[] { 0.0, 0.0, 0.0, 0.0,
				Double.MIN_VALUE };
		setLayout(gridBagLayout);

		JPanel meshResolutionLabelPanel = new JPanel();
		GridBagConstraints gbc_meshResolutionLabelPanel = new GridBagConstraints();
		gbc_meshResolutionLabelPanel.anchor = GridBagConstraints.EAST;
		gbc_meshResolutionLabelPanel.insets = new Insets(0, 0, 5, 5);
		gbc_meshResolutionLabelPanel.fill = GridBagConstraints.VERTICAL;
		gbc_meshResolutionLabelPanel.gridx = 0;
		gbc_meshResolutionLabelPanel.gridy = 0;
		add(meshResolutionLabelPanel, gbc_meshResolutionLabelPanel,
				DetailPanelMode.ADVANCED);
		meshResolutionLabelPanel.setLayout(new GridLayout(0, 1, 0, 0));

		meshResolutionLabelPanel.add(meshResolutionLabel_);

		JPanel meshResolutionComboBoxPanel = new JPanel();
		GridBagConstraints gbc_meshResolutionComboBoxPanel = new GridBagConstraints();
		gbc_meshResolutionComboBoxPanel.insets = new Insets(0, 0, 5, 0);
		gbc_meshResolutionComboBoxPanel.fill = GridBagConstraints.BOTH;
		gbc_meshResolutionComboBoxPanel.gridx = 1;
		gbc_meshResolutionComboBoxPanel.gridy = 0;
		add(meshResolutionComboBoxPanel, gbc_meshResolutionComboBoxPanel,
				DetailPanelMode.ADVANCED);
		meshResolutionComboBoxPanel.setLayout(new GridLayout(0, 1, 0, 0));

		meshResolutionComboBoxPanel.add(meshResolutionComboBox_);
		String[] meshResolutionStrings = new String[] { "High", "Normal", "Low" };
		meshResolutionComboBox_.setModel(new DefaultComboBoxModel(
				meshResolutionStrings));

		switch (Settings.MESH_RESOLUTION_DEFAULT) {
		case HIGH:
			meshResolutionComboBox_.setSelectedIndex(0);
			break;
		case NORMAL:
			meshResolutionComboBox_.setSelectedIndex(1);
			break;
		case LOW:
			meshResolutionComboBox_.setSelectedIndex(2);
			break;
		}

		JPanel meshStrokeThicknessLabelPanel = new JPanel();
		GridBagConstraints gbc_meshStrokeThicknessLabelPanel = new GridBagConstraints();
		gbc_meshStrokeThicknessLabelPanel.anchor = GridBagConstraints.EAST;
		gbc_meshStrokeThicknessLabelPanel.insets = new Insets(0, 0, 5, 5);
		gbc_meshStrokeThicknessLabelPanel.fill = GridBagConstraints.VERTICAL;
		gbc_meshStrokeThicknessLabelPanel.gridx = 0;
		gbc_meshStrokeThicknessLabelPanel.gridy = 1;
		add(meshStrokeThicknessLabelPanel, gbc_meshStrokeThicknessLabelPanel,
				DetailPanelMode.ADVANCED);
		meshStrokeThicknessLabelPanel.setLayout(new GridLayout(0, 1, 0, 0));

		meshStrokeThicknessLabelPanel.add(meshStrokeThicknessLabel_);

		JPanel meshStrokeThicknessSpinnerPanel = new JPanel();
		GridBagConstraints gbc_meshStrokeThicknessSpinnerPanel = new GridBagConstraints();
		gbc_meshStrokeThicknessSpinnerPanel.insets = new Insets(0, 0, 5, 0);
		gbc_meshStrokeThicknessSpinnerPanel.fill = GridBagConstraints.BOTH;
		gbc_meshStrokeThicknessSpinnerPanel.gridx = 1;
		gbc_meshStrokeThicknessSpinnerPanel.gridy = 1;
		add(meshStrokeThicknessSpinnerPanel,
				gbc_meshStrokeThicknessSpinnerPanel, DetailPanelMode.ADVANCED);
		meshStrokeThicknessSpinnerPanel.setLayout(new GridLayout(0, 1, 0, 0));

		meshStrokeThicknessSpinnerPanel.add(meshStrokeThicknessSpinner_);
		meshStrokeThicknessSpinner_.setModel(new SpinnerNumberModel(
				new Integer(Settings.STROKE_THICKNESS_DEFAULT), new Integer(1),
				new Integer(10), new Integer(1)));

		JPanel meshDepthTransparencyLabelPanel = new JPanel();
		GridBagConstraints gbc_meshDepthTransparencyLabelPanel = new GridBagConstraints();
		gbc_meshDepthTransparencyLabelPanel.anchor = GridBagConstraints.EAST;
		gbc_meshDepthTransparencyLabelPanel.insets = new Insets(0, 0, 5, 5);
		gbc_meshDepthTransparencyLabelPanel.fill = GridBagConstraints.VERTICAL;
		gbc_meshDepthTransparencyLabelPanel.gridx = 0;
		gbc_meshDepthTransparencyLabelPanel.gridy = 2;
		add(meshDepthTransparencyLabelPanel,
				gbc_meshDepthTransparencyLabelPanel, DetailPanelMode.ADVANCED);
		meshDepthTransparencyLabelPanel.setLayout(new GridLayout(0, 1, 0, 0));

		meshDepthTransparencyLabelPanel.add(meshDepthTransparencyLabel_);

		JPanel meshDepthTransparencySpinnerPanel = new JPanel();
		GridBagConstraints gbc_meshDepthTransparencySpinnerPanel = new GridBagConstraints();
		gbc_meshDepthTransparencySpinnerPanel.insets = new Insets(0, 0, 5, 0);
		gbc_meshDepthTransparencySpinnerPanel.fill = GridBagConstraints.BOTH;
		gbc_meshDepthTransparencySpinnerPanel.gridx = 1;
		gbc_meshDepthTransparencySpinnerPanel.gridy = 2;
		add(meshDepthTransparencySpinnerPanel,
				gbc_meshDepthTransparencySpinnerPanel, DetailPanelMode.ADVANCED);
		meshDepthTransparencySpinnerPanel.setLayout(new GridLayout(0, 1, 0, 0));
		meshDepthTransparencySpinnerPanel.add(meshDepthTransparencySpinner_);
		meshDepthTransparencySpinner_.setModel(new SpinnerNumberModel(
				new Integer(Settings.DEPTH_TRANSPARENCY_DEFAULT),
				new Integer(1), new Integer(100), new Integer(1)));

		JPanel refreshDuringOptimizationPanel = new JPanel();
		GridBagConstraints gbc_refreshDuringOptimizationPanel = new GridBagConstraints();
		gbc_refreshDuringOptimizationPanel.fill = GridBagConstraints.BOTH;
		gbc_refreshDuringOptimizationPanel.gridx = 1;
		gbc_refreshDuringOptimizationPanel.gridy = 3;
		add(refreshDuringOptimizationPanel, gbc_refreshDuringOptimizationPanel,
				DetailPanelMode.ADVANCED);
		refreshDuringOptimizationPanel.setLayout(new GridLayout(0, 1, 0, 0));
		refreshDuringOptimizationPanel.add(refreshDuringOptimizationCheckbox_);
		refreshDuringOptimizationPanel.add(refreshSnakeDrag_);
		
		refreshDuringOptimizationCheckbox_
				.setSelected(Settings.REFRESH_SCREEN_DEFAULT);
		refreshSnakeDrag_.setSelected(Settings.REFRESH_SNAKE_DRAG_DEFAULT);

		setVisibility(activeMode_);

		refreshSnakeDrag_.addItemListener(this);
		refreshDuringOptimizationCheckbox_.addItemListener(this);
		meshDepthTransparencySpinner_.addChangeListener(this);
		meshResolutionComboBox_.addItemListener(this);
		meshStrokeThicknessSpinner_.addChangeListener(this);
	}

	// ----------------------------------------------------------------------------

	public DisplaySettings getDisplaySettings() {
		Settings.MeshResolution meshResolution_ = null;
		switch (meshResolutionComboBox_.getSelectedIndex()) {
		case 0:
			meshResolution_ = Settings.MeshResolution.HIGH;
			break;
		case 1:
			meshResolution_ = Settings.MeshResolution.NORMAL;
			break;
		case 2:
			meshResolution_ = Settings.MeshResolution.LOW;
			break;
		default:
			System.err.println("Error loading display parameters");
			return null;
		}
		return new DisplaySettings(
				refreshSnakeDrag_.isSelected(),
				refreshDuringOptimizationCheckbox_.isSelected(),
				(Integer) meshStrokeThicknessSpinner_.getValue(),
				(Integer) meshDepthTransparencySpinner_.getValue(),
				meshResolution_);
	}

	// ----------------------------------------------------------------------------
	// CHANGELISTENER METHODS

	/** Handles the events of the spinners. */
	@Override
	public void stateChanged(ChangeEvent e) {
		pushParameters();
	}

	// ----------------------------------------------------------------------------
	// ITEMLISTENER METHODS

	/**
	 * Handles the events of the combo boxes and the checkbox. It fires an
	 * update of the active snake.
	 */
	@Override
	public void itemStateChanged(ItemEvent e) {
		pushParameters();
	}

	// ============================================================================
	// PRIVATE METHODS

	private void pushParameters() {
		DisplaySettings displaySettings = getDisplaySettings();
		if (displaySettings != null) {
			bigSnake_.setDisplaySettings(displaySettings);
		}
	}
}
