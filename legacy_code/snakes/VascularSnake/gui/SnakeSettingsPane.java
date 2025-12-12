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
import plugins.big.vascular.snake.CylinderSnakeEnergyType;
import plugins.big.vascular.snake.CylinderSnakeParameters;
import plugins.big.vascular.snake.CylinderSnakeTargetType;

/**
 * Panel in which the user specifies the parameters of the Sphere-Snake.
 * 
 * @version January 17, 2013
 * 
 * @author Ricard Delgado-Gonzalo (ricard.delgado@gmail.com)
 * @author Nicolas Chenouard (nicolas.chenouard@gmail.com)
 */
public class SnakeSettingsPane extends CollapsiblePanel implements
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

	private final JComboBox targetBrightnessComboBox_ = new JComboBox();

	private final JSpinner numControlPointsSpinner_ = new JSpinner();

	private final JComboBox energyTypeComboBox_ = new JComboBox();

	private final JSpinner alphaSpinner_ = new JSpinner();

	private final JSpinner gammaSpinner_ = new JSpinner();

	private final JSpinner maxIterationsSpinner_ = new JSpinner();

	private final JCheckBox immortalCheckbox_ = new JCheckBox("Immortal");

	private final JLabel targetBrightnessLabel_ = new JLabel(
			"Target brightness");;

	private final JLabel numControlPointsLabel_ = new JLabel("Control points");

	private final JLabel energyTypeLabel_ = new JLabel("Energy type");

	private final JLabel alphaLabel_ = new JLabel("Alpha");

	private final JLabel gammaLabel_ = new JLabel("Gamma");

	private final JLabel maxIterationsLabel_ = new JLabel("Max iterations");

	private final String[] energyTypeStrings_ = new String[] { "Contour",
			"Region", "Mixture" };

	private final String[] detectTypeStrings_ = new String[] { "Dark", "Bright" };

	private BIGVascular bigSnake_ = null;

	// ============================================================================
	// PUBLIC METHODS

	/** Default constructor. */
	public SnakeSettingsPane(String title, BIGVascular bigSnake) {
		super(title);
		bigSnake_ = bigSnake;

		GridBagLayout gridBagLayout = new GridBagLayout();
		gridBagLayout.columnWidths = new int[] { 120, 150, 0 };
		gridBagLayout.rowHeights = new int[] { 27, 28, 27, 28, 28, 28, 23, 0 };
		gridBagLayout.columnWeights = new double[] { 0.0, 1.0, Double.MIN_VALUE };
		gridBagLayout.rowWeights = new double[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
				Double.MIN_VALUE };
		setLayout(gridBagLayout);

		JPanel targetBrightnessLabelPanel = new JPanel();
		GridBagConstraints gbc_targetBrightnessLabelPanel = new GridBagConstraints();
		gbc_targetBrightnessLabelPanel.anchor = GridBagConstraints.EAST;
		gbc_targetBrightnessLabelPanel.insets = new Insets(0, 0, 5, 5);
		gbc_targetBrightnessLabelPanel.fill = GridBagConstraints.VERTICAL;
		gbc_targetBrightnessLabelPanel.gridx = 0;
		gbc_targetBrightnessLabelPanel.gridy = 0;
		add(targetBrightnessLabelPanel, gbc_targetBrightnessLabelPanel,
				DetailPanelMode.BASICS);
		targetBrightnessLabelPanel.setLayout(new GridLayout(0, 1, 0, 0));

		targetBrightnessLabelPanel.add(targetBrightnessLabel_);

		JPanel targetBrightnessComboBoxPanel = new JPanel();
		GridBagConstraints gbc_targetBrightnessComboBoxPanel = new GridBagConstraints();
		gbc_targetBrightnessComboBoxPanel.insets = new Insets(0, 0, 5, 0);
		gbc_targetBrightnessComboBoxPanel.fill = GridBagConstraints.BOTH;
		gbc_targetBrightnessComboBoxPanel.gridx = 1;
		gbc_targetBrightnessComboBoxPanel.gridy = 0;
		add(targetBrightnessComboBoxPanel, gbc_targetBrightnessComboBoxPanel,
				DetailPanelMode.BASICS);
		targetBrightnessComboBoxPanel.setLayout(new GridLayout(0, 1, 0, 0));

		targetBrightnessComboBoxPanel.add(targetBrightnessComboBox_);
		targetBrightnessComboBox_.setModel(new DefaultComboBoxModel(
				detectTypeStrings_));
		targetBrightnessComboBox_.setSelectedIndex(0);

		JPanel numControlPointsLabelPanel = new JPanel();
		GridBagConstraints gbc_numControlPointsLabelPanel = new GridBagConstraints();
		gbc_numControlPointsLabelPanel.anchor = GridBagConstraints.EAST;
		gbc_numControlPointsLabelPanel.insets = new Insets(0, 0, 5, 5);
		gbc_numControlPointsLabelPanel.fill = GridBagConstraints.VERTICAL;
		gbc_numControlPointsLabelPanel.gridx = 0;
		gbc_numControlPointsLabelPanel.gridy = 1;
		add(numControlPointsLabelPanel, gbc_numControlPointsLabelPanel,
				DetailPanelMode.BASICS);
		numControlPointsLabelPanel.setLayout(new GridLayout(0, 1, 0, 0));

		numControlPointsLabelPanel.add(numControlPointsLabel_);

		JPanel numControlPointsSpinnerPanel = new JPanel();
		GridBagConstraints gbc_numControlPointsSpinnerPanel = new GridBagConstraints();
		gbc_numControlPointsSpinnerPanel.insets = new Insets(0, 0, 5, 0);
		gbc_numControlPointsSpinnerPanel.fill = GridBagConstraints.BOTH;
		gbc_numControlPointsSpinnerPanel.gridx = 1;
		gbc_numControlPointsSpinnerPanel.gridy = 1;
		add(numControlPointsSpinnerPanel, gbc_numControlPointsSpinnerPanel,
				DetailPanelMode.BASICS);
		numControlPointsSpinnerPanel.setLayout(new GridLayout(0, 1, 0, 0));

		numControlPointsSpinnerPanel.add(numControlPointsSpinner_);
		numControlPointsSpinner_
				.setModel(new SpinnerNumberModel(new Integer(4),
						new Integer(3), new Integer(10), new Integer(1)));

		JPanel energyTypeLabelPanel = new JPanel();
		GridBagConstraints gbc_energyTypeLabelPanel = new GridBagConstraints();
		gbc_energyTypeLabelPanel.anchor = GridBagConstraints.EAST;
		gbc_energyTypeLabelPanel.insets = new Insets(0, 0, 5, 5);
		gbc_energyTypeLabelPanel.fill = GridBagConstraints.VERTICAL;
		gbc_energyTypeLabelPanel.gridx = 0;
		gbc_energyTypeLabelPanel.gridy = 2;
		add(energyTypeLabelPanel, gbc_energyTypeLabelPanel,
				DetailPanelMode.ADVANCED);
		energyTypeLabelPanel.setLayout(new GridLayout(0, 1, 0, 0));

		energyTypeLabelPanel.add(energyTypeLabel_);

		JPanel energyTypeComboBoxPanel = new JPanel();
		GridBagConstraints gbc_energyTypeComboBoxPanel = new GridBagConstraints();
		gbc_energyTypeComboBoxPanel.insets = new Insets(0, 0, 5, 0);
		gbc_energyTypeComboBoxPanel.fill = GridBagConstraints.BOTH;
		gbc_energyTypeComboBoxPanel.gridx = 1;
		gbc_energyTypeComboBoxPanel.gridy = 2;
		add(energyTypeComboBoxPanel, gbc_energyTypeComboBoxPanel,
				DetailPanelMode.ADVANCED);
		energyTypeComboBoxPanel.setLayout(new GridLayout(0, 1, 0, 0));

		energyTypeComboBoxPanel.add(energyTypeComboBox_);
		energyTypeComboBox_.setModel(new DefaultComboBoxModel(
				energyTypeStrings_));
		energyTypeComboBox_.setSelectedIndex(0);

		JPanel alphaLabelPanel = new JPanel();
		GridBagConstraints gbc_alphaLabelPanel = new GridBagConstraints();
		gbc_alphaLabelPanel.anchor = GridBagConstraints.EAST;
		gbc_alphaLabelPanel.insets = new Insets(0, 0, 5, 5);
		gbc_alphaLabelPanel.fill = GridBagConstraints.VERTICAL;
		gbc_alphaLabelPanel.gridx = 0;
		gbc_alphaLabelPanel.gridy = 3;
		add(alphaLabelPanel, gbc_alphaLabelPanel, DetailPanelMode.ADVANCED);
		alphaLabelPanel.setLayout(new GridLayout(0, 1, 0, 0));

		alphaLabelPanel.add(alphaLabel_);

		JPanel alphaSpinnerPanel = new JPanel();
		GridBagConstraints gbc_alphaSpinnerPanel = new GridBagConstraints();
		gbc_alphaSpinnerPanel.insets = new Insets(0, 0, 5, 0);
		gbc_alphaSpinnerPanel.fill = GridBagConstraints.BOTH;
		gbc_alphaSpinnerPanel.gridx = 1;
		gbc_alphaSpinnerPanel.gridy = 3;
		add(alphaSpinnerPanel, gbc_alphaSpinnerPanel, DetailPanelMode.ADVANCED);
		alphaSpinnerPanel.setLayout(new GridLayout(0, 1, 0, 0));

		alphaSpinnerPanel.add(alphaSpinner_);
		alphaSpinner_.setModel(new SpinnerNumberModel(0.0, 0.0, 1.0, 0.01));

		alphaSpinner_.setEnabled(energyTypeComboBox_.getSelectedItem()
				.toString().compareTo(energyTypeStrings_[2]) == 0);

		JPanel gammaLabelPanel = new JPanel();
		GridBagConstraints gbc_gammaLabelPanel = new GridBagConstraints();
		gbc_gammaLabelPanel.anchor = GridBagConstraints.EAST;
		gbc_gammaLabelPanel.insets = new Insets(0, 0, 5, 5);
		gbc_gammaLabelPanel.fill = GridBagConstraints.VERTICAL;
		gbc_gammaLabelPanel.gridx = 0;
		gbc_gammaLabelPanel.gridy = 4;
		add(gammaLabelPanel, gbc_gammaLabelPanel, DetailPanelMode.ADVANCED);
		gammaLabelPanel.setLayout(new GridLayout(0, 1, 0, 0));

		gammaLabelPanel.add(gammaLabel_);

		JPanel gammaSpinnerPanel = new JPanel();
		GridBagConstraints gbc_gammaSpinnerPanel = new GridBagConstraints();
		gbc_gammaSpinnerPanel.insets = new Insets(0, 0, 5, 0);
		gbc_gammaSpinnerPanel.fill = GridBagConstraints.BOTH;
		gbc_gammaSpinnerPanel.gridx = 1;
		gbc_gammaSpinnerPanel.gridy = 4;
		add(gammaSpinnerPanel, gbc_gammaSpinnerPanel, DetailPanelMode.ADVANCED);
		gammaSpinnerPanel.setLayout(new GridLayout(0, 1, 0, 0));

		gammaSpinnerPanel.add(gammaSpinner_);
		gammaSpinner_.setModel(new SpinnerNumberModel(10000, 0.0,
				Double.MAX_VALUE, 1000));

		JPanel maxIterationsLabelPanel = new JPanel();
		GridBagConstraints gbc_maxIterationsLabelPanel = new GridBagConstraints();
		gbc_maxIterationsLabelPanel.anchor = GridBagConstraints.EAST;
		gbc_maxIterationsLabelPanel.insets = new Insets(0, 0, 5, 5);
		gbc_maxIterationsLabelPanel.fill = GridBagConstraints.VERTICAL;
		gbc_maxIterationsLabelPanel.gridx = 0;
		gbc_maxIterationsLabelPanel.gridy = 5;
		add(maxIterationsLabelPanel, gbc_maxIterationsLabelPanel,
				DetailPanelMode.ADVANCED);
		maxIterationsLabelPanel.setLayout(new GridLayout(0, 1, 0, 0));

		maxIterationsLabelPanel.add(maxIterationsLabel_);

		JPanel maxIterationsSpinnerPanel = new JPanel();
		GridBagConstraints gbc_maxIterationsSpinnerPanel = new GridBagConstraints();
		gbc_maxIterationsSpinnerPanel.insets = new Insets(0, 0, 5, 0);
		gbc_maxIterationsSpinnerPanel.fill = GridBagConstraints.BOTH;
		gbc_maxIterationsSpinnerPanel.gridx = 1;
		gbc_maxIterationsSpinnerPanel.gridy = 5;
		add(maxIterationsSpinnerPanel, gbc_maxIterationsSpinnerPanel,
				DetailPanelMode.ADVANCED);
		maxIterationsSpinnerPanel.setLayout(new GridLayout(0, 1, 0, 0));

		maxIterationsSpinnerPanel.add(maxIterationsSpinner_);
		maxIterationsSpinner_.setModel(new SpinnerNumberModel(
				new Integer(10000), new Integer(1), null, new Integer(1)));

		JPanel immortalCheckboxPanel = new JPanel();
		immortalCheckboxPanel.setBorder(null);
		GridBagConstraints gbc_immortalCheckboxPanel = new GridBagConstraints();
		gbc_immortalCheckboxPanel.fill = GridBagConstraints.BOTH;
		gbc_immortalCheckboxPanel.gridx = 1;
		gbc_immortalCheckboxPanel.gridy = 6;
		add(immortalCheckboxPanel, gbc_immortalCheckboxPanel,
				DetailPanelMode.ADVANCED);
		immortalCheckboxPanel.setLayout(new GridLayout(0, 1, 0, 0));

		immortalCheckbox_.setSelected(false);
		maxIterationsSpinner_.setEnabled(!immortalCheckbox_.isSelected());

		immortalCheckboxPanel.add(immortalCheckbox_);

		setVisibility(activeMode_);

		numControlPointsSpinner_.addChangeListener(this);
		alphaSpinner_.addChangeListener(this);
		gammaSpinner_.addChangeListener(this);
		maxIterationsSpinner_.addChangeListener(this);
		targetBrightnessComboBox_.addItemListener(this);
		energyTypeComboBox_.addItemListener(this);
		immortalCheckbox_.addItemListener(this);
	}

	// ----------------------------------------------------------------------------

	/** Returns the brightness detection mode selected by the user. */
	public CylinderSnakeTargetType getTargetBrightness() {
		switch (targetBrightnessComboBox_.getSelectedIndex()) {
		case 0:
			return CylinderSnakeTargetType.DARK;
		case 1:
			return CylinderSnakeTargetType.BRIGHT;
		}
		return null;
	}

	// ----------------------------------------------------------------------------

	/** Returns the maximum number of iterations introduced by the user. */
	public Integer getMaxIterations() {
		return (Integer) maxIterationsSpinner_.getValue();
	}

	// ----------------------------------------------------------------------------

	/** Returns the number of control points introduced by the user. */
	public Integer getNumControlPoints() {
		return (Integer) numControlPointsSpinner_.getValue();
	}

	// ----------------------------------------------------------------------------

	/** Returns the energy tradeoff parameter introduced by the user. */
	public Double getAlpha() {
		return (Double) alphaSpinner_.getValue();
	}

	// ----------------------------------------------------------------------------

	/** Returns the weight of the stiffness of the surface. */
	public Double getGamma() {
		return (Double) gammaSpinner_.getValue();
	}

	// ----------------------------------------------------------------------------

	/**
	 * Returns <code>true</code> if the user wants to ignore the maximum number
	 * of iterations, and evolve the snake till convergence. Returns
	 * <code>false</code> otherwise.
	 */
	public boolean isImmortal() {
		return immortalCheckbox_.isSelected();
	}

	// ----------------------------------------------------------------------------

	/** Returns the energy type selected by the user. */
	public CylinderSnakeEnergyType getEnergyType() {
		switch (energyTypeComboBox_.getSelectedIndex()) {
		case 0:
			return CylinderSnakeEnergyType.CONTOUR;
		case 1:
			return CylinderSnakeEnergyType.REGION;
		case 2:
			return CylinderSnakeEnergyType.MIXTURE;
		}
		return null;
	}

	// ----------------------------------------------------------------------------

	/** Sets the snake parameters to the interface. */
	public void setSnakeParameters(CylinderSnakeParameters snakeParameters) {
		numControlPointsSpinner_.setValue(snakeParameters.getM());
		alphaSpinner_.setValue(snakeParameters.getAlpha());
		maxIterationsSpinner_.setValue(snakeParameters.getMaxLife());
		immortalCheckbox_.setSelected(snakeParameters.isImmortal());
		switch (snakeParameters.getEnergyType()) {
		case CONTOUR:
			energyTypeComboBox_.setSelectedIndex(0);
			break;
		case REGION:
			energyTypeComboBox_.setSelectedIndex(1);
			break;
		case MIXTURE:
			energyTypeComboBox_.setSelectedIndex(2);
			break;
		}
		switch (snakeParameters.getDetectType()) {
		case DARK:
			targetBrightnessComboBox_.setSelectedIndex(0);
			break;
		case BRIGHT:
			targetBrightnessComboBox_.setSelectedIndex(1);
			break;
		}
	}

	// ----------------------------------------------------------------------------

	/** Handles the events of the spinners. */
	@Override
	public void stateChanged(ChangeEvent e) {
		if (e.getSource() == numControlPointsSpinner_) {
			bigSnake_.loadSnakeParametersFromInterface();
		} else if (e.getSource() == alphaSpinner_) {
			bigSnake_.loadSnakeParametersFromInterface();
		} else if (e.getSource() == maxIterationsSpinner_) {
			bigSnake_.loadSnakeParametersFromInterface();
		} else if (e.getSource() == gammaSpinner_){
			bigSnake_.loadSnakeParametersFromInterface();
		}
	}

	// ----------------------------------------------------------------------------

	/**
	 * Handles the events of the combo boxes and the checkbox. It fires an
	 * update of the active snake.
	 */
	@Override
	public void itemStateChanged(ItemEvent e) {
		if (e.getSource() == targetBrightnessComboBox_) {
			if (e.getStateChange() == ItemEvent.SELECTED)
				bigSnake_.loadSnakeParametersFromInterface();
		} else if (e.getSource() == energyTypeComboBox_) {
			if (e.getStateChange() == ItemEvent.SELECTED) {
				alphaSpinner_.setEnabled(energyTypeComboBox_.getSelectedItem()
						.toString().compareTo(energyTypeStrings_[2]) == 0);
				bigSnake_.loadSnakeParametersFromInterface();
			}
		} else if (e.getSource() == immortalCheckbox_) {
			maxIterationsSpinner_.setEnabled(!immortalCheckbox_.isSelected());
			bigSnake_.loadSnakeParametersFromInterface();
		}
	}
}
