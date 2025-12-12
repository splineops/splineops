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

import icy.main.Icy;
import icy.sequence.Sequence;

import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.GridLayout;
import java.awt.Insets;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.HashMap;

import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSpinner;
import javax.swing.SpinnerNumberModel;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import plugins.big.bigsnakeutils.icy.gui.CollapsiblePanel;
import plugins.big.bigsnakeutils.icy.gui.DetailPanelMode;
import plugins.big.vascular.core.ImageLUTContainer;

/**
 * Panel in which the user specifies image-related parameters.
 * 
 * @version October 30, 2013
 * 
 * @author Ricard Delgado-Gonzalo (ricard.delgado@gmail.com)
 */
public class ImageSettingsPane extends CollapsiblePanel implements
		ActionListener, ChangeListener {

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
	private static final long serialVersionUID = 1437522417562022731L;

	/** Label for the selected image combo box. */
	private final JLabel selectedImageLabel_ = new JLabel("Image");

	/** Label for the selected channel combo box. */
	private final JLabel selectedChannelLabel_ = new JLabel("Channel");

	/**
	 * Label for the spinner that contains the standard deviation of the
	 * Gaussian prefiltering.
	 */
	private final JLabel gaussianBlurLabel_ = new JLabel("Smoothing");

	/** Label displaying the name of the selected image. */
	private final JLabel selectedImageName_ = new JLabel("");

	/** Combo box where the user selects the channel from the input image. */
	private final JComboBox selectedChannelComboBox_ = new JComboBox();

	/**
	 * Spinner that contains the standard deviation of the Gaussian
	 * prefiltering.
	 */
	private final JSpinner gaussianBlurSpinner_ = new JSpinner();

	/**
	 * Points to the HashMap of the plug-in that pairs the images and the LUTs.
	 */
	private HashMap<Sequence, ImageLUTContainer> imageLUTs_ = null;

	/** Focused image. */
	private Sequence activeSequence_ = null;

	// ============================================================================
	// PUBLIC METHODS

	/** Constructor. */
	public ImageSettingsPane(String title,
			HashMap<Sequence, ImageLUTContainer> imageLUTs) {
		super(title);
		imageLUTs_ = imageLUTs;

		GridBagLayout gridBagLayout = new GridBagLayout();
		gridBagLayout.columnWidths = new int[] { 120, 150, 0 };
		gridBagLayout.rowHeights = new int[] { 27, 27, 28, 0 };
		gridBagLayout.columnWeights = new double[] { 0.0, 1.0, Double.MIN_VALUE };
		gridBagLayout.rowWeights = new double[] { 0.0, 0.0, 0.0, 0.0,
				Double.MIN_VALUE };
		setLayout(gridBagLayout);

		JPanel selectedImageLabelPanel = new JPanel();
		GridBagConstraints gbc_panel_3 = new GridBagConstraints();
		gbc_panel_3.anchor = GridBagConstraints.EAST;
		gbc_panel_3.insets = new Insets(0, 0, 5, 5);
		gbc_panel_3.fill = GridBagConstraints.VERTICAL;
		gbc_panel_3.gridx = 0;
		gbc_panel_3.gridy = 0;
		add(selectedImageLabelPanel, gbc_panel_3, DetailPanelMode.BASICS);
		selectedImageLabelPanel.setLayout(new GridLayout(0, 1, 0, 0));

		selectedImageLabelPanel.add(selectedImageLabel_);

		JPanel selectedImageComboBoxPanel = new JPanel();
		GridBagConstraints gbc_panel = new GridBagConstraints();
		gbc_panel.insets = new Insets(0, 0, 5, 0);
		gbc_panel.fill = GridBagConstraints.BOTH;
		gbc_panel.gridx = 1;
		gbc_panel.gridy = 0;
		add(selectedImageComboBoxPanel, gbc_panel, DetailPanelMode.BASICS);
		selectedImageComboBoxPanel.setLayout(new GridLayout(0, 1, 0, 0));

		selectedImageComboBoxPanel.add(selectedImageName_);

		JPanel selectedChannelLabelPanel = new JPanel();
		GridBagConstraints gbc_panel_4 = new GridBagConstraints();
		gbc_panel_4.anchor = GridBagConstraints.EAST;
		gbc_panel_4.insets = new Insets(0, 0, 5, 5);
		gbc_panel_4.fill = GridBagConstraints.VERTICAL;
		gbc_panel_4.gridx = 0;
		gbc_panel_4.gridy = 1;
		add(selectedChannelLabelPanel, gbc_panel_4, DetailPanelMode.ADVANCED);
		selectedChannelLabelPanel.setLayout(new GridLayout(0, 1, 0, 0));

		selectedChannelLabelPanel.add(selectedChannelLabel_);

		JPanel selectedChannelComboBoxPanel = new JPanel();
		GridBagConstraints gbc_panel_1 = new GridBagConstraints();
		gbc_panel_1.insets = new Insets(0, 0, 5, 0);
		gbc_panel_1.fill = GridBagConstraints.BOTH;
		gbc_panel_1.gridx = 1;
		gbc_panel_1.gridy = 1;
		add(selectedChannelComboBoxPanel, gbc_panel_1, DetailPanelMode.ADVANCED);
		selectedChannelComboBoxPanel.setLayout(new GridLayout(0, 1, 0, 0));

		selectedChannelComboBoxPanel.add(selectedChannelComboBox_);

		JPanel gaussianBlurLabelPanel = new JPanel();
		GridBagConstraints gbc_panel_5 = new GridBagConstraints();
		gbc_panel_5.anchor = GridBagConstraints.EAST;
		gbc_panel_5.insets = new Insets(0, 0, 5, 5);
		gbc_panel_5.fill = GridBagConstraints.VERTICAL;
		gbc_panel_5.gridx = 0;
		gbc_panel_5.gridy = 2;
		add(gaussianBlurLabelPanel, gbc_panel_5, DetailPanelMode.ADVANCED);
		gaussianBlurLabelPanel.setLayout(new GridLayout(0, 1, 0, 0));

		gaussianBlurLabelPanel.add(gaussianBlurLabel_);

		JPanel gaussianBlurSpinnerPanel = new JPanel();
		GridBagConstraints gbc_gaussianBlurSpinnerPanel = new GridBagConstraints();
		gbc_gaussianBlurSpinnerPanel.insets = new Insets(0, 0, 5, 0);
		gbc_gaussianBlurSpinnerPanel.fill = GridBagConstraints.BOTH;
		gbc_gaussianBlurSpinnerPanel.gridx = 1;
		gbc_gaussianBlurSpinnerPanel.gridy = 2;
		add(gaussianBlurSpinnerPanel, gbc_gaussianBlurSpinnerPanel,
				DetailPanelMode.ADVANCED);
		gaussianBlurSpinnerPanel.setLayout(new GridLayout(0, 1, 0, 0));

		gaussianBlurSpinnerPanel.add(gaussianBlurSpinner_);
		gaussianBlurSpinner_.setModel(new SpinnerNumberModel(2,
				0.0, Double.MAX_VALUE, 1.0));

		JPanel actionPanel = new JPanel();
		GridBagConstraints gbc_panel_7 = new GridBagConstraints();
		gbc_panel_7.fill = GridBagConstraints.BOTH;
		gbc_panel_7.gridx = 1;
		gbc_panel_7.gridy = 3;
		add(actionPanel, gbc_panel_7, DetailPanelMode.BASICS);
		GridBagLayout gbl_panel_7 = new GridBagLayout();
		gbl_panel_7.columnWidths = new int[] { 0, 0, 0 };
		gbl_panel_7.rowHeights = new int[] { 0, 0 };
		gbl_panel_7.columnWeights = new double[] { 1.0, 0.0, Double.MIN_VALUE };
		gbl_panel_7.rowWeights = new double[] { 0.0, Double.MIN_VALUE };
		actionPanel.setLayout(gbl_panel_7);

		setVisibility(activeMode_);

		gaussianBlurSpinner_.addChangeListener(this);
		selectedChannelComboBox_.addActionListener(this);

		sequenceFocused(Icy.getMainInterface().getActiveSequence());
	}

	// ----------------------------------------------------------------------------

	/**
	 * Updates the interface to reflect the information of the image. If the
	 * image was already focused, the associated parameters are loaded.
	 */
	public void sequenceFocused(Sequence sequence) {
		activeSequence_ = sequence;
		gaussianBlurSpinner_.removeChangeListener(this);
		selectedChannelComboBox_.removeActionListener(this);

		if (sequence != null) {
			selectedImageName_.setText(sequence.getName());
			refreshChannelComboBox();
			if (imageLUTs_.containsKey(sequence)) {
				ImageLUTContainer imageLUT = imageLUTs_.get(sequence);
				selectedChannelComboBox_.setSelectedIndex(imageLUT
						.getChannelNumber());
				gaussianBlurSpinner_.setValue(imageLUT.getSigma());
			} else {
				ImageLUTContainer imageLUT = new ImageLUTContainer();
				imageLUT.setSequence(activeSequence_);
				imageLUT.setChannelNumber(selectedChannelComboBox_
						.getSelectedIndex());
				imageLUT.setSigma((Double) gaussianBlurSpinner_.getValue());
				imageLUTs_.put(sequence, imageLUT);
			}
		}
		gaussianBlurSpinner_.addChangeListener(this);
		selectedChannelComboBox_.addActionListener(this);
	}

	// ----------------------------------------------------------------------------

	/** Erases from internal tables the entries of a particular image. */
	public void sequenceClosed(Sequence sequence) {
		if (sequence != null) {
			imageLUTs_.remove(sequence);
		}
	}

	// ----------------------------------------------------------------------------

	/** Handles the events of the spinner. */
	@Override
	public void stateChanged(ChangeEvent e) {
		if (e.getSource() == gaussianBlurSpinner_) {
			if (imageLUTs_.containsKey(activeSequence_)) {
				ImageLUTContainer imageLUT = imageLUTs_.get(activeSequence_);
				imageLUT.setSigma((Double) gaussianBlurSpinner_.getValue());
			} else {
				ImageLUTContainer imageLUT = new ImageLUTContainer();
				imageLUT.setSequence(activeSequence_);
				imageLUT.setChannelNumber(selectedChannelComboBox_
						.getSelectedIndex());
				imageLUT.setSigma((Double) gaussianBlurSpinner_.getValue());
				imageLUTs_.put(activeSequence_, imageLUT);
			}
		}
	}

	// ----------------------------------------------------------------------------

	/** Handles the events of the combo box. */
	@Override
	public void actionPerformed(ActionEvent e) {
		if (e.getSource() == selectedChannelComboBox_) {
			if (imageLUTs_.containsKey(activeSequence_)) {
				ImageLUTContainer imageLUT = imageLUTs_.get(activeSequence_);
				imageLUT.setChannelNumber(selectedChannelComboBox_
						.getSelectedIndex());
			} else {
				// adds the sequence and the parameters to the table
				ImageLUTContainer imageLUT = new ImageLUTContainer();
				imageLUT.setSequence(activeSequence_);
				imageLUT.setChannelNumber(selectedChannelComboBox_
						.getSelectedIndex());
				imageLUT.setSigma((Double) gaussianBlurSpinner_.getValue());
				imageLUTs_.put(activeSequence_, imageLUT);
			}
		}
	}

	// ============================================================================
	// PRIVATE METHODS

	/**
	 * Updates the content of the combo box associated to the selection of the
	 * channel.
	 */
	private void refreshChannelComboBox() {
		selectedChannelComboBox_.removeAllItems();
		if (activeSequence_ != null) {
			int numChannels = activeSequence_.getSizeC();
			for (int i = 0; i < numChannels; i++) {
				selectedChannelComboBox_.addItem(activeSequence_
						.getChannelName(i));
			}
		}
	}
}
