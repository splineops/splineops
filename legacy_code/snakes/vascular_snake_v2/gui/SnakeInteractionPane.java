
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
 *     Christophe Gaudet-Blavignac (chrisgaubla@gmail.com)
 ******************************************************************************/
package plugins.big.vascular.gui;

import icy.sequence.Sequence;

import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.GridLayout;
import java.awt.Insets;
import java.awt.Panel;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;

import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JPanel;

import com.sun.media.ui.ComboBox;

import plugins.big.bigsnakeutils.BIGSnakeUtils;
import plugins.big.vascular.BIGVascular;
import plugins.big.vascular.core.Settings;

/**
 * Panel in which the user specifies the parameters of the Sphere-Snake.
 * 
 * @version January 17, 2013
 * 
 * @author Ricard Delgado-Gonzalo (ricard.delgado@gmail.com)
 * @author Nicolas Chenouard (nicolas.chenouard@gmail.com)
 * @author Christophe Gaudet-Blavignac (chrisgaubla@gmail.com)
 * 
 */
public class SnakeInteractionPane extends Panel implements ActionListener, ItemListener {

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

	/** Label for the selected snake combo box. */
	private final JLabel selectedSnakeLabel_ = new JLabel("Selected Snake");

	/** Combo box where the user selects the snake to interact with. */
	private JComboBox selectedSnakeComboBox_ = new JComboBox();
	


	private String[] snakeList = null;
	private BIGVascular mBigsnake;
	private int mSizeOfKeeperList;
	private int mSelectedSnakeIndex;

	// ============================================================================
	// PUBLIC METHODS

	/** Constructor. */
	public SnakeInteractionPane(String title, BIGVascular bigSnake) {
		super();

		this.mBigsnake = bigSnake;
		this.mSizeOfKeeperList = 0;
		this.mSelectedSnakeIndex = 0;

		GridBagLayout gridBagLayout = new GridBagLayout();
		gridBagLayout.columnWidths = new int[] { 120, 150, 0 };
		gridBagLayout.rowHeights = new int[] { 27, 27, 28, 0 };
		gridBagLayout.columnWeights = new double[] { 0.0, 1.0, Double.MIN_VALUE };
		gridBagLayout.rowWeights = new double[] { 0.0, 0.0, 0.0, 0.0,
				Double.MIN_VALUE };
		setLayout(gridBagLayout);

		JPanel selectedSnakeLabelPanel = new JPanel();
		GridBagConstraints gbc_panel_4 = new GridBagConstraints();
		gbc_panel_4.anchor = GridBagConstraints.EAST;
		gbc_panel_4.insets = new Insets(0, 0, 5, 5);
		gbc_panel_4.fill = GridBagConstraints.VERTICAL;
		gbc_panel_4.gridx = 0;
		gbc_panel_4.gridy = 1;
		add(selectedSnakeLabelPanel, gbc_panel_4);
		selectedSnakeLabelPanel.setLayout(new GridLayout(0, 1, 0, 0));

		selectedSnakeLabelPanel.add(selectedSnakeLabel_);

		JPanel selectedSnakeComboBoxPanel = new JPanel();
		GridBagConstraints gbc_panel_1 = new GridBagConstraints();
		gbc_panel_1.insets = new Insets(0, 0, 5, 0);
		gbc_panel_1.fill = GridBagConstraints.BOTH;
		gbc_panel_1.gridx = 1;
		gbc_panel_1.gridy = 1;
		add(selectedSnakeComboBoxPanel, gbc_panel_1);
		selectedSnakeComboBoxPanel.setLayout(new GridLayout(0, 1, 0, 0));

		selectedSnakeComboBoxPanel.add(selectedSnakeComboBox_);
		


		selectedSnakeComboBox_.addActionListener(this);

	}

	// ----------------------------------------------------------------------------

	// ----------------------------------------------------------------------------

	/** Erases from internal tables the entries of a particular image. */
	public void sequenceClosed(Sequence sequence) {
		if (sequence != null) {
		}
	}

	// ----------------------------------------------------------------------------

	/** Handles the events of the combo box. */
	@Override
	public void actionPerformed(ActionEvent e) {
		if (e.getSource() == selectedSnakeComboBox_) {
			mSelectedSnakeIndex = selectedSnakeComboBox_.getSelectedIndex();
			changeSelectedSnake(mSelectedSnakeIndex);
		}
	}
	@Override
	public void itemStateChanged(ItemEvent arg0) {
		
		
	}

	private void changeSelectedSnake(int selectedSnakeIndex) {
		mBigsnake.activateSnakeFromIndex(selectedSnakeIndex);;
	}

	public void updateComboBox(int numberOfSnakesAlive) {
		selectedSnakeComboBox_.removeAllItems();
		snakeList = new String[numberOfSnakesAlive];
		for (int i = 0; i < numberOfSnakesAlive; i++) {
			snakeList[i] = "Snake number : " + (i + 1);
			selectedSnakeComboBox_.addItem(snakeList[i]);
		}
		if(snakeList.length-1 >=0){
		selectedSnakeComboBox_.setSelectedItem(snakeList[snakeList.length-1]);
		}
	}

	// ============================================================================
	// PRIVATE METHODS

	/**
	 * Updates the content of the combo box associated to the selection of the
	 * channel.
	 */

}
