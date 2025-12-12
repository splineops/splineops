
/*******************************************************************************
 * Copyright (c) 2012-2015 Biomedical Image Group (BIG), EPFL, Switzerland.
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
package plugins.big.bigsnake3d.gui;


import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.GridLayout;
import java.awt.Insets;
import java.awt.Panel;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JPanel;
import plugins.big.bigsnake3d.BIGSnake3DMSBrain;;

public class SnakeInteractionPaneBrain extends Panel implements ActionListener {

	private static final long serialVersionUID = 1437522417562022731L;

	/** Label for the selected snake combo box. */
	private final JLabel selectedSnakeLabel_ = new JLabel("Selected Snake");

	/** Combo box where the user selects the snake to interact with. */
	private JComboBox selectedSnakeComboBox_ = new JComboBox();

	private String[] snakeList = null;
	private BIGSnake3DMSBrain mBigsnake;
	private int mSelectedSnakeIndex;

	// ============================================================================
	// PUBLIC METHODS

	/** Constructor. */
	public SnakeInteractionPaneBrain(String title, BIGSnake3DMSBrain bigSnake) {
		super();

		this.mBigsnake = bigSnake;
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
	/** Handles the events of the combo box. */
	@Override
	public void actionPerformed(ActionEvent e) {
		if (e.getSource() == selectedSnakeComboBox_) {
			mSelectedSnakeIndex = selectedSnakeComboBox_.getSelectedIndex();
			changeSelectedSnake(mSelectedSnakeIndex);
		}
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
}
