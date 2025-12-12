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
package plugins.big.bigsnake3d.gui;

import icy.gui.frame.IcyFrame;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;

import plugins.big.bigsnake3d.BIGSnake3DMSBrain;
import plugins.big.bigsnake3d.core.SettingsMSBrain;
import plugins.big.bigsnakeutils.system.OpenBrowser;

/**
 * Frame that contains the plug-in user interface.
 * 
 * @version May 3, 2014
 * 
 * @author Ricard Delgado-Gonzalo (ricard.delgado@gmail.com)
 */
public class PlugInFrameMSBrain extends IcyFrame implements ActionListener {

	// ----------------------------------------------------------------------------
	// MENU FIELDS

	/** Menu bar of the frame. */
	private final JMenuBar menuBar_ = new JMenuBar();

	/** File menu in the menu bar. */
	private final JMenu fileMenu_ = new JMenu("File");

	private final JMenu loadSubMenu_ = new JMenu("Load");
	private final JMenuItem loadFromXMLItem_ = new JMenuItem("From XML");
	private final JMenuItem loadFromBinaryImageItem_ = new JMenuItem(
			"From binary image");

	private final JMenu saveSubMenu_ = new JMenu("Save");
	private final JMenuItem saveToXMLItem_ = new JMenuItem("To XML");
	private final JMenuItem saveToBinaryImageItem_ = new JMenuItem(
			"To binary image");

	/** Analyze menu in the menu bar. */
	private final JMenu analyzeMenu_ = new JMenu("Analyze");
	/** Generate mask of the active snake item in the file analyze menu. */
	private final JMenuItem binarizeActiveSnakeItem_ = new JMenuItem(
			"Generate mask of the active snake");

	/** Help menu in the menu bar. */
	private final JMenu helpMenu_ = new JMenu("Help");
	/** Documentation item in the help menu. */
	private final JMenuItem documentationItem_ = new JMenuItem(
			"Documentation (online)");
	/** About item in the help menu. */
	private final JMenuItem aboutItem_ = new JMenuItem("About");

	// ----------------------------------------------------------------------------
	// AUX FIELDS

	/** Reference to the main plug-in class. */
	private BIGSnake3DMSBrain bigSnakePlugin_ = null;

	// ============================================================================
	// PUBLIC METHODS

	/** Constructor. */
	public PlugInFrameMSBrain(BIGSnake3DMSBrain bigSnakePlugin) {
		super("", false, true);
		SettingsMSBrain settings = SettingsMSBrain.getInstance();
		setTitle(settings.getAppName() + " " + settings.getAppVersion());
		settings.setWindowIcon(this);

		bigSnakePlugin_ = bigSnakePlugin;

		setJMenuBar(menuBar_);
		menuBar_.add(fileMenu_);

		fileMenu_.add(loadSubMenu_);
		loadSubMenu_.add(loadFromXMLItem_);
		loadSubMenu_.add(loadFromBinaryImageItem_);

		fileMenu_.add(saveSubMenu_);
		saveSubMenu_.add(saveToXMLItem_);
		saveSubMenu_.add(saveToBinaryImageItem_);

		menuBar_.add(analyzeMenu_);
		analyzeMenu_.add(binarizeActiveSnakeItem_);

		menuBar_.add(helpMenu_);
		helpMenu_.add(documentationItem_);
		helpMenu_.add(aboutItem_);

		loadFromXMLItem_.addActionListener(this);
		loadFromBinaryImageItem_.addActionListener(this);
		saveToXMLItem_.addActionListener(this);
		saveToBinaryImageItem_.addActionListener(this);
		binarizeActiveSnakeItem_.addActionListener(this);
		documentationItem_.addActionListener(this);
		aboutItem_.addActionListener(this);
	}

	// ----------------------------------------------------------------------------

	/** Closes frame and terminates plug-in. */
	@Override
	public void onClosed() {
		bigSnakePlugin_.terminatePlugin();
		super.onClosed();
	}

	// ----------------------------------------------------------------------------

	/** Handles the actions created by the menu bar. */
	@Override
	public void actionPerformed(ActionEvent e) {
		if (e.getSource() == loadFromXMLItem_) {
			bigSnakePlugin_.loadSnakesFromXML();
		} else if (e.getSource() == saveToXMLItem_) {
			bigSnakePlugin_.saveSnakesToXML();
		} else if (e.getSource() == loadFromBinaryImageItem_) {
			bigSnakePlugin_.loadSnakesFromBinaryImage();
		} else if (e.getSource() == saveToBinaryImageItem_) {
			bigSnakePlugin_.saveSnakesToBinaryImage();
		} else if (e.getSource() == documentationItem_) {
			OpenBrowser
					.openURL("http://icy.bioimageanalysis.org/plugin/Active_Cells_3D/");
		} else if (e.getSource() == aboutItem_) {
			CreditsPaneMSBrain about = new CreditsPaneMSBrain();
			about.run();
		} else if (e.getSource() == binarizeActiveSnakeItem_) {
			bigSnakePlugin_.rasterizeActiveSnake();
		}
	}
}
