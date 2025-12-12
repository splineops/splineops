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

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.RenderingHints;
import java.awt.Toolkit;

import javax.swing.JDialog;
import javax.swing.JLabel;
import javax.swing.JPanel;

import plugins.big.bigsnake3d.core.SettingsMSBrain;
import plugins.big.bigsnake3d.rsc.ResourceUtilMSBrain;

/**
 * Frame that contains the information of the developers.
 * 
 * @version November 19, 2014
 * 
 * @author Ricard Delgado-Gonzalo (ricard.delgado@gmail.com)
 */
public class CreditsPaneMSBrain extends JDialog {

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
	private static final long serialVersionUID = 4359134367179472853L;

	/** About image. */
	private Image image_ = null;

	/** About panel. */
	private About about_;

	/** Determines if the image could be found. */
	private boolean imageFound_ = false;

	// ============================================================================
	// PUBLIC METHODS

	/** Default constructor. */
	public CreditsPaneMSBrain() {
		super();
		setModal(true);

		SettingsMSBrain settings = SettingsMSBrain.getInstance();
		ResourceUtilMSBrain resourceUtil = ResourceUtilMSBrain.getInstance();

		setTitle(settings.getAppName() + " " + settings.getAppVersion());
		setBackground(Color.WHITE);
		setAlwaysOnTop(true);

		image_ = resourceUtil.getAboutBanner();

		if (image_ == null) {
			System.err.println("The image could not be located.");
			imageFound_ = false;
			return;
		} else {
			imageFound_ = true;
		}
		image_.flush();

		about_ = new About();
		about_.setLayout(null);

		int width = image_.getWidth(null);
		int height = image_.getHeight(null);

		Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
		setLocation(screenSize.width / 2 - (width / 2), screenSize.height / 2
				- (height / 2));
		getContentPane().setLayout(new BorderLayout(0, 0));

		about_.setPreferredSize(new Dimension(width, height));
		getContentPane().add(about_, BorderLayout.CENTER);

		JLabel info = new JLabel(
				"<html>Project website: http://bigwww.epfl.ch/<br></html>");

		getContentPane().add(info, BorderLayout.SOUTH);
		pack();
		settings.setWindowIcon(this);
	}

	// ----------------------------------------------------------------------------

	/** Makes the panel visible. */
	public void run() {
		if (imageFound_) {
			setResizable(false);
			setVisible(true);
		}
	}

	// ============================================================================
	// INNER CLASS

	/** JPanel on which the about image is paint on. */
	private class About extends JPanel {

		/**
		 * Determines if a de-serialized file is compatible with this class.
		 * 
		 * Maintainers must change this value if and only if the new version of
		 * this class is not compatible with old versions. See Sun docs for <a
		 * href=http://java.sun.com/products/jdk/1.1/docs/guide
		 * /serialization/spec/version.doc.html> details. </a>
		 * 
		 * Not necessary to include in first version of the class, but included
		 * here as a reminder of its importance.
		 */
		private static final long serialVersionUID = -3653965982443315943L;

		// ============================================================================
		// PUBLIC METHODS

		/** Paint the given image as background of the panel. */
		@Override
		public void paintComponent(Graphics g) {
			Graphics2D g2 = (Graphics2D) g;
			g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING,
					RenderingHints.VALUE_ANTIALIAS_ON); // anti-aliasing
			g2.setRenderingHint(RenderingHints.KEY_RENDERING,
					RenderingHints.VALUE_RENDER_QUALITY);
			g2.drawImage(image_, 0, 0, this);
		}
	}
}
