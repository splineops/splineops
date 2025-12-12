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

import java.util.Random;

/**
 * Class with some tips about how to use the plug-in.
 * 
 * @version September 27, 2012
 * 
 * @author Ricard Delgado-Gonzalo (ricard.delgado@gmail.com)
 */
public class ToolTipsMessages {

	/** Useful messages and tips for the user. */
	final static String[] messages_ = { "Click here to get usage tips",
			"CTRL+C copies a snake into the clipboard",
			"CTRL+V pastes a snake from the clipboard",
			"Click on + to add a snake",
			"Click on > to optimize the active snake",
			"Click on >> top optimize all snakes",
			"Press delete to delete the active snake",
			"Press backspace to delete the active snake" };

	/** Random generator. */
	private static Random ran = new Random();

	// ============================================================================
	// PUBLIC METHODS

	/** Default constructor. */
	public ToolTipsMessages() {
	}

	// ----------------------------------------------------------------------------

	/** Returns the list of tips. */
	public static String[] getToolTipMessages() {
		return messages_;
	}

	// ----------------------------------------------------------------------------

	/** Returns a particular tip in the list. */
	public static String getToolTipMessage(int i) {
		if (i >= 0 && i < messages_.length) {
			return messages_[i];
		} else {
			return null;
		}
	}

	// ----------------------------------------------------------------------------

	/** Returns a random tip in the list. */
	public static String getToolTipMessage() {
		return messages_[ran.nextInt(messages_.length - 1) + 1];
	}
}
