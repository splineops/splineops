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
package plugins.big.bigsnake3d.core;

import plugins.big.bigsnake3d.snake.SphereSnakeParametersMSBrain;
import plugins.big.bigsnakeutils.icy.snake3D.Snake3DNode;

/**
 * Container where the defining nodes and parameters of a snake can be stored.
 * 
 * @version May 3, 2014
 * 
 * @author Emrah Bostan (emrah.bostan@gmail.com)
 * @author Ricard Delgado-Gonzalo (ricard.delgado@gmail.com)
 */
public class Snake3DClipboardMSBrain {

	/** Snake execution parameters. */
	private SphereSnakeParametersMSBrain parameters_ = null;
	/** Snake-defining control points. */
	private Snake3DNode[] nodes_ = null;

	// ============================================================================
	// PUBLIC METHODS

	public Snake3DClipboardMSBrain() {
	}

	// ----------------------------------------------------------------------------

	public Snake3DNode[] getSnakeNodes() {
		return nodes_;
	}

	// ----------------------------------------------------------------------------

	public SphereSnakeParametersMSBrain getSnakeParameters() {
		return parameters_;
	}

	// ----------------------------------------------------------------------------

	public void setSnakeNodes(Snake3DNode[] nodes) {
		nodes_ = nodes;
	}

	// ----------------------------------------------------------------------------

	public void setSnakeParameters(SphereSnakeParametersMSBrain parameters) {
		parameters_ = parameters;
	}

	// ----------------------------------------------------------------------------

	/**
	 * Returns <code>true</code> if there are no snake nodes or snake parameters
	 * in the clipboard. Otherwise, it returns <code>false</code>
	 */
	public boolean isEmpty() {
		if (nodes_ == null || parameters_ == null) {
			return true;
		} else {
			return false;
		}
	}
}
