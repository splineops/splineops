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
package plugins.big.bigsnake3d.keeper;

import java.util.LinkedList;

import plugins.big.bigsnake3d.core.DisplaySettingsMSBrain;
import plugins.big.bigsnake3d.roi.ActionPlaneMSBrain;
import plugins.big.bigsnake3d.roi.SnakeEditModeMSBrain;

/**
 * Class that encapsulates a list of SnakeKeepers.
 * 
 * @version May 3, 2014
 * 
 * @author Ricard Delgado-Gonzalo (ricard.delgado@gmail.com)
 */
public class KeepersListMSBrain {

	/** List of all <code>SnakeKeeper</code>. */
	private LinkedList<SnakeKeeperMSBrain> snakeKeepers_ = null;

	/** <code>SnakeKeeper</code> associated to the active snake. */
	private SnakeKeeperMSBrain activeSnakeKeeper_ = null;

	// ============================================================================
	// PUBLIC METHODS

	/** Default constructor. */
	public KeepersListMSBrain() {
		snakeKeepers_ = new LinkedList<SnakeKeeperMSBrain>();
		activeSnakeKeeper_ = null;
	}

	// ----------------------------------------------------------------------------

	/** Returns the active <code>SnakeKeeper</code>. */
	public SnakeKeeperMSBrain getActiveSnakeKeeper() {
		return activeSnakeKeeper_;
	}

	// ----------------------------------------------------------------------------

	/**
	 * Adds a <code>SnakeKeeper</code> at the end of the list, and sets it to
	 * selected.
	 */
	public void addAndActivateKeeper(SnakeKeeperMSBrain keeper) {
		if (keeper != null) {
			snakeKeepers_.add(keeper);
			activateSnakeKeeper(keeper);
		}
	}

	// ----------------------------------------------------------------------------

	/**
	 * Sends a message to the active snake that it should be converted to a
	 * binary mask.
	 */
	public void rasterizeActiveSnake() {
		if (activeSnakeKeeper_ != null) {
			activeSnakeKeeper_.rasterizeSnake();
		}
	}

	// ----------------------------------------------------------------------------

	/**
	 * Removes the active <code>SnakeKeeper</code> from the list and sets to
	 * active the next one in the list. If the removed keeper is the last one in
	 * the list,
	 */
	public void removeActiveSnakeKeeper() {
		if (activeSnakeKeeper_ != null) {
			activeSnakeKeeper_.removeFromSequence();
			int index = snakeKeepers_.indexOf(activeSnakeKeeper_);
			snakeKeepers_.remove(index);
			if (!snakeKeepers_.isEmpty()) {
				activateSnakeKeeper(snakeKeepers_.get(Math.min(index,
						snakeKeepers_.size() - 1)));
			} else {
				activeSnakeKeeper_ = null;
			}
		}
	}

	// ----------------------------------------------------------------------------

	/**
	 * Removes all of the elements from the list, and removes the
	 * <code>SnakeKeeper</code> from the associated images.
	 */
	public void removeAllSnakeKeepers() {
		for (SnakeKeeperMSBrain keeper : snakeKeepers_) {
			if (keeper != null) {
				keeper.removeFromSequence();
			}
		}
		snakeKeepers_.clear();
		activeSnakeKeeper_ = null;
	}

	// ----------------------------------------------------------------------------

	/**
	 * If the <code>SnakeKeeper</code> passed as a parameter is in the list, it
	 * becomes activated and the method returns <code>true</code>. If the
	 * element is not in the list, it returns <code>false</code>.
	 */
	public boolean activateSnakeKeeper(SnakeKeeperMSBrain keeper) {
		if (snakeKeepers_.contains(keeper)) {
			activeSnakeKeeper_ = keeper;
			for (SnakeKeeperMSBrain k : snakeKeepers_) {
				k.setSelected(k == activeSnakeKeeper_);
			}
			return true;
		} else {
			return false;
		}
	}

	// ----------------------------------------------------------------------------

	/**
	 * Activates the i-th <code>SnakeKeeper</code> form the internal list. If
	 * the list have less than i-th elements, the method returns
	 * <code>false</code>, otherwise it returns <code>true</code>.
	 */
	public boolean activateSnakeKeeper(int i) {
		if (i >= 0 && i < snakeKeepers_.size()) {
			return activateSnakeKeeper(snakeKeepers_.get(i));
		}
		return false;
	}

	// ----------------------------------------------------------------------------

	/**
	 * Returns <code>true</code> if the <code>SnakeKeeper</code> passed as a
	 * parameters if the active one.
	 */
	public boolean isActiveSnakeKeeper(SnakeKeeperMSBrain keeper) {
		return activeSnakeKeeper_ == keeper;
	}

	// ----------------------------------------------------------------------------

	/**
	 * Sets the <code>ActionPlane</code> to all elements of
	 * <code>SnakeKeeper</code> contained in the list.
	 */
	public void setActionPlane(ActionPlaneMSBrain actionPlane) {
		for (SnakeKeeperMSBrain keeper : snakeKeepers_) {
			if (keeper != null) {
				keeper.setActionPlane(actionPlane);
			}
		}
	}

	// ----------------------------------------------------------------------------

	/**
	 * Sets the <code>DisplaySettings</code> to all elements of
	 * <code>SnakeKeeper</code> contained in the list.
	 */
	public void setDisplaySettings(DisplaySettingsMSBrain displaySettings) {
		for (SnakeKeeperMSBrain keeper : snakeKeepers_) {
			if (keeper != null) {
				keeper.setDisplaySettings(displaySettings);
			}
		}
	}

	// ----------------------------------------------------------------------------

	/**
	 * Sets the <code>SnakeEditMode</code> to all elements of
	 * <code>SnakeKeeper</code> contained in the list.
	 */
	public void setSnakeEditMode(SnakeEditModeMSBrain editingMode) {
		for (SnakeKeeperMSBrain keeper : snakeKeepers_) {
			if (keeper != null) {
				keeper.setSnakeEditMode(editingMode);
			}
		}
	}

	// ----------------------------------------------------------------------------

	/** Returns the number of <code>SnakeKeeper</code> in the internal list. */
	public int getNumKeepers() {
		return snakeKeepers_.size();
	}

	// ----------------------------------------------------------------------------

	/** Returns <code>true</code> is the list contains no elements. */
	public boolean isEmpty() {
		return snakeKeepers_.isEmpty();
	}
}
