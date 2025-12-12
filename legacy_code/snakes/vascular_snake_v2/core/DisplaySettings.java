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
package plugins.big.vascular.core;

/**
 * Class that encapsulates the display options.
 * 
 * @version May 6, 2013
 * 
 * @author Ricard Delgado-Gonzalo (ricard.delgado@gmail.com)
 */
public class DisplaySettings {

	private final boolean refreshDrag_ ;
	private final boolean refresh_;
	private final int strokeThickness_;
	private final int depthTransparency_;
	private final Settings.MeshResolution meshResolution_;

	// ============================================================================
	// PUBLIC METHODS

	/** Constructor. */
	public DisplaySettings(boolean refreshDrag, boolean refresh,
			int strokeThickness, int depthTransparency,
			Settings.MeshResolution meshResolution) {
		refreshDrag_ = refreshDrag;
		refresh_ = refresh;
		strokeThickness_ = strokeThickness;
		depthTransparency_ = depthTransparency;
		meshResolution_ = meshResolution;
	}

	// ----------------------------------------------------------------------------

	public boolean refreshDrag() {
		return refreshDrag_;
	}

	// ----------------------------------------------------------------------------

	public boolean refresh() {
		return refresh_;
	}

	// ----------------------------------------------------------------------------

	public int getDepthTransparency() {
		return depthTransparency_;
	}

	// ----------------------------------------------------------------------------

	public Settings.MeshResolution getMeshResolution() {
		return meshResolution_;
	}

	// ----------------------------------------------------------------------------

	public int getStrokeThickness() {
		return strokeThickness_;
	}

	// ----------------------------------------------------------------------------

	@Override
	public String toString() {
		return new String(("[meshResolution_ = " + meshResolution_ + "; "
				+ "strokeThickness_ = " + strokeThickness_ + "; "
				+ "depthTransparency_ = " + depthTransparency_ + "; "
				+ "refresh_ = " + refresh_ + "refreshDrag_ = " + refreshDrag_ + " ]"));
	}
}
