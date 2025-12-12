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
 *     Daniel Schmitter (daniel.schmitter@epfl.ch)
 ******************************************************************************/
package plugins.big.bigsnake3d.core;

import icy.gui.frame.IcyFrame;

import java.awt.Image;
import java.util.ArrayList;
import java.util.List;

import javax.swing.ImageIcon;
import javax.swing.JDialog;
import javax.swing.JFrame;
import javax.swing.JInternalFrame;

import plugins.big.bigsnake3d.rsc.ResourceUtilMSBrain;
import plugins.big.bigsnake3d.rsc.icon.SnakeIconsMSBrain;
import plugins.big.bigsnake3d.snake.*;

/**
 * Offers global parameters (settings) and functions used by the plug-in.
 * 
 * Settings makes use of the Singleton design pattern: There's at most one
 * instance present, which can only be accessed through getInstance().
 * 
 * @version November 19, 2014
 * 
 * @author Ricard Delgado-Gonzalo (ricard.delgado@gmail.com)
 * @author Daniel Schmitter (daniel.schmitter@epfl.ch)
 */
public class SettingsMSBrain {

	/** Set to true to enable DEBUG mode */
	public static final boolean DEBUG = false;

	/** The unique instance of Settings (Singleton design pattern). */
	private static SettingsMSBrain instance_ = null;

	// ----------------------------------------------------------------------------
	// PLUG-IN PARAMETERS

	/** Name of the application */
	private final String appName_ = "MSpline Sphere Snake";
	/** Version of the application */
	private final String appVersion_ = "1.0";
	/** Minimal version of Icy required to run the plug-in. */
	private final String icyRequiredVersion_ = "1.5.3.1";

	// ----------------------------------------------------------------------------
	// DISPAY PARAMETERS

	public static final MeshResolution MESH_RESOLUTION_DEFAULT = MeshResolution.LOW;
	public static final int STROKE_THICKNESS_DEFAULT = 1;
	public static final int DEPTH_TRANSPARENCY_DEFAULT = 5;
	public static final boolean REFRESH_SCREEN_DEFAULT = true;
	
	// ----------------------------------------------------------------------------
		// DEFAULT PLUG-IN PARAMETERS

		/** Default value of the type of features to detect (bright or dark). */
		public static final SphereSnakeTargetTypeMSBrain TARGET_TYPE_DEFAULT = SphereSnakeTargetTypeMSBrain.DARK;
		/** Default value of the energy function of the snake. */
		public static final SphereSnakeEnergyTypeMSBrain ENERGY_TYPE_DEFAULT = SphereSnakeEnergyTypeMSBrain.REGION;
		/** Default value of the prior-shape information. */
		public static final SphereSnakePriorShapeTypeBrain PRIOR_SHAPE_DEFAULT = SphereSnakePriorShapeTypeBrain.NONE;
		/** Default value of shape space information. */
		public static final ShapeSpaceTypeBrain SHAPE_SPACE_DEFAULT = ShapeSpaceTypeBrain.SIMILARITY;
		/** Default prior-shape energy tradeoff factor. */
		public static final double BETA_DEFAULT = 0;
		/**
		 * Default maximum number of iterations allowed when the snake is not
		 * immortal.
		 */
		public static final int MAX_LIFE_DEFAULT = 500;
		/** Default number of spline vector coefficients. */
		public static final int M_DEFAULT = 4;
		/** Default energy tradeoff factor. */
		public static final double ALPHA_DEFAULT = 0;

		/**
		 * Default value of the immortal flag. If <code>true</code> indicates that
		 * the snake will keep iterating till the optimizer decides so.
		 */
		public static final boolean IMMORTAL_DEFAULT = false;


	// ----------------------------------------------------------------------------
	// ENUMS

	public enum MeshResolution {
		HIGH, NORMAL, LOW
	};

	// ============================================================================
	// PUBLIC METHODS

	/** Get Settings instance */
	static public SettingsMSBrain getInstance() {
		if (instance_ == null) {
			instance_ = new SettingsMSBrain();
		}
		return instance_;
	}

	// ----------------------------------------------------------------------------

	/** Set the window icon. */
	public void setWindowIcon(JDialog dialog) {
		ResourceUtilMSBrain resourceUtil = ResourceUtilMSBrain.getInstance();
		try {
			List<Image> icons = new ArrayList<Image>();
			icons.add(resourceUtil.getIcon(SnakeIconsMSBrain.OUROBOROS16));
			icons.add(resourceUtil.getIcon(SnakeIconsMSBrain.OUROBOROS24));
			icons.add(resourceUtil.getIcon(SnakeIconsMSBrain.OUROBOROS32));
			icons.add(resourceUtil.getIcon(SnakeIconsMSBrain.OUROBOROS48));
			icons.add(resourceUtil.getIcon(SnakeIconsMSBrain.OUROBOROS64));
			icons.add(resourceUtil.getIcon(SnakeIconsMSBrain.OUROBOROS128));
			icons.add(resourceUtil.getIcon(SnakeIconsMSBrain.OUROBOROS256));
			dialog.setIconImages(icons);
		} catch (Exception e) {
			System.err.println("Failed to set the window icon.");
			e.printStackTrace();
		}
	}

	// ----------------------------------------------------------------------------

	/** Set the window icon. */
	public void setWindowIcon(IcyFrame icyFrame) {
		ResourceUtilMSBrain resourceUtil = ResourceUtilMSBrain.getInstance();
		try {
			List<Image> icons = new ArrayList<Image>();
			icons.add(resourceUtil.getIcon(SnakeIconsMSBrain.OUROBOROS16));
			icons.add(resourceUtil.getIcon(SnakeIconsMSBrain.OUROBOROS24));
			icons.add(resourceUtil.getIcon(SnakeIconsMSBrain.OUROBOROS32));
			icons.add(resourceUtil.getIcon(SnakeIconsMSBrain.OUROBOROS48));
			icons.add(resourceUtil.getIcon(SnakeIconsMSBrain.OUROBOROS64));
			icons.add(resourceUtil.getIcon(SnakeIconsMSBrain.OUROBOROS128));
			icons.add(resourceUtil.getIcon(SnakeIconsMSBrain.OUROBOROS256));
			JFrame externalFrame = icyFrame.getExternalFrame();
			externalFrame.setIconImages(icons);
			JInternalFrame internalFrame = icyFrame.getInternalFrame();
			internalFrame.setFrameIcon(new ImageIcon(resourceUtil
					.getIcon(SnakeIconsMSBrain.OUROBOROS16)));
		} catch (Exception e) {
			System.err.println("Failed to set the window icon.");
			e.printStackTrace();
		}
	}

	// ============================================================================
	// SETTERS AND GETTERS

	/** Returns the name of the plug-in. */
	public String getAppName() {
		return appName_;
	}

	// ----------------------------------------------------------------------------

	/** Returns the version of the plug-in. */
	public String getAppVersion() {
		return appVersion_;
	}

	// ----------------------------------------------------------------------------

	/** Returns the minimum version of ICY in which the plug-in works properly. */
	public String getIcyRequiredVersion() {
		return icyRequiredVersion_;
	}
}
