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
package plugins.big.vascular.core;

import icy.gui.frame.IcyFrame;

import java.awt.Image;
import java.util.ArrayList;
import java.util.List;

import javax.swing.ImageIcon;
import javax.swing.JDialog;
import javax.swing.JFrame;
import javax.swing.JInternalFrame;

import plugins.big.vascular.rsc.ResourceUtil;
import plugins.big.vascular.rsc.icon.SnakeIcons;
import plugins.big.vascular.snake.CylinderSnakeEnergyType;
import plugins.big.vascular.snake.CylinderSnakePriorShapeType;
import plugins.big.vascular.snake.CylinderSnakeTargetType;
import plugins.big.vascular.snake.ShapeSpaceType;

/**
 * Offers global parameters (settings) and functions used by the plug-in.
 * 
 * Settings makes use of the Singleton design pattern: There's at most one
 * instance present, which can only be accessed through getInstance().
 * 
 * @version April 29, 2013
 * 
 * @author Ricard Delgado-Gonzalo (ricard.delgado@gmail.com)
 */
public class Settings {

	/** Set to true to enable DEBUG mode */
	public static final boolean DEBUG = false;

	/** The unique instance of Settings (Singleton design pattern). */
	private static Settings instance_ = null;

	// ----------------------------------------------------------------------------
	// PLUG-IN PARAMETERS

	/** Name of the application */
	private final String appName_ = "Cylinder Snake 3D";
	/** Version of the application */
	private final String appVersion_ = "1.0";
	/** Minimal version of Icy required to run the plug-in. */
	private final String icyRequiredVersion_ = "1.3.6.0";

	// ----------------------------------------------------------------------------
	// DISPAY PARAMETERS

	public static final MeshResolution MESH_RESOLUTION_DEFAULT = MeshResolution.LOW;
	public static final int STROKE_THICKNESS_DEFAULT = 1;
	public static final int DEPTH_TRANSPARENCY_DEFAULT = 5;
	public static final boolean REFRESH_SCREEN_DEFAULT = true;
	public static final boolean REFRESH_SNAKE_DRAG_DEFAULT = false;

	// ----------------------------------------------------------------------------
	// DEFAULT PLUG-IN PARAMETERS

	/** Default value of the type of features to detect (bright or dark). */
	public static final CylinderSnakeTargetType TARGET_TYPE_DEFAULT = CylinderSnakeTargetType.DARK;
	/** Default value of the energy function of the snake. */
	public static final CylinderSnakeEnergyType ENERGY_TYPE_DEFAULT = CylinderSnakeEnergyType.REGION;
	/** Default value of the prior-shape information. */
	public static final CylinderSnakePriorShapeType PRIOR_SHAPE_DEFAULT = CylinderSnakePriorShapeType.NONE;
	/** Default value of shape space information. */
	public static final ShapeSpaceType SHAPE_SPACE_DEFAULT = ShapeSpaceType.SIMILARITY;
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
	public static final double GAMMA_DEFAULT = 0;

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
	static public Settings getInstance() {
		if (instance_ == null) {
			instance_ = new Settings();
		}
		return instance_;
	}

	// ----------------------------------------------------------------------------

	/** Set the window icon. */
	public void setWindowIcon(JDialog dialog) {
		ResourceUtil resourceUtil = ResourceUtil.getInstance();
		try {
			List<Image> icons = new ArrayList<Image>();
			icons.add(resourceUtil.getIcon(SnakeIcons.OUROBOROS16));
			icons.add(resourceUtil.getIcon(SnakeIcons.OUROBOROS24));
			icons.add(resourceUtil.getIcon(SnakeIcons.OUROBOROS32));
			icons.add(resourceUtil.getIcon(SnakeIcons.OUROBOROS48));
			icons.add(resourceUtil.getIcon(SnakeIcons.OUROBOROS64));
			icons.add(resourceUtil.getIcon(SnakeIcons.OUROBOROS128));
			icons.add(resourceUtil.getIcon(SnakeIcons.OUROBOROS256));
			dialog.setIconImages(icons);
		} catch (Exception e) {
			System.err.println("Failed to set the window icon.");
			e.printStackTrace();
		}
	}

	// ----------------------------------------------------------------------------

	/** Set the window icon. */
	public void setWindowIcon(IcyFrame icyFrame) {
		ResourceUtil resourceUtil = ResourceUtil.getInstance();
		try {
			List<Image> icons = new ArrayList<Image>();
			icons.add(resourceUtil.getIcon(SnakeIcons.OUROBOROS16));
			icons.add(resourceUtil.getIcon(SnakeIcons.OUROBOROS24));
			icons.add(resourceUtil.getIcon(SnakeIcons.OUROBOROS32));
			icons.add(resourceUtil.getIcon(SnakeIcons.OUROBOROS48));
			icons.add(resourceUtil.getIcon(SnakeIcons.OUROBOROS64));
			icons.add(resourceUtil.getIcon(SnakeIcons.OUROBOROS128));
			icons.add(resourceUtil.getIcon(SnakeIcons.OUROBOROS256));
			JFrame externalFrame = icyFrame.getExternalFrame();
			externalFrame.setIconImages(icons);
			JInternalFrame internalFrame = icyFrame.getInternalFrame();
			internalFrame.setFrameIcon(new ImageIcon(resourceUtil
					.getIcon(SnakeIcons.OUROBOROS16)));
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
