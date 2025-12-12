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
package plugins.big.vascular.snake;

import icy.util.XMLUtil;

import org.w3c.dom.Element;
import org.w3c.dom.Node;

import plugins.big.vascular.core.Settings;
import plugins.big.vascular.snake.CylinderSnakeEnergyType;
import plugins.big.vascular.snake.CylinderSnakePriorShapeType;
import plugins.big.vascular.snake.CylinderSnakeTargetType;
import plugins.big.vascular.snake.ShapeSpaceType;

/**
 * Class that wraps the parameters of the E-Snake.
 * 
 * @version September 24, 2012
 * 
 * @author Ricard Delgado-Gonzalo (ricard.delgado@gmail.com)
 */
public class CylinderSnakeParameters {

	/**
	 * If <code>true</code> indicates that the snake will keep iterating till
	 * the optimizer decides so.
	 */
	private boolean immortal_ = true;

	/** Number of spline vector coefficients. */
	private int M_ = 0;

	/** Energy tradeoff factor. */
	private double alpha_ = 0;

	/** Surface stiffness. */
	private double gamma_ = 0;

	/** Maximum number of iterations allowed when the immortal is false. */
	private int maxLife_ = 0;

	/** Indicates the energy function of the snake. **/
	private CylinderSnakeEnergyType energyType_;

	/** Indicates the type of features to detect (bright or dark), **/
	private CylinderSnakeTargetType detectType_;
	/** Prior shape information. */
	private CylinderSnakePriorShapeType priorShapeType_ = Settings.PRIOR_SHAPE_DEFAULT;
	/** Shape space information. */
	private ShapeSpaceType shapeSpaceType_ = Settings.SHAPE_SPACE_DEFAULT;
	/** Prior-shape energy tradeoff factor. */
	private double beta_ = Settings.BETA_DEFAULT;

	/** Label of the XML tag containing the maximum number of iterations. */
	public static final String ID_MAX_LIFE = "maxLife";

	/**
	 * Label of the XML tag informing if the number of iterations of the snake
	 * is limited or not.
	 */
	public static final String ID_IMMORTAL = "immortal";

	/**
	 * Label of the XML tag containing the number of control points of the
	 * snake.
	 */
	public static final String ID_M = "M";

	/**
	 * Label of the XML tag containing the energy trade-off between the contour
	 * energy and the region energy.
	 */
	public static final String ID_ALPHA = "alpha";

	/**
	 * Label of the XML tag containing the surface stiffness.
	 */
	public static final String ID_GAMMA = "gamma";

	/**
	 * Label of the XML tag informing about the type of energy the snake uses
	 * (contour, region or mixture).
	 */
	public static final String ID_ENERGY_TYPE = "energyType";

	/**
	 * Label of the XML tag informing about the brightness of the target to
	 * segment (brighter or darker than the background).
	 */
	public static final String ID_DETECT_TYPE = "detectType";

	/**
	 * Label of the XML tag informing about the type of prior shape that is used
	 * in the energy.
	 */
	public static final String ID_PRIOR_SHAPE_TYPE = "priorShapeType";
	/**
	 * Label of the XML tag informing about the type of shape space that is used
	 * in the energy.
	 */
	public static final String ID_SHAPE_SPACE_TYPE = "priorShapeType";
	/**
	 * Label of the XML tag containing the energy weight of the prior shape
	 * energy.
	 */
	public static final String ID_BETA = "beta";

	// ============================================================================
	// PUBLIC METHODS

	/** Default constructor. */
	public CylinderSnakeParameters() {
	}

	/**
	 * Consructor
	 * 
	 * @param maxLife
	 * @param M
	 * @param alpha
	 * @param gamma
	 * @param immortal
	 * @param detectType
	 * @param energyType
	 */
	public CylinderSnakeParameters(int maxLife, int M, double alpha,
			double gamma, boolean immortal, CylinderSnakeTargetType detectType,
			CylinderSnakeEnergyType energyType) {
		setMaxLife(maxLife);
		setM(M);
		setAlpha(alpha);
		setGamma(gamma);
		setImmortal(immortal);
		setDetectType(detectType);
		setEnergyType(energyType);
	}

	// ----------------------------------------------------------------------------

	@Override
	public String toString() {
		return new String("[E-Snake parameters: immortal = " + immortal_
				+ ", M = " + M_ + ", alpha = " + alpha_ + ", gamma = " + gamma_
				+ ", maximum number of iterations = " + maxLife_
				+ ", energy type = " + energyType_ + ", detect type = "
				+ detectType_ + "]");
	}

	// ----------------------------------------------------------------------------

	public void saveToXML(Element node) {
		XMLUtil.setElementDoubleValue(node,
				CylinderSnakeParameters.ID_MAX_LIFE, maxLife_);
		XMLUtil.setElementIntValue(node, CylinderSnakeParameters.ID_M, M_);
		XMLUtil.setElementDoubleValue(node, CylinderSnakeParameters.ID_ALPHA,
				alpha_);
		XMLUtil.setElementDoubleValue(node, CylinderSnakeParameters.ID_GAMMA,
				gamma_);
		XMLUtil.setElementBooleanValue(node,
				CylinderSnakeParameters.ID_IMMORTAL, immortal_);
		switch (energyType_) {
		case CONTOUR:
			XMLUtil.setElementValue(node,
					CylinderSnakeParameters.ID_ENERGY_TYPE, "CONTOUR");
			break;
		case REGION:
			XMLUtil.setElementValue(node,
					CylinderSnakeParameters.ID_ENERGY_TYPE, "REGION");
			break;
		case MIXTURE:
			XMLUtil.setElementValue(node,
					CylinderSnakeParameters.ID_ENERGY_TYPE, "MIXTURE");
			break;
		}
		switch (detectType_) {
		case BRIGHT:
			XMLUtil.setElementValue(node,
					CylinderSnakeParameters.ID_DETECT_TYPE, "BRIGHT");
			break;
		case DARK:
			XMLUtil.setElementValue(node,
					CylinderSnakeParameters.ID_DETECT_TYPE, "DARK");
			break;
		}
	}

	public void loadFromToXML(Node node) {

		Element detectTypeAsStringElement = XMLUtil.getElement(node,
				ID_DETECT_TYPE);
		if (detectTypeAsStringElement != null) {
			String detectTypeAsString = detectTypeAsStringElement
					.getTextContent();
			if (detectTypeAsString.compareToIgnoreCase("BRIGHT") == 0) {
				setDetectType(CylinderSnakeTargetType.BRIGHT);
			} else if (detectTypeAsString.compareToIgnoreCase("DARK") == 0) {
				setDetectType(CylinderSnakeTargetType.DARK);
			} else {
				setDetectType(Settings.TARGET_TYPE_DEFAULT);
			}
		} else {
			setDetectType(Settings.TARGET_TYPE_DEFAULT);
		}

		Element energyTypeAsStringElement = XMLUtil.getElement(node,
				ID_ENERGY_TYPE);
		if (energyTypeAsStringElement != null) {
			String energyTypeAsString = energyTypeAsStringElement
					.getTextContent();
			if (energyTypeAsString.compareToIgnoreCase("CONTOUR") == 0) {
				setEnergyType(CylinderSnakeEnergyType.CONTOUR);
			} else if (energyTypeAsString.compareToIgnoreCase("REGION") == 0) {
				setEnergyType(CylinderSnakeEnergyType.REGION);
			} else if (energyTypeAsString.compareToIgnoreCase("MIXTURE") == 0) {
				setEnergyType(CylinderSnakeEnergyType.MIXTURE);
			} else {
				setEnergyType(Settings.ENERGY_TYPE_DEFAULT);
			}
		} else {
			setEnergyType(Settings.ENERGY_TYPE_DEFAULT);
		}

		Element priorShapeTypeAsStringElement = XMLUtil.getElement(node,
				ID_PRIOR_SHAPE_TYPE);
		if (priorShapeTypeAsStringElement != null) {
			String priorShapeTypeAsString = priorShapeTypeAsStringElement
					.getTextContent();
			if (priorShapeTypeAsString.compareToIgnoreCase("CUSTOM") == 0) {
				setPriorShapeType(CylinderSnakePriorShapeType.CUSTOM);
			} else {
				setPriorShapeType(Settings.PRIOR_SHAPE_DEFAULT);
			}
		} else {
			setPriorShapeType(Settings.PRIOR_SHAPE_DEFAULT);
		}

		Element shapeSpaceTypeAsStringElement = XMLUtil.getElement(node,
				ID_SHAPE_SPACE_TYPE);
		if (shapeSpaceTypeAsStringElement != null) {
			String shapeSpaceTypeAsString = shapeSpaceTypeAsStringElement
					.getTextContent();
			if (shapeSpaceTypeAsString.compareToIgnoreCase("SIMILARITY") == 0) {
				setShapeSpaceType(ShapeSpaceType.SIMILARITY);
			} else if (shapeSpaceTypeAsString.compareToIgnoreCase("AFFINE") == 0) {
				setShapeSpaceType(ShapeSpaceType.AFFINE);
			}
		} else {
			setShapeSpaceType(Settings.SHAPE_SPACE_DEFAULT);
		}

		setMaxLife(XMLUtil.getElementIntValue(node, ID_MAX_LIFE,
				Settings.MAX_LIFE_DEFAULT));
		setM(XMLUtil.getElementIntValue(node, ID_M, Settings.M_DEFAULT));
		setAlpha(XMLUtil.getElementDoubleValue(node, ID_ALPHA,
				Settings.ALPHA_DEFAULT));
		setBeta(XMLUtil.getElementDoubleValue(node, ID_BETA,
				Settings.BETA_DEFAULT));
		setGamma(XMLUtil.getElementDoubleValue(node, ID_GAMMA, Settings.GAMMA_DEFAULT));
		setImmortal(XMLUtil.getElementBooleanValue(node, ID_IMMORTAL,
				Settings.IMMORTAL_DEFAULT));
	}

	// ============================================================================
	// GETTERS AND SETTERS

	public double getAlpha() {
		return alpha_;
	}

	// ----------------------------------------------------------------------------

	public void setAlpha(double alpha) {
		alpha_ = alpha;
	}

	// ----------------------------------------------------------------------------

	public double getBeta() {
		return beta_;
	}

	// ----------------------------------------------------------------------------

	public void setBeta(double beta) {
		beta_ = beta;
	}

	// ----------------------------------------------------------------------------

	public double getGamma() {
		return gamma_;
	}

	// ----------------------------------------------------------------------------

	public void setGamma(double gamma) {
		gamma_ = gamma;
	}

	// ----------------------------------------------------------------------------

	public boolean isImmortal() {
		return immortal_;
	}

	// ----------------------------------------------------------------------------

	public void setImmortal(boolean immortal) {
		immortal_ = immortal;
	}

	// ----------------------------------------------------------------------------

	public int getM() {
		return M_;
	}

	// ----------------------------------------------------------------------------

	public void setM(int M) {
		M_ = M;
	}

	// ----------------------------------------------------------------------------

	public CylinderSnakeTargetType getDetectType() {
		return detectType_;
	}

	// ----------------------------------------------------------------------------

	public void setDetectType(CylinderSnakeTargetType detectType) {
		detectType_ = detectType;
	}

	// ----------------------------------------------------------------------------

	public CylinderSnakeEnergyType getEnergyType() {
		return energyType_;
	}

	// ----------------------------------------------------------------------------

	public void setEnergyType(CylinderSnakeEnergyType energyType) {
		energyType_ = energyType;
	}

	// ----------------------------------------------------------------------------

	public int getMaxLife() {
		return maxLife_;
	}

	// ----------------------------------------------------------------------------

	public void setMaxLife(int maxLife) {
		maxLife_ = maxLife;
	}

	// ----------------------------------------------------------------------------

	public void setPriorShapeType(CylinderSnakePriorShapeType priorShapeType) {
		priorShapeType_ = priorShapeType;
	}

	// ----------------------------------------------------------------------------

	public void setShapeSpaceType(ShapeSpaceType shapeSpaceType) {
		shapeSpaceType_ = shapeSpaceType;
	}
}
