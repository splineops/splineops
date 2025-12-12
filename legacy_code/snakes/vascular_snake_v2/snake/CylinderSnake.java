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

import icy.gui.frame.progress.AnnounceFrame;
import icy.gui.frame.progress.ProgressFrame;
import icy.image.IcyBufferedImage;
import icy.sequence.Sequence;
import icy.type.DataType;
import icy.type.rectangle.Rectangle3D;
import icy.util.XMLUtil;

import java.awt.Color;
import java.awt.Polygon;
import java.awt.geom.Point2D;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import javax.vecmath.Point3d;

import org.w3c.dom.Element;

import plugins.big.bigsnakeutils.icy.snake2D.Snake2DNode;
import plugins.big.bigsnakeutils.icy.snake3D.Snake3D;
import plugins.big.bigsnakeutils.icy.snake3D.Snake3DNode;
import plugins.big.bigsnakeutils.icy.snake3D.Snake3DScale;
import plugins.big.bigsnakeutils.process.process1D.BSplineBasis;
import plugins.big.bigsnakeutils.process.process1D.BSplineBasis.BSplineBasisType;
import plugins.big.bigsnakeutils.process.process3D.SequenceToArrayConverter;
import plugins.big.bigsnakeutils.shape.utils.Binarizer3D;
import plugins.big.bigsnakeutils.shape.utils.ContourTracing;
import plugins.big.bigsnakeutils.shape.utils.Geometry2D;
import plugins.big.vascular.core.ImageLUTContainer;
import vtk.vtkPoints;
import vtk.vtkPolyData;
import vtk.vtkQuad;
import Jama.Matrix;

/**
 * Three-dimensional exponential spline snake (E-Snake).
 * 
 * @version October 30, 2013
 * 
 * @author Ricard Delgado-Gonzalo (ricard.delgado@gmail.com)
 * @author Nicolas Chenouard (nicolas.chenouard@gmail.com)
 * @author Christophe Gaudet-Blavignac (chrisgaubla@gmail.com)
 */
public class CylinderSnake implements Snake3D {

	/** Snake defining nodes. */
	private Snake3DNode[] coef_ = null;

	// ----------------------------------------------------------------------------
	// SNAKE CONTOUR

	/** Samples of the x coordinates of the snake contour. */
	private double[][] xSnakeContour_ = null;
	/** Samples of the y coordinates of the snake contour. */
	private double[][] ySnakeContour_ = null;
	/** Samples of the z coordinates of the snake contour. */
	private double[][] zSnakeContour_ = null;

	// /** Bounding box that encloses the outline of the snake. */
	// private Rectangle3D snakeSurfaceBoundingBox_ = null;

	/** LUT with the samples of the x coordinates of the snake contour. */
	private double[][] xSnakeShellContour_ = null;
	/** LUT with the samples of the y coordinates of the snake contour. */
	private double[][] ySnakeShellContour_ = null;
	/** LUT with the samples of the z coordinates of the snake contour. */
	private double[][] zSnakeShellContour_ = null;

	/** Number of samples at which each segment of the contour is discretized. */
	private int nSamplesPerSegment_ = 39;

	// ----------------------------------------------------------------------------
	// SNAKE TANGENT BUNDLE

	/**
	 * LUT with the samples of the first component of the first vector of the
	 * tangent bundle.
	 */
	private double[][] dxdu_ = null;
	/**
	 * LUT with the samples of the second component of the first vector of the
	 * tangent bundle.
	 */
	private double[][] dydu_ = null;
	/**
	 * LUT with the samples of the third component of the first vector of the
	 * tangent bundle.
	 */
	private double[][] dzdu_ = null;
	/**
	 * LUT with the samples of the first component of the second vector of the
	 * tangent bundle.
	 */
	private double[][] dxdv_ = null;
	/**
	 * LUT with the samples of the second component of the second vector of the
	 * tangent bundle.
	 */
	private double[][] dydv_ = null;
	/**
	 * LUT with the samples of the third component of the second vector of the
	 * tangent bundle.
	 */
	private double[][] dzdv_ = null;

	/**
	 * LUT with the samples of the second derivative of the first component of
	 * the parameterization. $\frac{\partial\sigma(u,v)}{\partial u \partial u}$
	 */
	private double[][] ddxdudu_ = null;
	/**
	 * LUT with the samples of the second derivative of the first component of
	 * the parameterization. $\frac{\partial\sigma(u,v)}{\partial u \partial v}$
	 */
	private double[][] ddxdudv_ = null;
	/**
	 * LUT with the samples of the second derivative of the first component of
	 * the parameterization. $\frac{\partial\sigma(u,v)}{\partial v \partial v}$
	 */
	private double[][] ddxdvdv_ = null;
	/**
	 * LUT with the samples of the second derivative of the second component of
	 * the parameterization. $\frac{\partial\sigma(u,v)}{\partial u \partial u}$
	 */
	private double[][] ddydudu_ = null;
	/**
	 * LUT with the samples of the second derivative of the second component of
	 * the parameterization. $\frac{\partial\sigma(u,v)}{\partial u \partial v}$
	 */
	private double[][] ddydudv_ = null;
	/**
	 * LUT with the samples of the second derivative of the second component of
	 * the parameterization. $\frac{\partial\sigma(u,v)}{\partial v \partial v}$
	 */
	private double[][] ddydvdv_ = null;
	/**
	 * LUT with the samples of the second derivative of the third component of
	 * the parameterization. $\frac{\partial\sigma(u,v)}{\partial u \partial u}$
	 */
	private double[][] ddzdudu_ = null;
	/**
	 * LUT with the samples of the second derivative of the third component of
	 * the parameterization. $\frac{\partial\sigma(u,v)}{\partial u \partial v}$
	 */
	private double[][] ddzdudv_ = null;
	/**
	 * LUT with the samples of the second derivative of the third component of
	 * the parameterization. $\frac{\partial\sigma(u,v)}{\partial v \partial v}$
	 */
	private double[][] ddzdvdv_ = null;

	/** LUT with the samples of the first component of the normal vector. */
	private double[][] dyWedgedz_ = null;
	/** LUT with the samples of the second component of the normal vector. */
	private double[][] dzWedgedx_ = null;
	/** LUT with the samples of the third component of the normal vector. */
	private double[][] dxWedgedy_ = null;

	// ----------------------------------------------------------------------------
	// SNAKE STATUS FIELDS

	/**
	 * If <code>true</code> indicates that the snake is able to keep being
	 * optimized.
	 */
	private boolean alive_ = true;
	/** Area of the snake surface. */
	private double area_ = 0;
	/**
	 * If <code>true</code>, the snake has gone through the initialization
	 * process.
	 */
	private boolean isInitialized_ = false;
	/**
	 * Number of iterations left when <code>immortal_</code> is
	 * <code>false</code>.
	 */
	private int life_ = 0;
	/**
	 * If <code>false</code> indicates that the snake takes an ill conditioned
	 * shape.
	 */
	private final boolean valid_ = true;
	/** Volume enclosed by the snake. */
	private double volume_ = 0;

	// ----------------------------------------------------------------------------
	// SNAKE OPTION FIELDS

	/** Parameter that determines the number of degrees of freedom of the snake. */
	private int M_ = 0;
	/** Energy trade-off factor. */
	private double alpha_ = 0;
	/** Weight factor for the reparameterization energy. */
	private double gamma_ = 0;
	/**
	 * Maximum number of iterations allowed when the <code>immortal_</code> is
	 * <code>false</code>.
	 */
	private int maxLife_ = 0;
	/** Snake energy used during the optimization process. */
	private CylinderSnakeEnergyType energyType_ = CylinderSnakeEnergyType.REGION;
	/** Indicates the type of features to detect (bright or dark). **/
	private CylinderSnakeTargetType detectType_ = CylinderSnakeTargetType.BRIGHT;
	/**
	 * If <code>true</code> indicates that the snake will keep iterating till
	 * the optimizer decides so.
	 */
	private boolean immortal_ = true;

	// ----------------------------------------------------------------------------
	// SPLINE LUTS

	/** Basis function of the surface. */
	private static final BSplineBasis.BSplineBasisType BASIS_FUNCTION = BSplineBasisType.MSPLINE;
	/**
	 * LUT with the samples of the B-spline basis function along the first
	 * dimension in the parametric space.
	 */
	private double[] bSpline1LUT_ = null;
	/**
	 * LUT with the samples of the derivative of the basis function along the
	 * first dimension in the parametric space.
	 */
	private double[] bSpline1DerivativeLUT_ = null;
	/**
	 * LUT with the samples of the second derivative of the basis function along
	 * the first dimension in the parametric space.
	 */
	private double[] bSpline1SecondDerivativeLUT_ = null;

	/**
	 * LUT with the samples of the B-spline basis function along the second
	 * dimension in the parametric space.
	 */
	private double[] bSpline2LUT_ = null;
	/**
	 * LUT with the samples of the derivative of the basis function along the
	 * second dimension in the parametric space.
	 */
	private double[] bSpline2DerivativeLUT_ = null;
	/**
	 * LUT with the samples of the second derivative of the basis function along
	 * the second dimension in the parametric space.
	 */
	private double[] bSpline2SecondDerivativeLUT_ = null;

	// ----------------------------------------------------------------------------
	// IMAGE FIELDS

	/** Container of the LUTs. */
	private ImageLUTContainer imageLUTs_ = null;

	// ----------------------------------------------------------------------------
	// AUXILIARY FIELDS AND LUTS

	/** PI/M. */
	private double PIM_ = 0;
	/** 2*PI/M. */
	private double PI2M_ = 0;
	/**
	 * Support of the basis function multiplied by twice
	 * <code>nSamplesPerSegment_</code>.
	 */
	private int NR2_ = 0;
	/** Number of control points multiplied by <code>nSamplesPerSegment_</code>. */
	private int MR_ = 0;
	/**
	 * Number of control points multiplied by twice
	 * <code>nSamplesPerSegment_</code>.
	 */
	private int MR2_ = 0;

	/** LUT with samples of one period of a sine. */
	private double[] sinLUT_ = null;
	/** LUT with samples of one period of a cosine. */
	private double[] cosLUT_ = null;

	// ----------------------------------------------------------------------------
	// CONSTANTS

	/** Normalization factor for the Jacobian (magic number). */
	private static double JACOBIANCONSTANT = 10000;
	/** Cube root of two. */
	private static double CBRT = Math.cbrt(2.0);
	/** Square root of the <code>float</code> machine precision. */
	private static final double SQRT_TINY = Math.sqrt(Float
			.intBitsToFloat(0x33FFFFFF));
	// ----------------------------------------------------------------------------
	// XML TAGS

	/**
	 * Label of the XML tag containing the list of snake-defining control
	 * points.
	 */
	public static final String ID_CONTROL_POINTS = "control_points";
	/**
	 * Label of the XML tag containing the a single snake-defining control
	 * point.
	 */
	public static final String ID_CONTROL_POINT = "control_point";


	// ============================================================================
	// PUBLIC METHODS

	/** Default constructor. */
	public CylinderSnake(ImageLUTContainer imageLUTs,
			CylinderSnakeParameters parameters) {
		if (imageLUTs == null) {
			System.err.println("Image not properly loaded.");
			return;
		}
		imageLUTs_ = imageLUTs;
		if (parameters == null) {
			System.err.println("Snake parameters not properly loaded.");
			return;
		}
		M_ = parameters.getM();
		if (M_ < 3) {
			throw new IllegalArgumentException(
					"Error: M needs to be equal or larger than 3.");
		}
		nSamplesPerSegment_ = (int) Math.round((double) nSamplesPerSegment_
				/ (double) M_);

		maxLife_ = parameters.getMaxLife();
		alpha_ = parameters.getAlpha();
		gamma_ = parameters.getGamma();
		immortal_ = parameters.isImmortal();
		energyType_ = parameters.getEnergyType();
		detectType_ = parameters.getDetectType();

		initialize();
	}

	// ----------------------------------------------------------------------------

	/**
	 * The purpose of this method is to compute the energy of the snake. This
	 * energy is usually made of three additive terms: 1) the image energy,
	 * which gives the driving force associated to the data; 2) the internal
	 * energy, which favors smoothness of the snake; and 3) the constraint
	 * energy, which incorporates a priori knowledge. This method is called
	 * repeatedly during the optimization of the snake, but only as long as the
	 * method <code>isAlive()</code> returns <code>true</code>. It is imperative
	 * that this function be everywhere differentiable with respect to the
	 * snake-defining nodes.
	 */
	@Override
	public double energy() {
		if (!immortal_) {
			life_--;
			if (life_ <= 0) {
				System.out.println("gamma: "+ gamma_ );
				alive_ = false;
				long stop = System.currentTimeMillis();
				System.out.println("Stop "+ stop);
			}
		}

		double E;
		switch (energyType_) {
		case CONTOUR:
			E = computeContourEnergy();
			break;
		case REGION:
			E = computeRegionEnergy();
			break;
		case MIXTURE:
			E = alpha_ * computeContourEnergy() + (1 - alpha_)
					* computeRegionEnergy();
			break;
		default:
			E = Double.MAX_VALUE;
			break;
		}

		double boundaryEnergy = computeImageEdgeEnergy();
		double stiffnessEnergy = computeReparameterizationEnergy();

				
		if (detectType_ == CylinderSnakeTargetType.DARK) {
			return -E + boundaryEnergy + gamma_ * stiffnessEnergy;
		} else if (detectType_ == CylinderSnakeTargetType.BRIGHT) {
			return E + boundaryEnergy + gamma_ * stiffnessEnergy;
		} else {
			return Double.MAX_VALUE;
		}
	}

	// ----------------------------------------------------------------------------

	/** Initializes the snake class. */
	@Override
	public void initialize() {
		switch (BASIS_FUNCTION) {
		case ESPLINE3:
			NR2_ = 2 * BSplineBasis.ESPLINE3SUPPORT * nSamplesPerSegment_;
			break;
		case ESPLINE4:
			NR2_ = 2 * BSplineBasis.ESPLINE4SUPPORT * nSamplesPerSegment_;
			break;
		case LINEARBSPLINE:
			NR2_ = 2 * BSplineBasis.LINEARBSPLINESUPPORT * nSamplesPerSegment_;
			break;
		case QUADRATICBSPLINE:
			NR2_ = 2 * BSplineBasis.QUADRATICBSPLINESUPPORT
					* nSamplesPerSegment_;
			break;
		case CUBICBSPLINE:
			NR2_ = 2 * BSplineBasis.CUBICBSPLINESUPPORT * nSamplesPerSegment_;
			break;
		case MSPLINE:
			NR2_ = 2 * BSplineBasis.MSPLINESUPPORT * nSamplesPerSegment_;
			break;
		}

		MR_ = M_ * nSamplesPerSegment_;
		MR2_ = 2 * MR_;

		PIM_ = Math.PI / M_;
		PI2M_ = 2 * PIM_;

		xSnakeContour_ = new double[MR_][MR_ + 1];
		ySnakeContour_ = new double[MR_][MR_ + 1];
		zSnakeContour_ = new double[MR_][MR_ + 1];
		xSnakeShellContour_ = new double[MR_][MR_ + 1];
		ySnakeShellContour_ = new double[MR_][MR_ + 1];
		zSnakeShellContour_ = new double[MR_][MR_ + 1];

		dxdu_ = new double[MR_][MR_ + 1];
		dydu_ = new double[MR_][MR_ + 1];
		dzdu_ = new double[MR_][MR_ + 1];

		dxdv_ = new double[MR_][MR_ + 1];
		dydv_ = new double[MR_][MR_ + 1];
		dzdv_ = new double[MR_][MR_ + 1];

		ddxdudu_ = new double[MR_][MR_ + 1];
		ddxdudv_ = new double[MR_][MR_ + 1];
		ddxdvdv_ = new double[MR_][MR_ + 1];

		ddydudu_ = new double[MR_][MR_ + 1];
		ddydudv_ = new double[MR_][MR_ + 1];
		ddydvdv_ = new double[MR_][MR_ + 1];

		ddzdudu_ = new double[MR_][MR_ + 1];
		ddzdudv_ = new double[MR_][MR_ + 1];
		ddzdvdv_ = new double[MR_][MR_ + 1];

		dyWedgedz_ = new double[MR_][MR_ + 1];
		dzWedgedx_ = new double[MR_][MR_ + 1];
		dxWedgedy_ = new double[MR_][MR_ + 1];

		// snakeSurfaceBoundingBox_ = new Rectangle3D();

		life_ = maxLife_;
		buildLUTs();
		initializeDefaultShape();

		updateSnakeSkin();
		updateSnakeTangentBundle();
		updateArea();
		if (energyType_ == CylinderSnakeEnergyType.REGION
				|| energyType_ == CylinderSnakeEnergyType.MIXTURE) {
			updateVolume();
		}
		if (energyType_ == CylinderSnakeEnergyType.REGION
				|| energyType_ == CylinderSnakeEnergyType.MIXTURE) {
			updateSnakeShell();
		}
		isInitialized_ = true;
	}

	// ----------------------------------------------------------------------------

	/**
	 * Sets the status of the snake to alive, and restores the maximum number
	 * iterations to the original one.
	 */
	@Override
	public void reviveSnake() {
		alive_ = true;
		life_ = maxLife_;
	}

	// ----------------------------------------------------------------------------

	/** Saves the snake-defining parameters in an XML file. , should overide a function in Snake3D*/
	
	public void saveToXML(Element node) {
		getSnakeParameters().saveToXML(node);
		Element controlPointsElement = XMLUtil.addElement(node,
				ID_CONTROL_POINTS);
		for (Snake3DNode controlPoint : coef_) {
			Element controlPointElement = XMLUtil.addElement(
					controlPointsElement, ID_CONTROL_POINT);
			controlPoint.saveToXML(controlPointElement);
		}
	}
	// ----------------------------------------------------------------------------

	/** This method provides a mutator to the snake-defining nodes. */
	@Override
	public void setNodes(Snake3DNode[] node) {
		for (int i = 0; i < coef_.length; i++) {
			coef_[i].x = node[i].x;
			coef_[i].y = node[i].y;
			coef_[i].z = node[i].z;
		}
		updateSnakeSkin();
		updateSnakeTangentBundle();
		updateArea();
		if (energyType_ == CylinderSnakeEnergyType.REGION
				|| energyType_ == CylinderSnakeEnergyType.MIXTURE) {
			updateVolume();
		}
		if (energyType_ == CylinderSnakeEnergyType.REGION
				|| energyType_ == CylinderSnakeEnergyType.MIXTURE) {
			updateSnakeShell();
		}
	}

	// ----------------------------------------------------------------------------

	/** Sets the snake execution parameters. */
	public void setSnakeParameters(CylinderSnakeParameters parameters) {
		maxLife_ = parameters.getMaxLife();
		alpha_ = parameters.getAlpha();
		gamma_ = parameters.getGamma();
		immortal_ = parameters.isImmortal();
		energyType_ = parameters.getEnergyType();
		detectType_ = parameters.getDetectType();
		int M = parameters.getM();
		if (M != M_) {
			new AnnounceFrame(
					"Changing number of control points during runtime not supported yet.");
			// TODO Enable to change the number of control points on the fly.
		}
		reviveSnake();
	}

	// ----------------------------------------------------------------------------

	/**
	 * The purpose of this method is to compute the gradient of the snake energy
	 * with respect to the snake-defining nodes.
	 */
	@Override
	public Point3d[] getEnergyGradient() {
		return null;
	}

	// ----------------------------------------------------------------------------

	/** This method provides an accessor to the snake-defining nodes. */
	@Override
	public Snake3DNode[] getNodes() {
		return coef_;
	}

	// ----------------------------------------------------------------------------

	/** Returns the number of points that define the snake. */
	@Override
	public int getNumNodes() {
		return coef_.length;
	}

	// ----------------------------------------------------------------------------

	/**
	 * Returns the number of scales provided by the method
	 * <code>getScales()</code>.
	 */
	@Override
	public int getNumScales() {
		return 2 * MR_ + 1;
	}

	// ----------------------------------------------------------------------------

	/** Computes the i-th scale. */
	@Override
	public Snake3DScale getScale(int i) {
		Snake3DScale skin = null;
		if (i >= 0 && i < MR_) {
			skin = new Snake3DScale(MR_ + 1, Color.RED, false);
			double[][] points = skin.getCoordinates();
			for (int v = 0; v <= MR_; v++) {
				points[v][0] = xSnakeContour_[i][v];
				points[v][1] = ySnakeContour_[i][v];
				points[v][2] = zSnakeContour_[i][v];
			}
		} else if (i >= MR_ && i < 2 * MR_ + 1) {
			skin = new Snake3DScale(MR_, Color.RED, true);
			double[][] points = skin.getCoordinates();
			for (int u = 0; u < MR_; u++) {
				points[u][0] = xSnakeContour_[u][i - MR_];
				points[u][1] = ySnakeContour_[u][i - MR_];
				points[u][2] = zSnakeContour_[u][i - MR_];
			}
		}
		return skin;
	}

	// ----------------------------------------------------------------------------

	/**
	 * The purpose of this method is to determine what to draw on screen, given
	 * the current configuration of nodes. Collectively, the array of scales
	 * forms the skin of the snake.
	 */
	@Override
	public Snake3DScale[] getScales() {
		int numScales = getNumScales();
		Snake3DScale[] skin = new Snake3DScale[numScales];
		for (int i = 0; i < numScales; i++) {
			skin[i] = getScale(i);
		}
		return skin;
	}

	// ----------------------------------------------------------------------------

	/** Retrieves the area of the surface of the snake. */
	public double getArea() {
		return Math.abs(area_);
	}

	// ----------------------------------------------------------------------------

	/**
	 * Returns a <code>Sequence</code> object containing a binary image
	 * representing the voxels enclosed by the snake surface.
	 */
	@Override
	public Sequence getBinaryMask() {

		vtkPolyData mesh = getQuadMesh();

		ProgressFrame pFrame = new ProgressFrame("Computing binary mask");
		pFrame.setLength(imageLUTs_.getImageDepth());

		double[] bounds = mesh.GetBounds();

		int z0 = Math.max((int) Math.floor(bounds[4]), 0);
		int z1 = Math.min((int) Math.ceil(bounds[5]),
				imageLUTs_.getImageDepth() - 1);

		Binarizer3D[] rays = new Binarizer3D[z1 - z0 + 1];
		for (int z = z0; z <= z1; z++) {
			rays[z - z0] = new Binarizer3D(imageLUTs_.getImageWidth(),
					imageLUTs_.getImageHeight(), pFrame, z, mesh, bounds);
		}

		ExecutorService executor = Executors.newFixedThreadPool(8);
		for (Binarizer3D task : rays) {
			executor.execute(task);
		}
		executor.shutdown();
		while (!executor.isTerminated()) {
		}

		Sequence seq = new Sequence();
		seq.setName("Binary mask");
		for (int z = 0; z < imageLUTs_.getImageDepth(); z++) {
			if (z >= z0 && z <= z1) {
				seq.addImage(rays[z - z0].getMask());
			} else {
				seq.addImage(new IcyBufferedImage(imageLUTs_.getImageWidth(),
						imageLUTs_.getImageHeight(), 1, DataType.BYTE));
			}
		}
		pFrame.close();
		return seq;
	}

	// ----------------------------------------------------------------------------

	/**
	 * Returns the bounding box aligned with the axis containing the snake
	 * surface.
	 */
	public Rectangle3D getBounds() {
		// TODO Auto-generated method stub
		return null;
	}

	// ----------------------------------------------------------------------------

	/**
	 * Returns a point with the position of the center of gravity of the scales.
	 */
	@Override
	public Point3d getCentroid() {
		int np = M_ * (M_ - 1) + 2;
		Point3d centroid = new Point3d();
		for (int i = 0; i < np; i++) {
			centroid.x += coef_[i].x;
			centroid.y += coef_[i].y;
			centroid.z += coef_[i].z;
		}
		centroid.x /= np;
		centroid.y /= np;
		centroid.z /= np;
		return centroid;
	}

	// ----------------------------------------------------------------------------

	/**
	 * Returns a new container with the information of the execution parameters
	 * of the snake.
	 */
	public CylinderSnakeParameters getSnakeParameters() {
		return new CylinderSnakeParameters(maxLife_, M_, alpha_, gamma_,
				immortal_, detectType_, energyType_);
	}

	// ----------------------------------------------------------------------------

	/** Retrieves the volume inside the surface determined by the snake. */
	public double getVolume() {
		return Math.abs(volume_);
	}

	// ----------------------------------------------------------------------------

	/** Returns a quad mesh representing the snake surface. */
	public vtkPolyData getQuadMesh() {

		ProgressFrame pFrame = new ProgressFrame("Computing quad mesh");
		pFrame.setLength(MR_);

		int nPoints = MR_ * (MR_ + 1);
		int nQuads = MR_ * MR_;

		vtkPoints points = new vtkPoints();
		points.SetNumberOfPoints(nPoints);

		vtkPolyData quadMesh = new vtkPolyData();
		quadMesh.Allocate(nQuads, nQuads);

		for (int u = 0; u < MR_; u++) {
			for (int v = 0; v <= MR_; v++) {
				points.InsertPoint(v + u * (MR_ + 1), xSnakeContour_[u][v],
						ySnakeContour_[u][v], zSnakeContour_[u][v]);
			}
		}

		for (int u = 0; u < MR_; u++) {
			for (int v = 0; v < MR_; v++) {
				vtkQuad quad = new vtkQuad();
				quad.GetPointIds().SetId(0, v + u * (MR_ + 1));
				quad.GetPointIds().SetId(1, v + ((u + 1) % MR_) * (MR_ + 1));
				quad.GetPointIds()
						.SetId(2, v + 1 + ((u + 1) % MR_) * (MR_ + 1));
				quad.GetPointIds().SetId(3, v + 1 + u * (MR_ + 1));
				quadMesh.InsertNextCell(quad.GetCellType(), quad.GetPointIds());
			}
			pFrame.incPosition();
		}
		quadMesh.SetPoints(points);
		pFrame.close();

		return quadMesh;
	}

	// ----------------------------------------------------------------------------

	/** The purpose of this method is to monitor the status of the snake. */
	@Override
	public boolean isAlive() {
		return alive_;
	}

	// ----------------------------------------------------------------------------

	/** Returns <code>true</code> if the snake has been initialized. */
	@Override
	public boolean isInitialized() {
		return isInitialized_;
	}

	// ----------------------------------------------------------------------------

	/**
	 * Returns <code>false</code> if the snake reaches an invalid configuration.
	 * It can be related to ill-posed configuration of the points or related to
	 * the image boundaries.
	 */
	public boolean isValid() {
		return valid_;
	}

	// ----------------------------------------------------------------------------
	// GEOMETRIC TRANSFORMATION METHODS

	public void dilateX(double a) {
		Point3d g = getCentroid();
		for (int i = 0; i < coef_.length; i++) {
			double x = coef_[i].x - g.x;
			double y = coef_[i].y - g.y;
			double z = coef_[i].z - g.z;
			coef_[i].x = a * x + g.x;
			coef_[i].y = y + g.y;
			coef_[i].z = z + g.z;
		}
	}

	// ----------------------------------------------------------------------------

	public void dilateY(double b) {
		Point3d g = getCentroid();
		for (int i = 0; i < coef_.length; i++) {
			double x = coef_[i].x - g.x;
			double y = coef_[i].y - g.y;
			double z = coef_[i].z - g.z;
			coef_[i].x = x + g.x;
			coef_[i].y = b * y + g.y;
			coef_[i].z = z + g.z;
		}
	}

	// ----------------------------------------------------------------------------

	public void dilateZ(double c) {
		Point3d g = getCentroid();
		for (int i = 0; i < coef_.length; i++) {
			double x = coef_[i].x - g.x;
			double y = coef_[i].y - g.y;
			double z = coef_[i].z - g.z;
			coef_[i].x = x + g.x;
			coef_[i].y = y + g.y;
			coef_[i].z = c * z + g.z;
		}
	}

	// ----------------------------------------------------------------------------

	public void rotateX(double gamma) {
		Point3d g = getCentroid();
		for (int i = 0; i < coef_.length; i++) {
			double x = coef_[i].x - g.x;
			double y = coef_[i].y - g.y;
			double z = coef_[i].z - g.z;
			coef_[i].x = x + g.x;
			coef_[i].y = y * Math.cos(gamma) - z * Math.sin(gamma) + g.y;
			coef_[i].z = y * Math.sin(gamma) + z * Math.cos(gamma) + g.z;
		}

	}

	// ----------------------------------------------------------------------------

	public void rotateY(double beta) {
		Point3d g = getCentroid();
		for (int i = 0; i < coef_.length; i++) {
			double x = coef_[i].x - g.x;
			double y = coef_[i].y - g.y;
			double z = coef_[i].z - g.z;
			coef_[i].x = x * Math.cos(beta) + z * Math.sin(beta) + g.x;
			coef_[i].y = y + g.y;
			coef_[i].z = -x * Math.sin(beta) + z * Math.cos(beta) + g.z;
		}
	}

	// ----------------------------------------------------------------------------

	public void rotateZ(double alpha) {
		Point3d g = getCentroid();
		for (int i = 0; i < coef_.length; i++) {
			double x = coef_[i].x - g.x;
			double y = coef_[i].y - g.y;
			double z = coef_[i].z - g.z;
			coef_[i].x = x * Math.cos(alpha) - y * Math.sin(alpha) + g.x;
			coef_[i].y = x * Math.sin(alpha) + y * Math.cos(alpha) + g.y;
			coef_[i].z = z + g.z;
		}
	}

	// ============================================================================
	// PRIVATE METHODS

	/** Initializes all LUTs of the class. */
	private void buildLUTs() {
		bSpline1LUT_ = new double[NR2_];
		bSpline2LUT_ = new double[NR2_];
		bSpline1DerivativeLUT_ = new double[NR2_];
		bSpline2DerivativeLUT_ = new double[NR2_];
		bSpline1SecondDerivativeLUT_ = new double[NR2_];
		bSpline2SecondDerivativeLUT_ = new double[NR2_];
		cosLUT_ = new double[M_];
		sinLUT_ = new double[M_];

		double currentVal = 0;
		double R2 = 2.0 * nSamplesPerSegment_;
		for (int i = 0; i < NR2_; i++) {
			currentVal = i / R2;
			bSpline1LUT_[i] = BSplineBasis.MSpline(currentVal, PI2M_);
			bSpline2LUT_[i] = BSplineBasis.Keys(currentVal);
			bSpline1DerivativeLUT_[i] = BSplineBasis.MSpline_Prime(currentVal,
					PI2M_);
			bSpline2DerivativeLUT_[i] = BSplineBasis.Keys_Prime(currentVal);
			bSpline1SecondDerivativeLUT_[i] = BSplineBasis.MSpline_PrimePrime(
					currentVal, PI2M_);
			bSpline2SecondDerivativeLUT_[i] = BSplineBasis
					.Keys_PrimePrime(currentVal);
		}
		// double aux = 2.0 * (1.0 - Math.cos(PI2M_))
		// / (Math.cos(PIM_) - Math.cos(PI2M_ + PIM_));
		for (int i = 0; i < M_; i++) {
			double theta = PI2M_ * i;
			cosLUT_[i] = Math.cos(theta); // aux * Math.cos(theta);
			sinLUT_[i] = Math.sin(theta); // aux * Math.sin(theta);
		}
	}

	// ----------------------------------------------------------------------------

	/** Initializes the snake control points with a predefined shape. */
	private void initializeDefaultShape() {
		coef_ = new Snake3DNode[M_ * (M_ + 3)];

		double r = 3; //Math.min(Math.min(imageLUTs_.getImageWidth() / 10.0, imageLUTs_.getImageHeight() / 10.0), imageLUTs_.getImageDepth() / 50.0);
		double factor = 15;
		double x0 = imageLUTs_.getImageWidth() / 2.2;
		double y0 = imageLUTs_.getImageHeight() / 1.6;
		double z0 = (imageLUTs_.getImageDepth() /2.0 )- (((M_)*factor)/2);
		

		for (int k = 0; k < M_; k++) {
			for (int l = 1; l <= M_ + 3; l++) {
				coef_[k * (M_ + 3) + (l - 1)] = new Snake3DNode(x0 + r
						* cosLUT_[k], y0 + r * sinLUT_[k], z0 + factor
						* (l - 2));
			}
		}

	}

	// ----------------------------------------------------------------------------

	/** Computes the surface of the snake from the control points. */
	private void updateSnakeSkin() {
		int index_i, index_j;
		double xPosVal, yPosVal, zPosVal, aux, basisFactor;
		for (int i = 0; i < MR_; i++) {
			for (int j = 0; j <= MR_; j++) {
				xPosVal = 0;
				yPosVal = 0;
				zPosVal = 0;
				for (int l = 1; l <= M_ + 3; l++) {
					index_j = 2 * j + nSamplesPerSegment_ * (4 - 2 * (l - 2));
					if (index_j >= 0 && index_j < NR2_) {
						aux = bSpline2LUT_[index_j];
						for (int k = 0; k < M_; k++) {
							index_i = (2 * i + nSamplesPerSegment_
									* (4 - 2 * k) + MR2_)
									% MR2_;
							if (index_i >= 0 && index_i < NR2_) {
								basisFactor = aux * bSpline1LUT_[index_i];
								xPosVal += coef_[k * (M_ + 3) + l - 1].x
										* basisFactor;
								yPosVal += coef_[k * (M_ + 3) + l - 1].y
										* basisFactor;
								zPosVal += coef_[k * (M_ + 3) + l - 1].z
										* basisFactor;
							}
						}
					}
				}xSnakeContour_[i][j] = xPosVal;
				ySnakeContour_[i][j] = yPosVal;
				zSnakeContour_[i][j] = zPosVal;
			}
		}
	}

	// ----------------------------------------------------------------------------

	/** Computes the snake shell needed for the region energy. */
	private void updateSnakeShell() {
		double xg = 0;
		double yg = 0;
		double zg = 0;

		int length = MR_ * (MR_ + 1);
		// center of gravity
		for (int u = 0; u < MR_; u++) {
			for (int v = 0; v <= MR_; v++) {
				xg += xSnakeContour_[u][v];
				yg += ySnakeContour_[u][v];
				zg += zSnakeContour_[u][v];
			}
		}
		xg /= length;
		yg /= length;
		zg /= length;

		for (int u = 0; u < MR_; u++) {
			for (int v = 0; v <= MR_; v++) {
				xSnakeShellContour_[u][v] = CBRT * (xSnakeContour_[u][v] - xg)
						+ xg;
				ySnakeShellContour_[u][v] = CBRT * (ySnakeContour_[u][v] - yg)
						+ yg;
				zSnakeShellContour_[u][v] = CBRT * (zSnakeContour_[u][v] - zg)
						+ zg;
			}
		}
	}

	// ----------------------------------------------------------------------------

	/** Computes the vectors of the tangent bundle. */
	private void updateSnakeTangentBundle() {

		// d sigma / du
		int index_i, index_j;
		double xPosVal, yPosVal, zPosVal, aux, basisFactor;

		for (int i = 0; i < MR_; i++) {
			for (int j = 0; j <= MR_; j++) {
				xPosVal = 0;
				yPosVal = 0;
				zPosVal = 0;
				for (int l = 1; l <= M_ + 3; l++) {
					index_j = 2 * j + nSamplesPerSegment_ * (4 - 2 * (l - 2));
					if (index_j >= 0 && index_j < NR2_) {
						aux = bSpline2LUT_[index_j];
						for (int k = 0; k < M_; k++) {
							index_i = (2 * i + nSamplesPerSegment_
									* (4 - 2 * k) + MR2_)
									% MR2_;
							if (index_i >= 0 && index_i < NR2_) {
								basisFactor = aux
										* bSpline1DerivativeLUT_[index_i];
								xPosVal += coef_[k * (M_ + 3) + l - 1].x
										* basisFactor;
								yPosVal += coef_[k * (M_ + 3) + l - 1].y
										* basisFactor;
								zPosVal += coef_[k * (M_ + 3) + l - 1].z
										* basisFactor;
							}
						}
					}
				}

				
				dxdu_[i][j] = M_ * xPosVal;
				dydu_[i][j] = M_ * yPosVal;
				dzdu_[i][j] = M_ * zPosVal;
			}
		}

		// d sigma / dv
		for (int i = 0; i < MR_; i++) {
			for (int j = 0; j <= MR_; j++) {
				xPosVal = 0;
				yPosVal = 0;
				zPosVal = 0;
				for (int l = 1; l <= M_ + 3; l++) {
					index_j = 2 * j + nSamplesPerSegment_ * (4 - 2 * (l - 2));
					if (index_j >= 0 && index_j < NR2_) {
						aux = bSpline2DerivativeLUT_[index_j];
						for (int k = 0; k < M_; k++) {
							index_i = (2 * i + nSamplesPerSegment_
									* (4 - 2 * k) + MR2_)
									% MR2_;
							if (index_i >= 0 && index_i < NR2_) {
								basisFactor = aux * bSpline1LUT_[index_i];
								xPosVal += coef_[k * (M_ + 3) + l - 1].x
										* basisFactor;
								yPosVal += coef_[k * (M_ + 3) + l - 1].y
										* basisFactor;
								zPosVal += coef_[k * (M_ + 3) + l - 1].z
										* basisFactor;
							}
						}
					}
				}
				
				dxdv_[i][j] = M_ * xPosVal;
				dydv_[i][j] = M_ * yPosVal;
				dzdv_[i][j] = M_ * zPosVal;
			}
		}

		// normal vectorfield
		for (int i = 0; i < MR_; i++) {
			for (int j = 0; j <= MR_; j++) {
				dyWedgedz_[i][j] = (dydu_[i][j] * dzdv_[i][j] - dydv_[i][j]
						* dzdu_[i][j])
						/ JACOBIANCONSTANT;
				dzWedgedx_[i][j] = (dxdv_[i][j] * dzdu_[i][j] - dxdu_[i][j]
						* dzdv_[i][j])
						/ JACOBIANCONSTANT;
				dxWedgedy_[i][j] = (dxdu_[i][j] * dydv_[i][j] - dxdv_[i][j]
						* dydu_[i][j])
						/ JACOBIANCONSTANT;
			}
		}

	}

	// ----------------------------------------------------------------------------

	/** Computes the area of the surface of the snake. */
	private void updateArea() {
		area_ = 0;
		double delta_uv = 1.0 / (MR_ * MR_);
		for (int i = 0; i < MR_; i++) {
			for (int j = 0; j <= MR_; j++) {
				area_ += JACOBIANCONSTANT
						* delta_uv
						* Math.sqrt((dyWedgedz_[i][j] * dyWedgedz_[i][j]
								+ dzWedgedx_[i][j] * dzWedgedx_[i][j] + dxWedgedy_[i][j]
								* dxWedgedy_[i][j]));
			}
		}
	}

	// ----------------------------------------------------------------------------

	/** Computes the volume enclosed by the snake. */
	private void updateVolume() {
		volume_ = 0;
		double delta_uv = 1.0 / (MR_ * MR_);
		for (int u = 0; u < MR_; u++) {
			for (int v = 0; v <= MR_; v++) {
				volume_ += ySnakeContour_[u][v] * delta_uv * dzWedgedx_[u][v];
			}
		}
		volume_ *= JACOBIANCONSTANT;
	}

	// ----------------------------------------------------------------------------
	// ENERGY METHODS

	/** Computes the contour energy. */
	private double computeContourEnergy() {
		double delta_uv = 1.0 / (MR_ * MR_);
		double energy = 0.0;
		double fyLap_val;
		int x0, x1, y0, y1, z0, z1;
		double DeltaX, DeltaY, DeltaZ;

		int imageWidth = imageLUTs_.getImageWidth();
		int imageHeight = imageLUTs_.getImageHeight();
		int imageDepth = imageLUTs_.getImageDepth();
		double[][] preintegratedFilteredImageDataArray = imageLUTs_
				.getPreintegratedFilteredImageDataArray();

		double deltax = 0;
		double deltay = -0.5;
		double deltaz = 0;
		for (int u = 0; u < MR_; u++) {
			for (int v = 0; v <= MR_; v++) {
				x0 = (int) Math.floor(xSnakeContour_[u][v] + deltax);
				y0 = (int) Math.floor(ySnakeContour_[u][v] + deltay);
				z0 = (int) Math.floor(zSnakeContour_[u][v] + deltaz);

				if (x0 < 1) {
					x0 = 1;
				} else if (x0 > imageWidth - 2) {
					x0 = imageWidth - 2;
				}

				if (y0 < 1) {
					y0 = 1;
				} else if (y0 > imageHeight - 2) {
					y0 = imageHeight - 2;
				}

				if (z0 < 1) {
					z0 = 1;
				} else if (z0 > imageDepth - 2) {
					z0 = imageDepth - 2;
				}

				x1 = x0 + 1;
				y1 = y0 + 1;
				z1 = z0 + 1;

				DeltaX = xSnakeContour_[u][v] + deltax - x0;
				DeltaY = ySnakeContour_[u][v] + deltay - y0;
				DeltaZ = zSnakeContour_[u][v] + deltaz - z0;

				double i1 = (1 - DeltaZ)
						* preintegratedFilteredImageDataArray[z0][x0
								+ imageWidth * y0]
						+ DeltaZ
						* preintegratedFilteredImageDataArray[z1][x0
								+ imageWidth * y0];
				double i2 = (1 - DeltaZ)
						* preintegratedFilteredImageDataArray[z0][x0
								+ imageWidth * y1]
						+ DeltaZ
						* preintegratedFilteredImageDataArray[z1][x0
								+ imageWidth * y1];
				double j1 = (1 - DeltaZ)
						* preintegratedFilteredImageDataArray[z0][x1
								+ imageWidth * y0]
						+ DeltaZ
						* preintegratedFilteredImageDataArray[z1][x1
								+ imageWidth * y0];
				double j2 = (1 - DeltaZ)
						* preintegratedFilteredImageDataArray[z0][x1
								+ imageWidth * y1]
						+ DeltaZ
						* preintegratedFilteredImageDataArray[z1][x1
								+ imageWidth * y1];
				double w1 = i1 * (1 - DeltaY) + i2 * DeltaY;
				double w2 = j1 * (1 - DeltaY) + j2 * DeltaY;
				fyLap_val = w1 * (1 - DeltaX) + w2 * DeltaX;
				energy += fyLap_val * delta_uv * dzWedgedx_[u][v];
			}
		}
		//System.out.println("energy  = " + 1000 * energy);
		return 10000 * energy;
	}

	// ----------------------------------------------------------------------------

	/** Computes the region energy. */
	private double computeRegionEnergy() {
		double[][] preintegratedImageDataArray = imageLUTs_
				.getPreintegratedImageDataArray();
		double internalContribution = computeSurfaceIntegral(
				preintegratedImageDataArray, xSnakeContour_, ySnakeContour_,
				zSnakeContour_);
		double externalContribution = CBRT
				* computeSurfaceIntegral(preintegratedImageDataArray,
						xSnakeShellContour_, ySnakeShellContour_,
						zSnakeShellContour_);
		double energy = 1000
				* (externalContribution - 2 * internalContribution) / volume_;
		return energy;
	}

	// ----------------------------------------------------------------------------

	/** Computes the image boundary energy. */
	private double computeImageEdgeEnergy() {
		double energy = 0.0;
		double x, y, z;

		double borderEnergy = 100000;
		double strength = 1;

		int imageWidth = imageLUTs_.getImageWidth();
		int imageHeight = imageLUTs_.getImageHeight();
		int imageDepth = imageLUTs_.getImageDepth();

		for (int u = 0; u < MR_; u++) {
			for (int v = 0; v <= MR_; v++) {
				x = xSnakeContour_[u][v];
				y = ySnakeContour_[u][v];
				z = zSnakeContour_[u][v];

				energy += borderEnergy * Math.exp(strength * (x - imageWidth));
				energy += borderEnergy * Math.exp(strength * (y - imageHeight));
				energy += borderEnergy * Math.exp(strength * (z - imageDepth));

				energy += borderEnergy * Math.exp(-strength * x);
				energy += borderEnergy * Math.exp(-strength * y);
				energy += borderEnergy * Math.exp(-strength * z);
			}
		}

		if (energyType_ == CylinderSnakeEnergyType.REGION
				|| energyType_ == CylinderSnakeEnergyType.MIXTURE) {
			for (int u = 0; u < MR_; u++) {
				for (int v = 0; v <= MR_; v++) {
					x = xSnakeShellContour_[u][v];
					y = ySnakeShellContour_[u][v];
					z = zSnakeShellContour_[u][v];

					energy += borderEnergy
							* Math.exp(strength * (x - imageWidth));
					energy += borderEnergy
							* Math.exp(strength * (y - imageHeight));
					energy += borderEnergy
							* Math.exp(strength * (z - imageDepth));

					energy += borderEnergy * Math.exp(-strength * x);
					energy += borderEnergy * Math.exp(-strength * y);
					energy += borderEnergy * Math.exp(-strength * z);
				}
			}
		}

		energy /= M_ * M_;
		return 100 * energy;
	}

	// ----------------------------------------------------------------------------

	/** Computes the reparameterization energy. */
	private double computeReparameterizationEnergy() {
		double reparameterizationEnergy = 0;
		double delta_uv = 1.0 / (MR_ * MR_);
		for (int i = 0; i < MR_; i++) {
			for (int j = 0; j <= MR_; j++) {
				reparameterizationEnergy += delta_uv
						* Math.abs(area_
								- JACOBIANCONSTANT
								* Math.sqrt((dyWedgedz_[i][j]
										* dyWedgedz_[i][j] + dzWedgedx_[i][j]
										* dzWedgedx_[i][j] + dxWedgedy_[i][j]
										* dxWedgedy_[i][j])));
			}
		}
		return reparameterizationEnergy / 1000;
	}

	// ----------------------------------------------------------------------------

	/** Integrates a function over a volume. */
	private double computeSurfaceIntegral(double[][] fy, double[][] x,
			double[][] y, double[][] z) {
		double delta_uv = 1.0 / (MR_ * MR_);
		double fy_val;
		int x0, x1, y0, y1, z0, z1;
		double DeltaX, DeltaY, DeltaZ;

		int imageWidth = imageLUTs_.getImageWidth();
		int imageHeight = imageLUTs_.getImageHeight();
		int imageDepth = imageLUTs_.getImageDepth();

		double surfaceIntegral = 0.0;
		for (int u = 0; u < MR_; u++) {
			for (int v = 0; v <= MR_; v++) {
				x0 = (int) Math.floor(x[u][v] + 0.5);
				y0 = (int) Math.floor(y[u][v] - 0.5);
				z0 = (int) Math.floor(z[u][v] + 0.5);

				if (x0 < 1) {
					x0 = 1;
				} else if (x0 > imageWidth - 2) {
					x0 = imageWidth - 2;
				}

				if (y0 < 1) {
					y0 = 1;
				} else if (y0 > imageHeight - 2) {
					y0 = imageHeight - 2;
				}

				if (z0 < 1) {
					z0 = 1;
				} else if (z0 > imageDepth - 2) {
					z0 = imageDepth - 2;
				}

				x1 = x0 + 1;
				y1 = y0 + 1;
				z1 = z0 + 1;

				DeltaX = x[u][v] - 0.5 - x0;
				DeltaY = y[u][v] - 0.5 - y0;
				DeltaZ = z[u][v] - 0.5 - z0;
				double i1 = (1 - DeltaZ) * fy[z0][x0 + imageWidth * y0]
						+ DeltaZ * fy[z1][x0 + imageWidth * y0];
				double i2 = (1 - DeltaZ) * fy[z0][x0 + imageWidth * y1]
						+ DeltaZ * fy[z1][x0 + imageWidth * y1];
				double j1 = (1 - DeltaZ) * fy[z0][x1 + imageWidth * y0]
						+ DeltaZ * fy[z1][x1 + imageWidth * y0];
				double j2 = (1 - DeltaZ) * fy[z0][x1 + imageWidth * y1]
						+ DeltaZ * fy[z1][x1 + imageWidth * y1];
				double w1 = i1 * (1 - DeltaY) + i2 * DeltaY;
				double w2 = j1 * (1 - DeltaY) + j2 * DeltaY;
				fy_val = w1 * (1 - DeltaX) + w2 * DeltaX;
				surfaceIntegral += fy_val * delta_uv * dzWedgedx_[u][v];
			}
		}
		return JACOBIANCONSTANT * surfaceIntegral;
	}

	// ----------------------------------------------------------------------------
	// EXPERIMENTAL METHODS

	/** Computes the Euler characteristic using the Gauss-Bonnet theorem. */
	@SuppressWarnings("unused")
	private double computeEulerCharacteristic() {
		// dd sigma / dudu
		int index_u, index_v;
		double xPosVal, yPosVal, zPosVal, aux, basisFactor;

		for (int i = 0; i < MR_; i++) {
			for (int j = 0; j <= MR_; j++) {
				xPosVal = 0;
				yPosVal = 0;
				zPosVal = 0;
				for (int l = 1; l <= M_ - 1; l++) {
					index_v = 2 * j + nSamplesPerSegment_ * (3 - 2 * l);
					if (index_v >= 0 && index_v < NR2_) {
						aux = bSpline2LUT_[index_v];
						for (int k = 0; k < M_; k++) {
							index_u = (2 * i + nSamplesPerSegment_
									* (3 - 2 * k) + MR2_)
									% MR2_;
							if (index_u >= 0 && index_u < NR2_) {
								basisFactor = aux
										* bSpline1SecondDerivativeLUT_[index_u];
								xPosVal += coef_[k + (l - 1) * M_].x
										* basisFactor;
								yPosVal += coef_[k + (l - 1) * M_].y
										* basisFactor;
								zPosVal += coef_[k + (l - 1) * M_].z
										* basisFactor;
							}
						}
					}
				}

				double scal = 1.0 / (M_ * bSpline2DerivativeLUT_[5 * nSamplesPerSegment_]);

				double NorthV1x = coef_[M_ * (M_ - 1) + 2].x
						- coef_[M_ * (M_ - 1)].x;
				double NorthV1y = coef_[M_ * (M_ - 1) + 2].y
						- coef_[M_ * (M_ - 1)].y;
				double NorthV1z = coef_[M_ * (M_ - 1) + 2].z
						- coef_[M_ * (M_ - 1)].z;

				double NorthV2x = coef_[M_ * (M_ - 1) + 3].x
						- coef_[M_ * (M_ - 1)].x;
				double NorthV2y = coef_[M_ * (M_ - 1) + 3].y
						- coef_[M_ * (M_ - 1)].y;
				double NorthV2z = coef_[M_ * (M_ - 1) + 3].z
						- coef_[M_ * (M_ - 1)].z;

				// l = -1
				index_v = 2 * j + nSamplesPerSegment_ * 5;
				if (index_v >= 0 && index_v < NR2_) {
					aux = bSpline2LUT_[index_v];
					for (int k = 0; k < M_; k++) {
						index_u = (2 * i + nSamplesPerSegment_ * (3 - 2 * k) + MR2_)
								% MR2_;
						if (index_u >= 0 && index_u < NR2_) {
							// compute c[k,-1]
							double ckminus1x = (coef_[k].x + scal
									* (cosLUT_[k] * NorthV1x + sinLUT_[k]
											* NorthV2x));
							double ckminus1y = (coef_[k].y + scal
									* (cosLUT_[k] * NorthV1y + sinLUT_[k]
											* NorthV2y));
							double ckminus1z = (coef_[k].z + scal
									* (cosLUT_[k] * NorthV1z + sinLUT_[k]
											* NorthV2z));

							basisFactor = aux
									* bSpline1SecondDerivativeLUT_[index_u];
							xPosVal += ckminus1x * basisFactor;
							yPosVal += ckminus1y * basisFactor;
							zPosVal += ckminus1z * basisFactor;
						}
					}
				}

				// l = 0
				index_v = 2 * j + nSamplesPerSegment_ * 3;
				if (index_v >= 0 && index_v < NR2_) {
					aux = bSpline2LUT_[index_v];
					for (int k = 0; k < M_; k++) {
						index_u = (2 * i + nSamplesPerSegment_ * (3 - 2 * k) + MR2_)
								% MR2_;
						if (index_u >= 0 && index_u < NR2_) {
							// compute c[k,-1]
							double ckminus1x = (coef_[k].x + scal
									* (cosLUT_[k] * NorthV1x + sinLUT_[k]
											* NorthV2x));
							double ckminus1y = (coef_[k].y + scal
									* (cosLUT_[k] * NorthV1y + sinLUT_[k]
											* NorthV2y));
							double ckminus1z = (coef_[k].z + scal
									* (cosLUT_[k] * NorthV1z + sinLUT_[k]
											* NorthV2z));

							// compute c[k,0]
							double ck0x = (coef_[M_ * (M_ - 1)].x - bSpline2LUT_[5 * nSamplesPerSegment_]
									* (ckminus1x + coef_[k].x))
									/ bSpline2LUT_[3 * nSamplesPerSegment_];
							double ck0y = (coef_[M_ * (M_ - 1)].y - bSpline2LUT_[5 * nSamplesPerSegment_]
									* (ckminus1y + coef_[k].y))
									/ bSpline2LUT_[3 * nSamplesPerSegment_];
							double ck0z = (coef_[M_ * (M_ - 1)].z - bSpline2LUT_[5 * nSamplesPerSegment_]
									* (ckminus1z + coef_[k].z))
									/ bSpline2LUT_[3 * nSamplesPerSegment_];

							basisFactor = aux
									* bSpline1SecondDerivativeLUT_[index_u];
							xPosVal += ck0x * basisFactor;
							yPosVal += ck0y * basisFactor;
							zPosVal += ck0z * basisFactor;
						}
					}
				}

				double SouthV1x = coef_[M_ * (M_ - 1) + 4].x
						- coef_[M_ * (M_ - 1) + 1].x;
				double SouthV1y = coef_[M_ * (M_ - 1) + 4].y
						- coef_[M_ * (M_ - 1) + 1].y;
				double SouthV1z = coef_[M_ * (M_ - 1) + 4].z
						- coef_[M_ * (M_ - 1) + 1].z;

				double SouthV2x = coef_[M_ * (M_ - 1) + 5].x
						- coef_[M_ * (M_ - 1) + 1].x;
				double SouthV2y = coef_[M_ * (M_ - 1) + 5].y
						- coef_[M_ * (M_ - 1) + 1].y;
				double SouthV2z = coef_[M_ * (M_ - 1) + 5].z
						- coef_[M_ * (M_ - 1) + 1].z;

				// l = M+1
				index_v = 2 * j + nSamplesPerSegment_ * (1 - 2 * M_);
				if (index_v >= 0 && index_v < NR2_) {
					aux = bSpline2LUT_[index_v];
					for (int k = 0; k < M_; k++) {
						index_u = (2 * i + nSamplesPerSegment_ * (3 - 2 * k) + MR2_)
								% MR2_;
						if (index_u >= 0 && index_u < NR2_) {
							// compute c[k,M+1]
							double ckMplusx = (coef_[k + (M_ - 2) * M_].x + scal
									* (cosLUT_[k] * SouthV1x + sinLUT_[k]
											* SouthV2x));
							double ckMplusy = (coef_[k + (M_ - 2) * M_].y + scal
									* (cosLUT_[k] * SouthV1y + sinLUT_[k]
											* SouthV2y));
							double ckMplusz = (coef_[k + (M_ - 2) * M_].z + scal
									* (cosLUT_[k] * SouthV1z + sinLUT_[k]
											* SouthV2z));

							basisFactor = aux
									* bSpline1SecondDerivativeLUT_[index_u];
							xPosVal += ckMplusx * basisFactor;
							yPosVal += ckMplusy * basisFactor;
							zPosVal += ckMplusz * basisFactor;
						}
					}
				}

				// l = M
				index_v = 2 * j + nSamplesPerSegment_ * (3 - 2 * M_);
				if (index_v >= 0 && index_v < NR2_) {
					aux = bSpline2LUT_[index_v];
					for (int k = 0; k < M_; k++) {
						index_u = (2 * i + nSamplesPerSegment_ * (3 - 2 * k) + MR2_)
								% MR2_;
						if (index_u >= 0 && index_u < NR2_) {
							// compute c[k,M+1]
							double ckMplusx = (coef_[k + (M_ - 2) * M_].x + scal
									* (cosLUT_[k] * SouthV1x + sinLUT_[k]
											* SouthV2x));
							double ckMplusy = (coef_[k + (M_ - 2) * M_].y + scal
									* (cosLUT_[k] * SouthV1y + sinLUT_[k]
											* SouthV2y));
							double ckMplusz = (coef_[k + (M_ - 2) * M_].z + scal
									* (cosLUT_[k] * SouthV1z + sinLUT_[k]
											* SouthV2z));

							// compute c[k,M]
							double ckMx = (coef_[M_ * (M_ - 1) + 1].x - bSpline2LUT_[5 * nSamplesPerSegment_]
									* (ckMplusx + coef_[k + (M_ - 2) * M_].x))
									/ bSpline2LUT_[3 * nSamplesPerSegment_];
							double ckMy = (coef_[M_ * (M_ - 1) + 1].y - bSpline2LUT_[5 * nSamplesPerSegment_]
									* (ckMplusy + coef_[k + (M_ - 2) * M_].y))
									/ bSpline2LUT_[3 * nSamplesPerSegment_];
							double ckMz = (coef_[M_ * (M_ - 1) + 1].z - bSpline2LUT_[5 * nSamplesPerSegment_]
									* (ckMplusz + coef_[k + (M_ - 2) * M_].z))
									/ bSpline2LUT_[3 * nSamplesPerSegment_];

							basisFactor = aux
									* bSpline1SecondDerivativeLUT_[index_u];
							xPosVal += ckMx * basisFactor;
							yPosVal += ckMy * basisFactor;
							zPosVal += ckMz * basisFactor;
						}
					}
				}
				ddxdudu_[i][j] = M_ * M_ * xPosVal;
				ddydudu_[i][j] = M_ * M_ * yPosVal;
				ddzdudu_[i][j] = M_ * M_ * zPosVal;
			}
		}

		// dd sigma / dvdv
		for (int i = 0; i < MR_; i++) {
			for (int j = 0; j <= MR_; j++) {
				xPosVal = 0;
				yPosVal = 0;
				zPosVal = 0;
				for (int l = 1; l <= M_ - 1; l++) {
					index_v = 2 * j + nSamplesPerSegment_ * (3 - 2 * l);
					if (index_v >= 0 && index_v < NR2_) {
						aux = bSpline2SecondDerivativeLUT_[index_v];
						for (int k = 0; k < M_; k++) {
							index_u = (2 * i + nSamplesPerSegment_
									* (3 - 2 * k) + MR2_)
									% MR2_;
							if (index_u >= 0 && index_u < NR2_) {
								basisFactor = aux * bSpline1LUT_[index_u];
								xPosVal += coef_[k + (l - 1) * M_].x
										* basisFactor;
								yPosVal += coef_[k + (l - 1) * M_].y
										* basisFactor;
								zPosVal += coef_[k + (l - 1) * M_].z
										* basisFactor;
							}
						}
					}
				}

				double scal = 1.0 / (M_ * bSpline2DerivativeLUT_[5 * nSamplesPerSegment_]);

				double NorthV1x = coef_[M_ * (M_ - 1) + 2].x
						- coef_[M_ * (M_ - 1)].x;
				double NorthV1y = coef_[M_ * (M_ - 1) + 2].y
						- coef_[M_ * (M_ - 1)].y;
				double NorthV1z = coef_[M_ * (M_ - 1) + 2].z
						- coef_[M_ * (M_ - 1)].z;

				double NorthV2x = coef_[M_ * (M_ - 1) + 3].x
						- coef_[M_ * (M_ - 1)].x;
				double NorthV2y = coef_[M_ * (M_ - 1) + 3].y
						- coef_[M_ * (M_ - 1)].y;
				double NorthV2z = coef_[M_ * (M_ - 1) + 3].z
						- coef_[M_ * (M_ - 1)].z;

				// l = -1
				index_v = 2 * j + nSamplesPerSegment_ * 5;
				if (index_v >= 0 && index_v < NR2_) {
					aux = bSpline2SecondDerivativeLUT_[index_v];
					for (int k = 0; k < M_; k++) {
						index_u = (2 * i + nSamplesPerSegment_ * (3 - 2 * k) + MR2_)
								% MR2_;
						if (index_u >= 0 && index_u < NR2_) {
							// compute c[k,-1]
							double ckminus1x = (coef_[k].x + scal
									* (cosLUT_[k] * NorthV1x + sinLUT_[k]
											* NorthV2x));
							double ckminus1y = (coef_[k].y + scal
									* (cosLUT_[k] * NorthV1y + sinLUT_[k]
											* NorthV2y));
							double ckminus1z = (coef_[k].z + scal
									* (cosLUT_[k] * NorthV1z + sinLUT_[k]
											* NorthV2z));

							basisFactor = aux * bSpline1LUT_[index_u];
							xPosVal += ckminus1x * basisFactor;
							yPosVal += ckminus1y * basisFactor;
							zPosVal += ckminus1z * basisFactor;
						}
					}
				}

				// l = 0
				index_v = 2 * j + nSamplesPerSegment_ * 3;
				if (index_v >= 0 && index_v < NR2_) {
					aux = bSpline2SecondDerivativeLUT_[index_v];
					for (int k = 0; k < M_; k++) {
						index_u = (2 * i + nSamplesPerSegment_ * (3 - 2 * k) + MR2_)
								% MR2_;
						if (index_u >= 0 && index_u < NR2_) {
							// compute c[k,-1]
							double ckminus1x = (coef_[k].x + scal
									* (cosLUT_[k] * NorthV1x + sinLUT_[k]
											* NorthV2x));
							double ckminus1y = (coef_[k].y + scal
									* (cosLUT_[k] * NorthV1y + sinLUT_[k]
											* NorthV2y));
							double ckminus1z = (coef_[k].z + scal
									* (cosLUT_[k] * NorthV1z + sinLUT_[k]
											* NorthV2z));

							// compute c[k,0]
							double ck0x = (coef_[M_ * (M_ - 1)].x - bSpline2LUT_[5 * nSamplesPerSegment_]
									* (ckminus1x + coef_[k].x))
									/ bSpline2LUT_[3 * nSamplesPerSegment_];
							double ck0y = (coef_[M_ * (M_ - 1)].y - bSpline2LUT_[5 * nSamplesPerSegment_]
									* (ckminus1y + coef_[k].y))
									/ bSpline2LUT_[3 * nSamplesPerSegment_];
							double ck0z = (coef_[M_ * (M_ - 1)].z - bSpline2LUT_[5 * nSamplesPerSegment_]
									* (ckminus1z + coef_[k].z))
									/ bSpline2LUT_[3 * nSamplesPerSegment_];

							basisFactor = aux * bSpline1LUT_[index_u];
							xPosVal += ck0x * basisFactor;
							yPosVal += ck0y * basisFactor;
							zPosVal += ck0z * basisFactor;
						}
					}
				}

				double SouthV1x = coef_[M_ * (M_ - 1) + 4].x
						- coef_[M_ * (M_ - 1) + 1].x;
				double SouthV1y = coef_[M_ * (M_ - 1) + 4].y
						- coef_[M_ * (M_ - 1) + 1].y;
				double SouthV1z = coef_[M_ * (M_ - 1) + 4].z
						- coef_[M_ * (M_ - 1) + 1].z;

				double SouthV2x = coef_[M_ * (M_ - 1) + 5].x
						- coef_[M_ * (M_ - 1) + 1].x;
				double SouthV2y = coef_[M_ * (M_ - 1) + 5].y
						- coef_[M_ * (M_ - 1) + 1].y;
				double SouthV2z = coef_[M_ * (M_ - 1) + 5].z
						- coef_[M_ * (M_ - 1) + 1].z;

				// l = M+1
				index_v = 2 * j + nSamplesPerSegment_ * (1 - 2 * M_);
				if (index_v >= 0 && index_v < NR2_) {
					aux = bSpline2SecondDerivativeLUT_[index_v];
					for (int k = 0; k < M_; k++) {
						index_u = (2 * i + nSamplesPerSegment_ * (3 - 2 * k) + MR2_)
								% MR2_;
						if (index_u >= 0 && index_u < NR2_) {
							// compute c[k,M+1]
							double ckMplusx = (coef_[k + (M_ - 2) * M_].x + scal
									* (cosLUT_[k] * SouthV1x + sinLUT_[k]
											* SouthV2x));
							double ckMplusy = (coef_[k + (M_ - 2) * M_].y + scal
									* (cosLUT_[k] * SouthV1y + sinLUT_[k]
											* SouthV2y));
							double ckMplusz = (coef_[k + (M_ - 2) * M_].z + scal
									* (cosLUT_[k] * SouthV1z + sinLUT_[k]
											* SouthV2z));

							basisFactor = aux * bSpline1LUT_[index_u];
							xPosVal += ckMplusx * basisFactor;
							yPosVal += ckMplusy * basisFactor;
							zPosVal += ckMplusz * basisFactor;
						}
					}
				}

				// l = M
				index_v = 2 * j + nSamplesPerSegment_ * (3 - 2 * M_);
				if (index_v >= 0 && index_v < NR2_) {
					aux = bSpline2SecondDerivativeLUT_[index_v];
					for (int k = 0; k < M_; k++) {
						index_u = (2 * i + nSamplesPerSegment_ * (3 - 2 * k) + MR2_)
								% MR2_;
						if (index_u >= 0 && index_u < NR2_) {
							// compute c[k,M+1]
							double ckMplusx = (coef_[k + (M_ - 2) * M_].x + scal
									* (cosLUT_[k] * SouthV1x + sinLUT_[k]
											* SouthV2x));
							double ckMplusy = (coef_[k + (M_ - 2) * M_].y + scal
									* (cosLUT_[k] * SouthV1y + sinLUT_[k]
											* SouthV2y));
							double ckMplusz = (coef_[k + (M_ - 2) * M_].z + scal
									* (cosLUT_[k] * SouthV1z + sinLUT_[k]
											* SouthV2z));

							// compute c[k,M]
							double ckMx = (coef_[M_ * (M_ - 1) + 1].x - bSpline2LUT_[5 * nSamplesPerSegment_]
									* (ckMplusx + coef_[k + (M_ - 2) * M_].x))
									/ bSpline2LUT_[3 * nSamplesPerSegment_];
							double ckMy = (coef_[M_ * (M_ - 1) + 1].y - bSpline2LUT_[5 * nSamplesPerSegment_]
									* (ckMplusy + coef_[k + (M_ - 2) * M_].y))
									/ bSpline2LUT_[3 * nSamplesPerSegment_];
							double ckMz = (coef_[M_ * (M_ - 1) + 1].z - bSpline2LUT_[5 * nSamplesPerSegment_]
									* (ckMplusz + coef_[k + (M_ - 2) * M_].z))
									/ bSpline2LUT_[3 * nSamplesPerSegment_];

							basisFactor = aux * bSpline1LUT_[index_u];
							xPosVal += ckMx * basisFactor;
							yPosVal += ckMy * basisFactor;
							zPosVal += ckMz * basisFactor;
						}
					}
				}
				ddxdvdv_[i][j] = M_ * M_ * xPosVal;
				ddydvdv_[i][j] = M_ * M_ * yPosVal;
				ddzdvdv_[i][j] = M_ * M_ * zPosVal;
			}
		}

		// dd sigma / dudv
		for (int i = 0; i < MR_; i++) {
			for (int j = 0; j <= MR_; j++) {
				xPosVal = 0;
				yPosVal = 0;
				zPosVal = 0;
				for (int l = 1; l <= M_ - 1; l++) {
					index_v = 2 * j + nSamplesPerSegment_ * (3 - 2 * l);
					if (index_v >= 0 && index_v < NR2_) {
						aux = bSpline2DerivativeLUT_[index_v];
						for (int k = 0; k < M_; k++) {
							index_u = (2 * i + nSamplesPerSegment_
									* (3 - 2 * k) + MR2_)
									% MR2_;
							if (index_u >= 0 && index_u < NR2_) {
								basisFactor = aux
										* bSpline1DerivativeLUT_[index_u];
								xPosVal += coef_[k + (l - 1) * M_].x
										* basisFactor;
								yPosVal += coef_[k + (l - 1) * M_].y
										* basisFactor;
								zPosVal += coef_[k + (l - 1) * M_].z
										* basisFactor;
							}
						}
					}
				}

				double scal = 1.0 / (M_ * bSpline2DerivativeLUT_[5 * nSamplesPerSegment_]);

				double NorthV1x = coef_[M_ * (M_ - 1) + 2].x
						- coef_[M_ * (M_ - 1)].x;
				double NorthV1y = coef_[M_ * (M_ - 1) + 2].y
						- coef_[M_ * (M_ - 1)].y;
				double NorthV1z = coef_[M_ * (M_ - 1) + 2].z
						- coef_[M_ * (M_ - 1)].z;

				double NorthV2x = coef_[M_ * (M_ - 1) + 3].x
						- coef_[M_ * (M_ - 1)].x;
				double NorthV2y = coef_[M_ * (M_ - 1) + 3].y
						- coef_[M_ * (M_ - 1)].y;
				double NorthV2z = coef_[M_ * (M_ - 1) + 3].z
						- coef_[M_ * (M_ - 1)].z;

				// l = -1
				index_v = 2 * j + nSamplesPerSegment_ * 5;
				if (index_v >= 0 && index_v < NR2_) {
					aux = bSpline2DerivativeLUT_[index_v];
					for (int k = 0; k < M_; k++) {
						index_u = (2 * i + nSamplesPerSegment_ * (3 - 2 * k) + MR2_)
								% MR2_;
						if (index_u >= 0 && index_u < NR2_) {
							// compute c[k,-1]
							double ckminus1x = (coef_[k].x + scal
									* (cosLUT_[k] * NorthV1x + sinLUT_[k]
											* NorthV2x));
							double ckminus1y = (coef_[k].y + scal
									* (cosLUT_[k] * NorthV1y + sinLUT_[k]
											* NorthV2y));
							double ckminus1z = (coef_[k].z + scal
									* (cosLUT_[k] * NorthV1z + sinLUT_[k]
											* NorthV2z));

							basisFactor = aux * bSpline1DerivativeLUT_[index_u];
							xPosVal += ckminus1x * basisFactor;
							yPosVal += ckminus1y * basisFactor;
							zPosVal += ckminus1z * basisFactor;
						}
					}
				}

				// l = 0
				index_v = 2 * j + nSamplesPerSegment_ * 3;
				if (index_v >= 0 && index_v < NR2_) {
					aux = bSpline2DerivativeLUT_[index_v];
					for (int k = 0; k < M_; k++) {
						index_u = (2 * i + nSamplesPerSegment_ * (3 - 2 * k) + MR2_)
								% MR2_;
						if (index_u >= 0 && index_u < NR2_) {
							// compute c[k,-1]
							double ckminus1x = (coef_[k].x + scal
									* (cosLUT_[k] * NorthV1x + sinLUT_[k]
											* NorthV2x));
							double ckminus1y = (coef_[k].y + scal
									* (cosLUT_[k] * NorthV1y + sinLUT_[k]
											* NorthV2y));
							double ckminus1z = (coef_[k].z + scal
									* (cosLUT_[k] * NorthV1z + sinLUT_[k]
											* NorthV2z));

							// compute c[k,0]
							double ck0x = (coef_[M_ * (M_ - 1)].x - bSpline2LUT_[5 * nSamplesPerSegment_]
									* (ckminus1x + coef_[k].x))
									/ bSpline2LUT_[3 * nSamplesPerSegment_];
							double ck0y = (coef_[M_ * (M_ - 1)].y - bSpline2LUT_[5 * nSamplesPerSegment_]
									* (ckminus1y + coef_[k].y))
									/ bSpline2LUT_[3 * nSamplesPerSegment_];
							double ck0z = (coef_[M_ * (M_ - 1)].z - bSpline2LUT_[5 * nSamplesPerSegment_]
									* (ckminus1z + coef_[k].z))
									/ bSpline2LUT_[3 * nSamplesPerSegment_];

							basisFactor = aux * bSpline1DerivativeLUT_[index_u];
							xPosVal += ck0x * basisFactor;
							yPosVal += ck0y * basisFactor;
							zPosVal += ck0z * basisFactor;
						}
					}
				}

				double SouthV1x = coef_[M_ * (M_ - 1) + 4].x
						- coef_[M_ * (M_ - 1) + 1].x;
				double SouthV1y = coef_[M_ * (M_ - 1) + 4].y
						- coef_[M_ * (M_ - 1) + 1].y;
				double SouthV1z = coef_[M_ * (M_ - 1) + 4].z
						- coef_[M_ * (M_ - 1) + 1].z;

				double SouthV2x = coef_[M_ * (M_ - 1) + 5].x
						- coef_[M_ * (M_ - 1) + 1].x;
				double SouthV2y = coef_[M_ * (M_ - 1) + 5].y
						- coef_[M_ * (M_ - 1) + 1].y;
				double SouthV2z = coef_[M_ * (M_ - 1) + 5].z
						- coef_[M_ * (M_ - 1) + 1].z;

				// l = M+1
				index_v = 2 * j + nSamplesPerSegment_ * (1 - 2 * M_);
				if (index_v >= 0 && index_v < NR2_) {
					aux = bSpline2DerivativeLUT_[index_v];
					for (int k = 0; k < M_; k++) {
						index_u = (2 * i + nSamplesPerSegment_ * (3 - 2 * k) + MR2_)
								% MR2_;
						if (index_u >= 0 && index_u < NR2_) {
							// compute c[k,M+1]
							double ckMplusx = (coef_[k + (M_ - 2) * M_].x + scal
									* (cosLUT_[k] * SouthV1x + sinLUT_[k]
											* SouthV2x));
							double ckMplusy = (coef_[k + (M_ - 2) * M_].y + scal
									* (cosLUT_[k] * SouthV1y + sinLUT_[k]
											* SouthV2y));
							double ckMplusz = (coef_[k + (M_ - 2) * M_].z + scal
									* (cosLUT_[k] * SouthV1z + sinLUT_[k]
											* SouthV2z));

							basisFactor = aux * bSpline1DerivativeLUT_[index_u];
							xPosVal += ckMplusx * basisFactor;
							yPosVal += ckMplusy * basisFactor;
							zPosVal += ckMplusz * basisFactor;
						}
					}
				}

				// l = M
				index_v = 2 * j + nSamplesPerSegment_ * (3 - 2 * M_);
				if (index_v >= 0 && index_v < NR2_) {
					aux = bSpline2DerivativeLUT_[index_v];
					for (int k = 0; k < M_; k++) {
						index_u = (2 * i + nSamplesPerSegment_ * (3 - 2 * k) + MR2_)
								% MR2_;
						if (index_u >= 0 && index_u < NR2_) {
							// compute c[k,M+1]
							double ckMplusx = (coef_[k + (M_ - 2) * M_].x + scal
									* (cosLUT_[k] * SouthV1x + sinLUT_[k]
											* SouthV2x));
							double ckMplusy = (coef_[k + (M_ - 2) * M_].y + scal
									* (cosLUT_[k] * SouthV1y + sinLUT_[k]
											* SouthV2y));
							double ckMplusz = (coef_[k + (M_ - 2) * M_].z + scal
									* (cosLUT_[k] * SouthV1z + sinLUT_[k]
											* SouthV2z));

							// compute c[k,M]
							double ckMx = (coef_[M_ * (M_ - 1) + 1].x - bSpline2LUT_[5 * nSamplesPerSegment_]
									* (ckMplusx + coef_[k + (M_ - 2) * M_].x))
									/ bSpline2LUT_[3 * nSamplesPerSegment_];
							double ckMy = (coef_[M_ * (M_ - 1) + 1].y - bSpline2LUT_[5 * nSamplesPerSegment_]
									* (ckMplusy + coef_[k + (M_ - 2) * M_].y))
									/ bSpline2LUT_[3 * nSamplesPerSegment_];
							double ckMz = (coef_[M_ * (M_ - 1) + 1].z - bSpline2LUT_[5 * nSamplesPerSegment_]
									* (ckMplusz + coef_[k + (M_ - 2) * M_].z))
									/ bSpline2LUT_[3 * nSamplesPerSegment_];

							basisFactor = aux * bSpline1DerivativeLUT_[index_u];
							xPosVal += ckMx * basisFactor;
							yPosVal += ckMy * basisFactor;
							zPosVal += ckMz * basisFactor;
						}
					}
				}
				ddxdudv_[i][j] = M_ * M_ * xPosVal;
				ddydudv_[i][j] = M_ * M_ * yPosVal;
				ddzdudv_[i][j] = M_ * M_ * zPosVal;
			}
		}

		double eulerCharacteristic = 0;
		double delta_uv = 1.0 / (MR_ * MR_);

		for (int u = 0; u < MR_; u++) {
			for (int v = 1; v < MR_; v++) {
				double normN = JACOBIANCONSTANT
						* Math.sqrt(dyWedgedz_[u][v] * dyWedgedz_[u][v]
								+ dzWedgedx_[u][v] * dzWedgedx_[u][v]
								+ dxWedgedy_[u][v] * dxWedgedy_[u][v]);
				if (normN > SQRT_TINY) {
					double E = (dxdu_[u][v] * dxdu_[u][v] + dydu_[u][v]
							* dydu_[u][v] + dzdu_[u][v] * dzdu_[u][v]);
					double F = (dxdu_[u][v] * dxdv_[u][v] + dydu_[u][v]
							* dydv_[u][v] + dzdu_[u][v] * dzdv_[u][v]);
					double G = (dxdv_[u][v] * dxdv_[u][v] + dydv_[u][v]
							* dydv_[u][v] + dzdv_[u][v] * dzdv_[u][v]);
					double detI = E * G - F * F;
					double e = JACOBIANCONSTANT
							* (ddxdudu_[u][v] * dyWedgedz_[u][v]
									+ ddydudu_[u][v] * dzWedgedx_[u][v] + ddzdudu_[u][v]
									* dxWedgedy_[u][v]) / normN;
					double f = JACOBIANCONSTANT
							* (ddxdudv_[u][v] * dyWedgedz_[u][v]
									+ ddydudv_[u][v] * dzWedgedx_[u][v] + ddzdudv_[u][v]
									* dxWedgedy_[u][v]) / normN;
					double g = JACOBIANCONSTANT
							* (ddxdvdv_[u][v] * dyWedgedz_[u][v]
									+ ddydvdv_[u][v] * dzWedgedx_[u][v] + ddzdvdv_[u][v]
									* dxWedgedy_[u][v]) / normN;
					double detII = e * g - f * f;
					eulerCharacteristic += detII / Math.sqrt(Math.abs(detI))
							* delta_uv;
				}
			}
		}
		eulerCharacteristic /= 2.0 * Math.PI;
		return eulerCharacteristic;
	}

	// ----------------------------------------------------------------------------

	public Snake3DNode[] setShape(Sequence initialPositionMask) {
		int zmin_ = Integer.MAX_VALUE;
		int zmax_ = 0;

		boolean[][] mask = SequenceToArrayConverter
				.sequenceToBinaryArray(initialPositionMask);

		int width = initialPositionMask.getSizeX();
		int height = initialPositionMask.getSizeY();
		int depth = initialPositionMask.getSizeZ();

		Polygon[] polygons_ = new Polygon[depth];
		for (int z = 0; z < depth; z++) {
			ContourTracing contourTracking = new ContourTracing(mask[z], width,
					height);
			contourTracking.trace();
			if (contourTracking.getNPoints() != 0) {
				if (z > zmax_) {
					zmax_ = z;
				}
				if (z < zmin_) {
					zmin_ = z;
				}
				polygons_[z] = contourTracking.getTrace();
			} else {
				polygons_[z] = null;
			}
		}

		double phi[][] = new double[MR_ * MR_ + MR_][M_ * M_ + 3 * M_];
		double sigmaX[][] = new double[MR_ * MR_ + MR_][1];
		double sigmaY[][] = new double[MR_ * MR_ + MR_][1];

		for (int j = 0; j <= MR_; j++) {
			double z = (zmax_ - zmin_) * ((double) j + 1) / ((double) MR_ + 1)
					+ zmin_;
			Point2D.Double[] p = Geometry2D.reverse(Geometry2D
					.arcLengthResampling(polygons_[(int) Math.round(z)], MR_));

			for (int i = 0; i < MR_; i++) {
				int ip = i + MR_ * j;
				sigmaX[ip][0] = p[i].x;
				sigmaY[ip][0] = p[i].y;

				for (int l = -1; l <= M_ + 1; l++) {
					for (int k = 0; k < M_; k++) {
						int kp = k + M_ * (l + 1);

						double basisFactor1, basisFactor2;

						int index_i = (2 * i + nSamplesPerSegment_
								* (3 - 2 * k) + MR2_)
								% MR2_;
						if (index_i >= 0 && index_i < NR2_) {
							basisFactor1 = bSpline1LUT_[index_i];
						} else {
							basisFactor1 = 0;
						}

						int index_j = 2 * j + nSamplesPerSegment_ * (3 - 2 * l);
						if (index_j >= 0 && index_j < NR2_) {
							basisFactor2 = bSpline2LUT_[index_j];
						} else {
							basisFactor2 = 0;
						}

						phi[ip][kp] += basisFactor1 * basisFactor2;
					}
				}
			}
		}

		Matrix Phi = new Matrix(phi);

		Matrix SigmaX = new Matrix(sigmaX);
		Matrix CX = Phi.solve(SigmaX);

		Matrix SigmaY = new Matrix(sigmaY);
		Matrix CY = Phi.solve(SigmaY);

		Point2D.Double[][] splineCoefs = new Point2D.Double[M_][M_ + 3];

		for (int l = -1; l <= M_ + 1; l++) {
			for (int k = 0; k < M_; k++) {
				int kp = k + M_ * (l + 1);
				double[][] cx = CX.getArray();
				double[][] cy = CY.getArray();
				splineCoefs[k][l + 1] = new Point2D.Double(cx[kp][0], cy[kp][0]);
			}
		}

		// setting coefficients
		for (int l = 1; l <= M_ - 1; l++) {
			double z = (zmax_ - zmin_) * ((double) l) / (M_) + zmin_;
			for (int k = 0; k < M_; k++) {
				coef_[k + (l - 1) * M_] = new Snake3DNode(
						splineCoefs[k][l + 1].x, splineCoefs[k][l + 1].y, z);
			}
		}

		int index_0 = nSamplesPerSegment_ * 3;
		double factor0 = 0;
		if (index_0 >= 0 && index_0 < NR2_) {
			factor0 = bSpline2LUT_[index_0];
		}
		int index_1 = nSamplesPerSegment_ * 5;
		double factor1 = 0;
		if (index_1 >= 0 && index_1 < NR2_) {
			factor1 = bSpline2LUT_[index_1];
		}
		double factorp1 = 0;
		if (index_1 >= 0 && index_1 < NR2_) {
			factorp1 = bSpline2DerivativeLUT_[index_1];
		}

		double xgN = 0;
		double ygN = 0;
		double xgS = 0;
		double ygS = 0;

		double xV1N = 0;
		double yV1N = 0;
		double xV2N = 0;
		double yV2N = 0;

		double xV1S = 0;
		double yV1S = 0;
		double xV2S = 0;
		double yV2S = 0;

		for (int k = 0; k < M_; k++) {
			xgN += factor0 * splineCoefs[k][1].x + factor1
					* (splineCoefs[k][0].x + splineCoefs[k][2].x);
			ygN += factor0 * splineCoefs[k][1].y + factor1
					* (splineCoefs[k][0].y + splineCoefs[k][2].y);
			xgS += factor0 * splineCoefs[k][M_ + 1].x + factor1
					* (splineCoefs[k][M_].x + splineCoefs[0][M_ + 2].x);
			ygS += factor0 * splineCoefs[k][M_ + 1].y + factor1
					* (splineCoefs[k][M_].y + splineCoefs[0][M_ + 2].y);

			xV1N += (sinLUT_[(k + 1) % M_]
					* (splineCoefs[k][0].x - splineCoefs[k][2].x) - sinLUT_[k]
					* (splineCoefs[(k + 1) % M_][0].x - splineCoefs[(k + 1)
							% M_][2].x))
					/ (sinLUT_[(k + 1) % M_] * cosLUT_[k] - sinLUT_[k]
							* cosLUT_[(k + 1) % M_]);
			yV1N += (sinLUT_[(k + 1) % M_]
					* (splineCoefs[k][0].y - splineCoefs[k][2].y) - sinLUT_[k]
					* (splineCoefs[(k + 1) % M_][0].y - splineCoefs[(k + 1)
							% M_][2].y))
					/ (sinLUT_[(k + 1) % M_] * cosLUT_[k] - sinLUT_[k]
							* cosLUT_[(k + 1) % M_]);

			xV2N += (cosLUT_[(k + 1) % M_]
					* (splineCoefs[k][0].x - splineCoefs[k][2].x) - cosLUT_[k]
					* (splineCoefs[(k + 1) % M_][0].x - splineCoefs[(k + 1)
							% M_][2].x))
					/ (cosLUT_[(k + 1) % M_] * sinLUT_[k] - cosLUT_[k]
							* sinLUT_[(k + 1) % M_]);
			yV2N += (cosLUT_[(k + 1) % M_]
					* (splineCoefs[k][0].y - splineCoefs[k][2].y) - cosLUT_[k]
					* (splineCoefs[(k + 1) % M_][0].y - splineCoefs[(k + 1)
							% M_][2].y))
					/ (cosLUT_[(k + 1) % M_] * sinLUT_[k] - cosLUT_[k]
							* sinLUT_[(k + 1) % M_]);

			xV1S += (sinLUT_[(k + 1) % M_]
					* (splineCoefs[k][M_ + 2].x - splineCoefs[k][M_].x) - sinLUT_[k]
					* (splineCoefs[(k + 1) % M_][M_ + 2].x - splineCoefs[(k + 1)
							% M_][M_].x))
					/ (sinLUT_[(k + 1) % M_] * cosLUT_[k] - sinLUT_[k]
							* cosLUT_[(k + 1) % M_]);
			yV1S += (sinLUT_[(k + 1) % M_]
					* (splineCoefs[k][M_ + 2].y - splineCoefs[k][M_].y) - sinLUT_[k]
					* (splineCoefs[(k + 1) % M_][M_ + 2].y - splineCoefs[(k + 1)
							% M_][M_].y))
					/ (sinLUT_[(k + 1) % M_] * cosLUT_[k] - sinLUT_[k]
							* cosLUT_[(k + 1) % M_]);

			xV2S += (cosLUT_[(k + 1) % M_]
					* (splineCoefs[k][M_ + 2].x - splineCoefs[k][M_].x) - cosLUT_[k]
					* (splineCoefs[(k + 1) % M_][M_ + 2].x - splineCoefs[(k + 1)
							% M_][M_].x))
					/ (cosLUT_[(k + 1) % M_] * sinLUT_[k] - cosLUT_[k]
							* sinLUT_[(k + 1) % M_]);
			yV2S += (cosLUT_[(k + 1) % M_]
					* (splineCoefs[k][M_ + 2].y - splineCoefs[k][M_].y) - cosLUT_[k]
					* (splineCoefs[(k + 1) % M_][M_ + 2].y - splineCoefs[(k + 1)
							% M_][M_].y))
					/ (cosLUT_[(k + 1) % M_] * sinLUT_[k] - cosLUT_[k]
							* sinLUT_[(k + 1) % M_]);
		}

		xgN /= M_;
		ygN /= M_;
		xgS /= M_;
		ygS /= M_;

		xV1N *= factorp1;
		yV1N *= factorp1;
		xV1S *= factorp1;
		yV1S *= factorp1;

		xV2N *= factorp1;
		yV2N *= factorp1;
		xV2S *= factorp1;
		yV2S *= factorp1;

		coef_[M_ * (M_ - 1)] = new Snake3DNode(xgN, ygN, zmin_);
		coef_[M_ * (M_ - 1) + 1] = new Snake3DNode(xgS, ygS, zmax_);

		coef_[M_ * (M_ - 1) + 2] = new Snake3DNode(xgN + xV1N, ygN + yV1N,
				zmin_);
		coef_[M_ * (M_ - 1) + 3] = new Snake3DNode(xgN + xV2N, ygN + yV2N,
				zmin_);

		coef_[M_ * (M_ - 1) + 4] = new Snake3DNode(xgS + xV1S, ygS + yV1S,
				zmax_);
		coef_[M_ * (M_ - 1) + 5] = new Snake3DNode(xgS + xV2S, ygS + yV2S,
				zmax_);
		return coef_;
	}
}
