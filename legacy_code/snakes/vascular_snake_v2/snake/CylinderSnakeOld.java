/*******************************************************************************
 * Copyright (c) 2012 Biomedical Image Group (BIG), EPFL, Switzerland.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Public License v3.0
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/gpl.html
 * 
 * Contributors:
 *     Ricard Delgado-Gonzalo (ricard.delgado@gmail.com)
 *     Nicolas Chenouard (nicolas.chenouard@gmail.com)
 *     Philippe Thévenaz (philippe.thevenaz@epfl.ch)
 *     Emrah Bostan (emrah.bostan@gmail.com)
 *     Ulugbek S. Kamilov (kamilov@gmail.com)
 *     Ramtin Madani (ramtin_madani@yahoo.com)
 *     Masih Nilchian (masih_n85@yahoo.com)
 *     Cédric Vonesch (cedric.vonesch@epfl.ch)
 *     Virginie Uhlmann (virginie.uhlmann@epfl.ch)
 ******************************************************************************/
package plugins.big.vascular.snake;

import icy.gui.frame.progress.AnnounceFrame;
import icy.gui.frame.progress.ProgressFrame;
import icy.image.IcyBufferedImage;
import icy.main.Icy;
import icy.sequence.Sequence;
import icy.type.DataType;
import icy.type.rectangle.Rectangle3D;

import java.awt.Color;
import java.awt.Polygon;
import java.awt.geom.Point2D;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import javax.vecmath.Point3d;

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
import plugins.big.vascular.utils.Const;
import vtk.vtkPoints;
import vtk.vtkPolyData;
import vtk.vtkQuad;
import Jama.Matrix;

/**
 * Three-dimensional exponential spline snake (E-Snake).
 * 
 * @version October 31, 2012
 * 
 * @author Ricard Delgado-Gonzalo (ricard.delgado@gmail.com)
 * @author Nicolas Chenouard (nicolas.chenouard@gmail.com)
 */
public class CylinderSnakeOld implements Snake3D {
	
	private static long startTime = 0;
	private static long endTime = 0;
	boolean started = false;

	/** Basis function of the curve. */
	//private BSplineBasisType basisFunction_ = BSplineBasisType.ESPLINE3;
	private BSplineBasis.BSplineBasisType basisFunction_ = BSplineBasisType.ESPLINE3;

	/** Snake defining nodes. */
	private Snake3DNode[] coef_ = null;

	/** LUT with the samples of the B-spline basis function. */
	private double[] splineMFunc_ = null;

	/** LUT with the samples of the B-spline basis function. */
	private double[] spline2MFunc_ = null;

	/** LUT with the samples of the derivative of the B-spline basis function. */
	private double[] splinePrimeMFunc_ = null;

	/** LUT with the samples of the derivative of the B-spline basis function. */
	private double[] splinePrime2MFunc_ = null;

	/**
	 * LUT with the samples of the second derivative of the B-spline basis
	 * function.
	 */
	private double[] splinePrimePrimeMFunc_ = null;
	/**
	 * LUT with the samples of the second derivative of the B-spline basis
	 * function.
	 */
	private double[] splinePrimePrime2MFunc_ = null;

	/**
	 * LUT with the B-spline coefficients for the perfect reproduction of a
	 * sine.
	 */
	private double[] cs_ = null;
	/**
	 * LUT with the B-spline coefficients for the perfect reproduction of a
	 * cosine.
	 */
	private double[] cc_ = null;

	/** LUT with the samples of the x coordinates of the snake contour. */
	private double[][] xSnake_ = null;
	/** LUT with the samples of the y coordinates of the snake contour. */
	private double[][] ySnake_ = null;
	/** LUT with the samples of the z coordinates of the snake contour. */
	private double[][] zSnake_ = null;
	
	private double[][] xSnakeUpdate_ = null;
	private double[][] ySnakeUpdate_ = null;
	private double[][] zSnakeUpdate_ = null;
	
	private double[][] xSnakeCopy_ = null;
	private double[][] ySnakeCopy_ = null;
	private double[][] zSnakeCopy_ = null;

	/** LUT with the samples of the x coordinates of the snake contour. */
	private double[][] xShell_ = null;
	/** LUT with the samples of the y coordinates of the snake contour. */
	private double[][] yShell_ = null;
	/** LUT with the samples of the z coordinates of the snake contour. */
	private double[][] zShell_ = null;

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
	 * LUT with the samples of the first component of the first vector of the
	 * tangent bundle.
	 */
	private double[][] dxdv_ = null;
	/**
	 * LUT with the samples of the second component of the first vector of the
	 * tangent bundle.
	 */
	private double[][] dydv_ = null;
	/**
	 * LUT with the samples of the third component of the first vector of the
	 * tangent bundle.
	 */
	private double[][] dzdv_ = null;

	private double[][] ddxdudu_ = null;
	private double[][] ddxdudv_ = null;
	private double[][] ddxdvdv_ = null;

	private double[][] ddydudu_ = null;
	private double[][] ddydudv_ = null;
	private double[][] ddydvdv_ = null;

	private double[][] ddzdudu_ = null;
	private double[][] ddzdudv_ = null;
	private double[][] ddzdvdv_ = null;

	/** LUT with the samples of the first component of the normal vector. */
	private double[][] dyWedgedz_ = null;
	/** LUT with the samples of the second component of the normal vector. */
	private double[][] dzWedgedx_ = null;
	/** LUT with the samples of the third component of the normal vector. */
	private double[][] dxWedgedy_ = null;

	/** Container of the LUTs. */
	private ImageLUTContainer imageLUTs_ = null;

	/** Parameter that determines the number of degrees of freedom of the snake. */
	private int M_ = 0;

	/**
	 * Number of iterations left when the <code>immortal_</code> is
	 * <code>false</code>.
	 */
	private int life_ = 0;

	/**
	 * Maximum number of iterations allowed when the <code>immortal_</code> is
	 * <code>false</code>.
	 */
	private int maxLife_ = 0;

	/** Volume enclosed by the snake. */
	private double volume_ = 0;

	/** Area of the snake surface. */
	private double surfaceArea_ = 0;

	/** Indicates the type of features to detect (bright or dark) **/
	private CylinderSnakeTargetType detectType_ = CylinderSnakeTargetType.BRIGHT;

	/** PI/M. */
	private double PIM_ = 0;

	/** 2*PI/M. */
	private double PI2M_ = 0;

	/** Sampling rate at which the contours are discretized. */
	private  int DISCRETIZATIONSAMPLINGRATE = 38;//-->40;//25;//35; //25;//39;//10; // 40 is the value for MP2RAGE

	/** 2*N*DISCRETIZATIONSAMPLINGRATE. */
	private int NR2_ = 0;

	/** M*DISCRETIZATIONSAMPLINGRATE. */
	private int MR_ = 0;

	/** 2*M*DISCRETIZATIONSAMPLINGRATE. */
	private int MR2_ = 0;

	/**
	 * If <code>true</code> indicates that the snake is able to keep being
	 * optimized.
	 */
	private boolean alive_ = true;

	/**
	 * If <code>false</code> indicates that the snake takes an ill conditioned
	 * shape.
	 */
	private boolean valid_ = true;

	/**
	 * If <code>true</code> indicates that the snake will keep iterating till
	 * the optimizer decides so.
	 */
	private boolean immortal_ = true;
	/**
	 * If <code>true</code>, the snake has gone through the initialization
	 * process.
	 */
	private boolean isInitialized_ = false;

	/** Trade-off factor between contour and regions energies. */
	private double alpha_ = 0;

	/** Weight factor for the reparameterization energy. */

	private double gamma_ = 0;

	/** Snake energy used during the optimization process. */
	private CylinderSnakeEnergyType energyType_ = CylinderSnakeEnergyType.REGION;

	/** Normalization factor for the Jacobian. */
	private static double JACOBIANCONSTANT = 10000;

	/** Cube root of two. */
	private static double CBRT = Math.cbrt(2.0);

	/** Square root of the <code>float</code> machine precision. */
	private static final double SQRT_TINY = Math.sqrt((double) Float
			.intBitsToFloat((int) 0x33FFFFFF));

	// ============================================================================
	// PUBLIC METHODS

	/** Default constructor. */
	public CylinderSnakeOld(ImageLUTContainer imageLUTs,
			CylinderSnakeParameters parameters) {
		if (imageLUTs == null) {
			System.err.println("Image not properly loaded.");
			return;
		}
		imageLUTs_ = imageLUTs;
		if (parameters == null) {
			System.err.println("E-Snake parameters not properly loaded.");
			return;
		}
		M_ = parameters.getM();
		if (M_ < 3) {
			throw new IllegalArgumentException(
					"Error: M needs to be equal or larger than 3.");
		}
		
		DISCRETIZATIONSAMPLINGRATE = (int) Math.round((double) DISCRETIZATIONSAMPLINGRATE
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
	
	// 	1. image energy:
	// 			- contour energy (in thesis called edge energy)
	//			- region energy
	//
	//	 2. internal energy:
	//			_ stiffness energ = reparametrization energy (in thesis called curvilinear reparametrization energy)
	// 			- 
	
	
	@Override
	public double energy() {
		if (!immortal_) {				
			
			if (!started){
				started = true;
				startTime = System.currentTimeMillis();
			}
			if (life_ == 1){
				endTime = System.currentTimeMillis();				
				
				long elapsedTimeMillis = endTime-startTime;
				float elapsedTimeSec = elapsedTimeMillis/(60*1000F);
				System.out.println("time snake  = " + elapsedTimeSec);
			}
			
			//System.out.println("life_ = " + life_ + ", gamma_ = " + gamma_);
			
			/*
			if (life_ == 20000)
				System.out.println("20'000 iterations left, gamma = " + gamma_ + ", surface = " + surfaceArea_);
			if (life_ == 10000)
				System.out.println("10'000 iterations left, gamma = " + gamma_ + ", surface = " + surfaceArea_);
			if (life_ == 5000)
				System.out.println("5'000 iterations left, gamma = " + gamma_ + ", surface = " + surfaceArea_);
			*/
			
			life_--;
			if (life_ <= 0)
				alive_ = false;
		}	
		
		/*
		//
		// firts 11'000 iterations should be done on the distance map
		
		if (life_ == 19900)
			System.out.println("life = " + life_ + ", gamma_ = " + gamma_);
		
		if (life_ == 9900)
			System.out.println("life = " + life_ + ", gamma_ = " + gamma_);
		*/
		
		
		
		int sign = 1; //sign of Energy E
		if (detectType_ == CylinderSnakeTargetType.BRAINMP2RAGE){
			if (life_ >= 0){
				imageLUTs_ = Const.LUT_MP2RAGE;
				//gamma_ = 0;//0.5;//70000000;
				sign = 1;
			}
			else{				
				
				imageLUTs_ = Const.LUT_Binary;
				//gamma_ = 30000000;
				sign = 1;
				//System.out.println("EN = " + imageLUTs_.getPreintegratedFilteredImageDataArray()[30][30]);
			}
		}
		//
		

		double E;
		switch (energyType_) {	
		
		case CONTOUR:
			if (detectType_ == CylinderSnakeTargetType.BRAINMP2RAGE)
				E = computedDistanceMapEnergy();
			else
				E = computeContourEnergy(); 	// contour energy			
			break;
		case REGION:
			E = computeRegionEnergy();		// region energy
			break;
		case MIXTURE:	
			if (detectType_ == CylinderSnakeTargetType.BRAINMP2RAGE){
				
				E = computeContourEnergy(); 
				
				/*
				if (life_ > 5000)
					E = computeContourEnergy(); 
				else
					E = alpha_ * computeContourEnergy() + (1 - alpha_) * computeRegionEnergy();
				*/
			}
			else			
				E = alpha_ * computeContourEnergy() + (1 - alpha_) * computeRegionEnergy(); // original +
			break;
		default:
			E = Double.MAX_VALUE;
			break;
		}

		double boundaryEnergy = computeImageEdgeEnergy();
		double stiffnessEnergy = computeReparameterizationEnergy();		
		
		/*
		System.out.println("it = " + life_ + ", E_norm = " + (E/surfaceArea_) + ", gamma = " + gamma_ + ", boundaryE = " + boundaryEnergy 
				+", E=" + E + ", surfaceArea = " + surfaceArea_ );
		*/
		
		if (life_ % 5000 == 0){
			System.out.println("it = " + life_ + ", E_norm = " + E + ", gamma = " + gamma_ + ", surfaceArea_ =" + surfaceArea_);
			
			/*
			System.out.println("it = " + life_ + ", E_norm = " + (E/surfaceArea_) +
					", E=" + E + ", surfaceArea = " + surfaceArea_ + ", gamma = " + gamma_ + ", boundaryE = " + boundaryEnergy);
					*/
		}

		//System.out.println("E = " + E);
		if (detectType_ == CylinderSnakeTargetType.DARK) {		
			return -E + boundaryEnergy + gamma_ * stiffnessEnergy;
		} else if (detectType_ == CylinderSnakeTargetType.BRIGHT) {
			return E + boundaryEnergy + gamma_ * stiffnessEnergy;
			
		}else if (detectType_ == CylinderSnakeTargetType.BRAINMP2RAGE) {				
			//return sign*E/surfaceArea_ + boundaryEnergy + gamma_ * stiffnessEnergy; // -
			//System.out.println("E = " + E);
			return E + boundaryEnergy;
		}
		
		else {
			return Double.MAX_VALUE;
		}
	}

	// ----------------------------------------------------------------------------

	/** Initializes the snake class. */
	@Override
	public void initialize() {
		switch (basisFunction_) {
		case ESPLINE3:
			NR2_ = 2 * BSplineBasis.ESPLINE3SUPPORT
					* DISCRETIZATIONSAMPLINGRATE;
			break;
		case ESPLINE4:
			NR2_ = 2 * BSplineBasis.ESPLINE4SUPPORT
					* DISCRETIZATIONSAMPLINGRATE;
			break;
		case LINEARBSPLINE:
			NR2_ = 2 * BSplineBasis.LINEARBSPLINESUPPORT
					* DISCRETIZATIONSAMPLINGRATE;
			break;
		case QUADRATICBSPLINE:
			NR2_ = 2 * BSplineBasis.QUADRATICBSPLINESUPPORT
					* DISCRETIZATIONSAMPLINGRATE;
			break;
		case CUBICBSPLINE:
			NR2_ = 2 * BSplineBasis.CUBICBSPLINESUPPORT
					* DISCRETIZATIONSAMPLINGRATE;
			break;
		case MSPLINE:
			NR2_ = 2 * BSplineBasis.MSPLINESUPPORT * DISCRETIZATIONSAMPLINGRATE;
			break;
		}

		MR_ = M_ * DISCRETIZATIONSAMPLINGRATE;
		MR2_ = 2 * MR_;

		PIM_ = Math.PI / M_;
		PI2M_ = 2 * PIM_;

		xSnake_ = new double[MR_][MR_ + 1];
		ySnake_ = new double[MR_][MR_ + 1];
		zSnake_ = new double[MR_][MR_ + 1];
		xShell_ = new double[MR_][MR_ + 1];
		yShell_ = new double[MR_][MR_ + 1];
		zShell_ = new double[MR_][MR_ + 1];

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

		life_ = maxLife_;
		buildLUTs();
		initializeDefaultShape();

		updateSnakeSkin();
		updateSnakeTangentBundle();
		updateArea();
		
		//updateVolume(); // D.S.
		
		if (energyType_ == CylinderSnakeEnergyType.REGION
				|| energyType_ == CylinderSnakeEnergyType.MIXTURE)
			updateVolume();
		if (energyType_ == CylinderSnakeEnergyType.REGION
				|| energyType_ == CylinderSnakeEnergyType.MIXTURE)
			updateSnakeShell();
		isInitialized_ = true;	
		
		//System.out.println("init = " + imageLUTs_.getPreintegratedFilteredImageDataArray()[30][30]);
		
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

	/**
	 * Implements a strategy to restart the snake shape from an invalid
	 * configuration.
	 */
	public void recoverFromInvalid() {
		// TODO Design a strategy to recover from invalid configuration.
	}

	// ----------------------------------------------------------------------------

	/** Saves the snake-defining parameters in an XML file.
	@Override
	public void saveToXML(Element node) {
		// TODO Auto-generated method stub

	}
	*/

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
				|| energyType_ == CylinderSnakeEnergyType.MIXTURE)
			updateVolume();
		if (energyType_ == CylinderSnakeEnergyType.REGION
				|| energyType_ == CylinderSnakeEnergyType.MIXTURE)
			updateSnakeShell();
	}

	// ----------------------------------------------------------------------------

	/** Sets the E-Snake execution parameters. */
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
			// TODO Enable to change the number of control points on the fly..
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
		// TODO
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
				points[v][0] = xSnake_[i][v];
				points[v][1] = ySnake_[i][v];
				points[v][2] = zSnake_[i][v];
			}
		} else if (i >= MR_ && i < 2 * MR_ + 1) {
			skin = new Snake3DScale(MR_, Color.RED, true);
			double[][] points = skin.getCoordinates();
			for (int u = 0; u < MR_; u++) {
				points[u][0] = xSnake_[u][i - MR_];
				points[u][1] = ySnake_[u][i - MR_];
				points[u][2] = zSnake_[u][i - MR_];
			}
		}
		return skin;
	}
	
	public Snake3DScale getScaleFinal(int i){
		Snake3DScale skin = null;
		if (i >= 0 && i < MR_) {
			skin = new Snake3DScale(MR_ + 1, Color.RED, false);
			double[][] points = skin.getCoordinates();
			for (int v = 0; v <= MR_; v++) {
				points[v][0] = xSnakeUpdate_[i][v];
				points[v][1] = ySnakeUpdate_[i][v];
				points[v][2] = zSnakeUpdate_[i][v];
			}
		} else if (i >= MR_ && i < 2 * MR_ + 1) {
			skin = new Snake3DScale(MR_, Color.RED, true);
			double[][] points = skin.getCoordinates();
			for (int u = 0; u < MR_; u++) {
				points[u][0] = xSnakeUpdate_[u][i - MR_];
				points[u][1] = ySnakeUpdate_[u][i - MR_];
				points[u][2] = zSnakeUpdate_[u][i - MR_];
			}
		}
		return skin;
	}
	

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
	
	public Snake3DScale[] getScalesFinal() {
		int numScales = getNumScales();
		Snake3DScale[] skin = new Snake3DScale[numScales];
		for (int i = 0; i < numScales; i++) {
			skin[i] = getScaleFinal(i);
		}
		return skin;
	}

	// ----------------------------------------------------------------------------

	/** Retrieves the area of the surface of the snake. */
	public double getArea() {
		return Math.abs(surfaceArea_);
	}

	// ----------------------------------------------------------------------------

	/** Returns a binary image stack representing the surface of the snake. */
	@Override
	public Sequence getBinaryMask() {

		vtkPolyData mesh = getQuadMesh();

		ProgressFrame pFrame = new ProgressFrame("Computing binary mask");
		pFrame.setLength(imageLUTs_.getImageDepth());

		double[] bounds = mesh.GetBounds();

		int z0 = Math.max((int) Math.floor(bounds[4]), 0);
		int z1 = Math.min((int) Math.ceil(bounds[5]),
				imageLUTs_.getImageDepth() - 1);
		
		//RasterizerSurface[] rays = new RasterizerSurface[z1 - z0 + 1];
		
		/*
		for (int z = z0; z <= z1; z++) {
			rays[z - z0] = new RasterizerSurface(imageLUTs_.getImageWidth(),
					imageLUTs_.getImageHeight(), pFrame, z, mesh, bounds);
		}
		*/		
		
		
		// oldMethod that corresponds to Binarizer now:
		// just replace Rasterizer by Binarizer3D
		
		Binarizer3D[] rays = new Binarizer3D[z1 - z0 + 1];
		for (int z = z0; z <= z1; z++) {
			rays[z - z0] = new Binarizer3D(imageLUTs_.getImageWidth(),
					imageLUTs_.getImageHeight(), pFrame, z, mesh, bounds);
		}	
		
		

		ExecutorService executor = Executors.newFixedThreadPool(8);
		for (int z = 0; z < rays.length; z++) {
			Runnable task = rays[z];
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
						imageLUTs_.getImageHeight(), 1, DataType.DOUBLE));
			}
		}
		pFrame.close();
		// seq.addPainter(new PolyMeshPainter(mesh));
		Const.binaryMask = seq;
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
	
	public void copyCoef(){
		Const.coef_ = coef_;
		updateVolume();
		Const.snakeVolume = getVolume();
		System.out.println("snake coefs copied to Const");
	}
	
	private Point3d crossProd(Point3d p1, Point3d p2){
		double a = p1.y*p2.z - p1.z*p2.y;
		double b = p1.z*p2.x - p1.x*p2.z;
		double c = p1.x*p2.y - p1.y*p2.x;
		
		return new Point3d(a, b, c);
	}
	
	private Point3d minus(Point3d p1, Point3d p2){
		return new Point3d(p1.x-p2.x, p1.y-p2.y, p1.z-p2.z);
	}
	
	private Point3d normal(Point3d n1, Point3d n2, Point3d n3, Point3d n4){
		double a = n1.x + n2.x + n3.x + n4.x;
		double b = n1.y + n2.y + n3.y + n4.y;
		double c = n1.z + n2.z + n3.z + n4.z;		
		
		return new Point3d(a, b, c);
	}
	
	private Point3d meanPoint(Point3d p1, Point3d p2, Point3d p3, Point3d p4){
		double a = 0.25*(p1.x + p2.x + p3.x + p4.x);
		double b = 0.25*(p1.y + p2.y + p3.y + p4.y);
		double c = 0.25*(p1.z + p2.z + p3.z + p4.z);	
		
		return new Point3d(a, b, c);
	}
	
	private Point3d calcSN(Point3d s, Point3d n){
		return new Point3d(s.x+n.x, s.y+n.y, s.z+n.z);		
	}
	
	// move the points of the mesh to the closest surface element of the hyst-strip image
	public void finalize(){		
		
		dilateXYZ(1.01);
		updateSnakeSkin();
		
		/*
		Sequence boundarySeq = Icy.getMainInterface().getSequences("Step 4: Hysteresis Thresholding").get(0);
		double[][][] boundaryVol = (new ImageVolume(boundarySeq)).volume;
		
		int nx = boundarySeq.getSizeX();
		int ny = boundarySeq.getSizeY();
		int nz = boundarySeq.getSizeZ();
		
		xSnakeCopy_ = xSnake_;
		ySnakeCopy_ = ySnake_;
		zSnakeCopy_ = zSnake_;
		
		
		xSnakeUpdate_ = xSnakeCopy_;
		ySnakeUpdate_ = ySnakeCopy_;
		zSnakeUpdate_ = zSnakeCopy_;
		
		Point3d p, p10, p_10, p0_1, p01;
		Point3d n, nSE, nNW, nWS, nEN;
		Point3d meanP, s, sn, st; 
		
		boolean[][] stop = new boolean[MR_][MR_ + 1];
		int[][] count = new int[MR_][MR_ + 1];
		
		for (int u = 0; u < MR_; u++) 
		for (int v = 0; v <= MR_; v++)
			stop[u][v] = false;
		
		double MU = 0.02;		
		double mu = MU;
		
		double muS = 0.5*MU;
		
		int nIts = 4000;	
		
		double theta;
		
		
		for (int i=1; i <= nIts; i++){
		
			xSnakeCopy_ = xSnakeUpdate_;
			ySnakeCopy_ = ySnakeUpdate_;
			zSnakeCopy_ = zSnakeUpdate_;
			
			for (int u = 0; u < MR_; u++) 
			for (int v = 0; v <= MR_; v++)
			if (!stop[u][v]){	
			
				p = new Point3d(xSnake_[u][v], ySnake_[u][v], zSnake_[u][v]);
				
				int u1 = u+1;
				int u_1 = u-1;
				int v1 = v+1;
				int v_1 = v-1;
			
				if (u>0 && u<MR_-1 && v>0 && v<MR_-1){
					p10 = new Point3d(xSnakeCopy_[u+1][v], ySnakeCopy_[u+1][v], zSnakeCopy_[u+1][v]);
					p_10 = new Point3d(xSnakeCopy_[u-1][v], ySnakeCopy_[u-1][v], zSnakeCopy_[u-1][v]);
					p0_1 = new Point3d(xSnakeCopy_[u][v-1], ySnakeCopy_[u][v-1], zSnakeCopy_[u][v-1]);
					p01 = new Point3d(xSnakeCopy_[u][v+1], ySnakeCopy_[u][v+1], zSnakeCopy_[u][v+1]);
				}
				else{
					u_1 = u-1;
					if (u == 0)
						u_1 = MR_-1;
				
					u1 = u+1;
					if (u == MR_-1)
						u1 = 0;
				
					v_1 = v-1;
					if (v == 0)
						v_1 = MR_-1;
				
					v1 = v+1;
					if (v == MR_)
						v1 = 0;
				
					p10 = new Point3d(xSnakeCopy_[u1][v], ySnakeCopy_[u1][v], zSnakeCopy_[u1][v]);
					p_10 = new Point3d(xSnakeCopy_[u_1][v], ySnakeCopy_[u_1][v], zSnakeCopy_[u_1][v]);
					p0_1 = new Point3d(xSnakeCopy_[u][v_1], ySnakeCopy_[u][v_1], zSnakeCopy_[u][v_1]);					
					p01 = new Point3d(xSnakeCopy_[u][v1], ySnakeCopy_[u][v1], zSnakeCopy_[u][v1]);				
				}
			
				nSE = new Point3d(crossProd(minus(p,p01), minus(p,p10)));
				nEN = new Point3d(crossProd(minus(p,p10), minus(p,p0_1)));
				nNW = new Point3d(crossProd(minus(p,p0_1), minus(p,p_10)));
				nWS = new Point3d(crossProd(minus(p,p_10), minus(p,p01)));
			
				n = normal(nSE, nEN, nNW, nWS);	
				double normN = Math.sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
				Point3d nNormal = new Point3d(n.x/normN, n.y/normN, n.z/normN);				
				
				meanP = meanPoint(p10, p_10, p0_1, p01);
				s = minus(p, meanP);
				
				double normS = Math.sqrt(s.x*s.x + s.y*s.y + s.z*s.z);
				Point3d sNormal = new Point3d(s.x/normS, s.y/normS, s.z/normS);
				
				// local curvature
				theta = Math.acos((nNormal.x*sNormal.x + nNormal.y*sNormal.y + nNormal.z*sNormal.z));				
				
				double angle = 0.2*Math.PI; //0.2
				if (theta > angle){
					mu = 0.01*mu;
				}	
				
				muS = 0.3*mu;
				
				// try
				double snScalarP = (s.x*2*n.x + s.y*2*n.y + s.z*2*n.z);
				sn = new Point3d(snScalarP*n.x, snScalarP*n.y, snScalarP*n.z);
				//double normSn = Math.sqrt(sn.x*sn.x + sn.y*sn.y + sn.z*sn.z);
				//Point3d snNormal = new Point3d(sn.x/normSn, sn.y/normSn, sn.z/normSn);
				
				st = minus(s, sn);
				double normSt = Math.sqrt(st.x*st.x + st.y*st.y + st.z*st.z);
				Point3d stNormal = new Point3d(st.x/normSt, st.y/normSt, st.z/normSt);
				// end try
				
				xSnakeUpdate_[u][v] = xSnakeCopy_[u][v] + mu*nNormal.x + muS*stNormal.x;// - muS*sNormal.x;
				ySnakeUpdate_[u][v] = ySnakeCopy_[u][v] + mu*nNormal.y + muS*stNormal.y;// - muS*sNormal.y;
				zSnakeUpdate_[u][v] = zSnakeCopy_[u][v] + mu*nNormal.z + muS*stNormal.z;// - muS*sNormal.z;
				
				mu = MU;				
				
				double interP = Const.getInterpolatedPixel(xSnakeUpdate_[u][v], ySnakeUpdate_[u][v], zSnakeUpdate_[u][v],
						nx, ny, nz, boundaryVol);
				
				if (interP != 0 || count[u][v] >= 200)
					stop[u][v] = true;
				
				if (stop[u1][v] || stop[u_1][v] || stop[u][v_1] || stop[u][v1])
					count[u][v]++;
				
			}						
		}	
		*/
		
		System.out.println("finalize SphereSnake done.");	
		
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
				points.InsertPoint(v + u * (MR_ + 1), xSnake_[u][v],
						ySnake_[u][v], zSnake_[u][v]);
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
	// INTERACTION METHODS

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
	
	public void dilateXYZ (double d){
		Point3d g = getCentroid();
		for (int i = 0; i < coef_.length; i++) {
			double x = coef_[i].x - g.x;
			double y = coef_[i].y - g.y;
			double z = coef_[i].z - g.z;
			coef_[i].x = d*x + g.x;
			coef_[i].y = d * y + g.y;
			coef_[i].z = d*z + g.z;
		}
	}

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
		splineMFunc_ = new double[NR2_];
		spline2MFunc_ = new double[NR2_];
		splinePrimeMFunc_ = new double[NR2_];
		splinePrime2MFunc_ = new double[NR2_];
		splinePrimePrimeMFunc_ = new double[NR2_];
		splinePrimePrime2MFunc_ = new double[NR2_];
		cc_ = new double[M_];
		cs_ = new double[M_];

		double currentVal = 0;
		double R2 = 2.0 * DISCRETIZATIONSAMPLINGRATE;
		for (int i = 0; i < NR2_; i++) {
			currentVal = i / R2;
			splineMFunc_[i] = BSplineBasis.ESpline3(currentVal, PI2M_);
			spline2MFunc_[i] = BSplineBasis.ESpline3(currentVal, PIM_);
			splinePrimeMFunc_[i] = BSplineBasis.ESpline3_Prime(currentVal,
					PI2M_);
			splinePrime2MFunc_[i] = BSplineBasis.ESpline3_Prime(currentVal,
					PIM_);
			splinePrimePrimeMFunc_[i] = BSplineBasis.ESpline3_PrimePrime(
					currentVal, PI2M_);
			splinePrimePrime2MFunc_[i] = BSplineBasis.ESpline3_PrimePrime(
					currentVal, PIM_);
		}
		double aux = 2.0 * (1.0 - Math.cos(PI2M_))
				/ (Math.cos(PIM_) - Math.cos(PI2M_ + PIM_));
		for (int i = 0; i < M_; i++) {
			double theta = PI2M_ * i;
			cc_[i] = aux * Math.cos(theta);
			cs_[i] = aux * Math.sin(theta);
		}
	}
	
	
	private double[] getX0Y0RxRyRz(){	
		
		if (Const.segmentBrainMP2RAGE){
			System.out.println("MP2RAGE");
			imageLUTs_.brainSequence = Icy.getMainInterface().getSequences("binaryImage").get(0);
		}
			
		int nx = imageLUTs_.brainSequence.getSizeX();
		int ny = imageLUTs_.brainSequence.getSizeY();
		int nz = imageLUTs_.brainSequence.getSizeZ();
		
		int minX = nx - 1;
		int minY = ny - 1;
		int maxX = 0;
		int maxY = 0;
		
		int z0 = nz/2 - 1;
		
		double val;
		for(int x=0 ; x<nx; x++)
		for(int y=0; y<ny; y++){		    
		    val = imageLUTs_.brainSequence.getData(0, z0, 0, y, x);
		   // if (val != 0){
		    if (val > 0){
		    	if (x < minX)
		    		minX = x;
		    	if (x > maxX)
		    		maxX = x;
		    	if (y < minY)
		    		minY = y;
		    	if (y > maxY)
		    		maxY = y;
		    }		    
		}	
		
		int minZ = nz -1;
		int maxZ = 0;
		
		int y0 = ny/2 - 1;
		for(int x=0 ; x<nx; x++)
		for(int z=0; z<nz; z++){		    
			val = imageLUTs_.brainSequence.getData(0, z, 0, y0, x);
			 if (val > 0){			 
			    if (z < minZ)
			    	minZ = z;
			    if (z > maxZ)
			    	maxZ = z;
			    }		    
			}
		
		
		double[] x0y0RxRyRz = {(0.52*(minX + maxX)), (0.4*(minY+maxY)), 0.4*(maxX-minX), 0.25*(maxY-minY), 0.4*(maxZ-minZ)};
		
		return x0y0RxRyRz;
	}

	// ----------------------------------------------------------------------------

	/** Initializes the snake control points with a predefined shape. */
	private void initializeDefaultShape() {
		coef_ = new Snake3DNode[M_ * (M_ - 1) + 6];
		

		double r = Math.min(
				Math.min(imageLUTs_.getImageWidth() / 5.0,
						imageLUTs_.getImageHeight() / 5.0),
				imageLUTs_.getImageDepth() / 5.0);
		
		
		
		double x0 = getX0Y0RxRyRz()[0];
		double y0 = getX0Y0RxRyRz()[1];
		double z0 = imageLUTs_.getImageDepth() / 2.0;		
		
		double rX0 = getX0Y0RxRyRz()[2];
		double rY0 = getX0Y0RxRyRz()[3];
		double rZ0 = getX0Y0RxRyRz()[4];
		
		/*
		double x0 = imageLUTs_.getImageWidth() / 2.5; // 108 DP_1, CG_1, DR_1
		double y0 = imageLUTs_.getImageHeight() / 2.7; //imageLUTs_.getImageHeight() / 2.4;GB_1, MB_1  //imageLUTs_.getImageHeight() / 3; DS_1 //imageLUTs_.getImageHeight() / 2.5 DR_1; // 90 DP_1, CG_1
		double z0 = imageLUTs_.getImageDepth() / 2.0; // 80 DP_1
		*/

		double scale2M = 2.0 * (1.0 - Math.cos(PIM_))
				/ (Math.cos(PIM_ / 2) - Math.cos(3.0 * PIM_ / 2));

		for (int k = 0; k < M_; k++) {
			for (int l = 1; l <= M_ - 1; l++) {
				double theta = PIM_ * l;
				
				coef_[k + (l - 1) * M_] = new Snake3DNode(x0 + rX0 * scale2M
						* Math.sin(theta) * cc_[k], y0 + rY0 * scale2M
						* Math.sin(theta) * cs_[k], z0 + rZ0 * scale2M
						* Math.cos(theta));
				
				/*
				coef_[k + (l - 1) * M_] = new Snake3DNode(x0 + 1.9*r * scale2M
						* Math.sin(theta) * cc_[k], y0 + 0.9*r * scale2M
						* Math.sin(theta) * cs_[k], z0 + 1.4*r * scale2M
						* Math.cos(theta));
				*/
				
				
				
				
				// values first series
				/*
				coef_[k + (l - 1) * M_] = new Snake3DNode(x0 + 1.9*r * scale2M
						* Math.sin(theta) * cc_[k], y0 + 1.0*r * scale2M
						* Math.sin(theta) * cs_[k], z0 + 1.0*r * scale2M
						* Math.cos(theta));
				*/
						
				
				// z0 + 1.6r DP_1, CG_1, DR_1
				// x0 + 2.4r DP_1, CG_1, DR_1
				
				// x0 + 2.0 DS_1, GB_1
				// y0 + 1.2r DP_1, CG_1, DR_1, DS_1, GB_1
				
				// x0 + 1.9r MB_1, PF_1, TH_1
				// y0 + 1.1r MB_1, PF_1, TH_1
			}
		}
		
		// north pole
		coef_[M_ * (M_ - 1)] = new Snake3DNode(x0, y0, z0 + rZ0);
		// south pole
		coef_[M_ * (M_ - 1) + 1] = new Snake3DNode(x0, y0, z0 - rZ0);
		
		/*
		// north pole
		coef_[M_ * (M_ - 1)] = new Snake3DNode(x0, y0, z0 + r);
		// south pole
		coef_[M_ * (M_ - 1) + 1] = new Snake3DNode(x0, y0, z0 - r);
		*/
		
		
		
		// north tangent plane
		coef_[M_ * (M_ - 1) + 2] = new Snake3DNode(x0 + Math.PI * r, y0, z0 + r);
		// south tangent plane
		coef_[M_ * (M_ - 1) + 3] = new Snake3DNode(x0, y0 + Math.PI * r, z0 + r);
		coef_[M_ * (M_ - 1) + 4] = new Snake3DNode(x0 + Math.PI * r, y0, z0 - r);
		coef_[M_ * (M_ - 1) + 5] = new Snake3DNode(x0, y0 + Math.PI * r, z0 - r);
		
		if (Const.snakeCopiedSelected){
			System.out.println("Const snake init");
			coef_ = Const.coef_;			
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
				for (int l = 1; l <= M_ - 1; l++) {
					index_j = 2 * j + DISCRETIZATIONSAMPLINGRATE * (3 - 2 * l);
					if (index_j >= 0 && index_j < NR2_) {
						aux = spline2MFunc_[index_j];
						for (int k = 0; k < M_; k++) {
							index_i = (2 * i + DISCRETIZATIONSAMPLINGRATE
									* (3 - 2 * k) + MR2_)
									% MR2_;
							if (index_i >= 0 && index_i < NR2_) {
								basisFactor = aux * splineMFunc_[index_i];
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

				double scal = 1.0 / (M_ * splinePrime2MFunc_[5 * DISCRETIZATIONSAMPLINGRATE]);

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
				index_j = 2 * j + DISCRETIZATIONSAMPLINGRATE * 5;
				if (index_j >= 0 && index_j < NR2_) {
					aux = spline2MFunc_[index_j];
					for (int k = 0; k < M_; k++) {
						index_i = (2 * i + DISCRETIZATIONSAMPLINGRATE
								* (3 - 2 * k) + MR2_)
								% MR2_;
						if (index_i >= 0 && index_i < NR2_) {
							// compute c[k,-1]
							double ckminus1x = (coef_[k].x + scal
									* (cc_[k] * NorthV1x + cs_[k] * NorthV2x));
							double ckminus1y = (coef_[k].y + scal
									* (cc_[k] * NorthV1y + cs_[k] * NorthV2y));
							double ckminus1z = (coef_[k].z + scal
									* (cc_[k] * NorthV1z + cs_[k] * NorthV2z));

							basisFactor = aux * splineMFunc_[index_i];
							xPosVal += ckminus1x * basisFactor;
							yPosVal += ckminus1y * basisFactor;
							zPosVal += ckminus1z * basisFactor;
						}
					}
				}

				// l = 0
				index_j = 2 * j + DISCRETIZATIONSAMPLINGRATE * 3;
				if (index_j >= 0 && index_j < NR2_) {
					aux = spline2MFunc_[index_j];
					for (int k = 0; k < M_; k++) {
						index_i = (2 * i + DISCRETIZATIONSAMPLINGRATE
								* (3 - 2 * k) + MR2_)
								% MR2_;
						if (index_i >= 0 && index_i < NR2_) {
							// compute c[k,-1]
							double ckminus1x = (coef_[k].x + scal
									* (cc_[k] * NorthV1x + cs_[k] * NorthV2x));
							double ckminus1y = (coef_[k].y + scal
									* (cc_[k] * NorthV1y + cs_[k] * NorthV2y));
							double ckminus1z = (coef_[k].z + scal
									* (cc_[k] * NorthV1z + cs_[k] * NorthV2z));

							// compute c[k,0]
							double ck0x = (coef_[M_ * (M_ - 1)].x - spline2MFunc_[5 * DISCRETIZATIONSAMPLINGRATE]
									* (ckminus1x + coef_[k].x))
									/ spline2MFunc_[3 * DISCRETIZATIONSAMPLINGRATE];
							double ck0y = (coef_[M_ * (M_ - 1)].y - spline2MFunc_[5 * DISCRETIZATIONSAMPLINGRATE]
									* (ckminus1y + coef_[k].y))
									/ spline2MFunc_[3 * DISCRETIZATIONSAMPLINGRATE];
							double ck0z = (coef_[M_ * (M_ - 1)].z - spline2MFunc_[5 * DISCRETIZATIONSAMPLINGRATE]
									* (ckminus1z + coef_[k].z))
									/ spline2MFunc_[3 * DISCRETIZATIONSAMPLINGRATE];

							basisFactor = aux * splineMFunc_[index_i];
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
				index_j = 2 * j + DISCRETIZATIONSAMPLINGRATE * (1 - 2 * M_);
				if (index_j >= 0 && index_j < NR2_) {
					aux = spline2MFunc_[index_j];
					for (int k = 0; k < M_; k++) {
						index_i = (2 * i + DISCRETIZATIONSAMPLINGRATE
								* (3 - 2 * k) + MR2_)
								% MR2_;
						if (index_i >= 0 && index_i < NR2_) {
							// compute c[k,M+1]
							double ckMplusx = (coef_[k + (M_ - 2) * M_].x + scal
									* (cc_[k] * SouthV1x + cs_[k] * SouthV2x));
							double ckMplusy = (coef_[k + (M_ - 2) * M_].y + scal
									* (cc_[k] * SouthV1y + cs_[k] * SouthV2y));
							double ckMplusz = (coef_[k + (M_ - 2) * M_].z + scal
									* (cc_[k] * SouthV1z + cs_[k] * SouthV2z));

							basisFactor = aux * splineMFunc_[index_i];
							xPosVal += ckMplusx * basisFactor;
							yPosVal += ckMplusy * basisFactor;
							zPosVal += ckMplusz * basisFactor;
						}
					}
				}

				// l = M
				index_j = 2 * j + DISCRETIZATIONSAMPLINGRATE * (3 - 2 * M_);
				if (index_j >= 0 && index_j < NR2_) {
					aux = spline2MFunc_[index_j];
					for (int k = 0; k < M_; k++) {
						index_i = (2 * i + DISCRETIZATIONSAMPLINGRATE
								* (3 - 2 * k) + MR2_)
								% MR2_;
						if (index_i >= 0 && index_i < NR2_) {
							// compute c[k,M+1]
							double ckMplusx = (coef_[k + (M_ - 2) * M_].x + scal
									* (cc_[k] * SouthV1x + cs_[k] * SouthV2x));
							double ckMplusy = (coef_[k + (M_ - 2) * M_].y + scal
									* (cc_[k] * SouthV1y + cs_[k] * SouthV2y));
							double ckMplusz = (coef_[k + (M_ - 2) * M_].z + scal
									* (cc_[k] * SouthV1z + cs_[k] * SouthV2z));

							// compute c[k,M]
							double ckMx = (coef_[M_ * (M_ - 1) + 1].x - spline2MFunc_[5 * DISCRETIZATIONSAMPLINGRATE]
									* (ckMplusx + coef_[k + (M_ - 2) * M_].x))
									/ spline2MFunc_[3 * DISCRETIZATIONSAMPLINGRATE];
							double ckMy = (coef_[M_ * (M_ - 1) + 1].y - spline2MFunc_[5 * DISCRETIZATIONSAMPLINGRATE]
									* (ckMplusy + coef_[k + (M_ - 2) * M_].y))
									/ spline2MFunc_[3 * DISCRETIZATIONSAMPLINGRATE];
							double ckMz = (coef_[M_ * (M_ - 1) + 1].z - spline2MFunc_[5 * DISCRETIZATIONSAMPLINGRATE]
									* (ckMplusz + coef_[k + (M_ - 2) * M_].z))
									/ spline2MFunc_[3 * DISCRETIZATIONSAMPLINGRATE];

							basisFactor = aux * splineMFunc_[index_i];
							xPosVal += ckMx * basisFactor;
							yPosVal += ckMy * basisFactor;
							zPosVal += ckMz * basisFactor;
						}
					}
				}
				xSnake_[i][j] = xPosVal;
				ySnake_[i][j] = yPosVal;
				zSnake_[i][j] = zPosVal;
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
				xg += xSnake_[u][v];
				yg += ySnake_[u][v];
				zg += zSnake_[u][v];
			}
		}
		xg /= length;
		yg /= length;
		zg /= length;

		for (int u = 0; u < MR_; u++) {
			for (int v = 0; v <= MR_; v++) {
				xShell_[u][v] = CBRT * (xSnake_[u][v] - xg) + xg;
				yShell_[u][v] = CBRT * (ySnake_[u][v] - yg) + yg;
				zShell_[u][v] = CBRT * (zSnake_[u][v] - zg) + zg;
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
				for (int l = 1; l <= M_ - 1; l++) {
					index_j = 2 * j + DISCRETIZATIONSAMPLINGRATE * (3 - 2 * l);
					if (index_j >= 0 && index_j < NR2_) {
						aux = spline2MFunc_[index_j];
						for (int k = 0; k < M_; k++) {
							index_i = (2 * i + DISCRETIZATIONSAMPLINGRATE
									* (3 - 2 * k) + MR2_)
									% MR2_;
							if (index_i >= 0 && index_i < NR2_) {
								basisFactor = aux * splinePrimeMFunc_[index_i];
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

				double scal = 1.0 / (M_ * splinePrime2MFunc_[5 * DISCRETIZATIONSAMPLINGRATE]);

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
				index_j = 2 * j + DISCRETIZATIONSAMPLINGRATE * 5;
				if (index_j >= 0 && index_j < NR2_) {
					aux = spline2MFunc_[index_j];
					for (int k = 0; k < M_; k++) {
						index_i = (2 * i + DISCRETIZATIONSAMPLINGRATE
								* (3 - 2 * k) + MR2_)
								% MR2_;
						if (index_i >= 0 && index_i < NR2_) {
							// compute c[k,-1]
							double ckminus1x = (coef_[k].x + scal
									* (cc_[k] * NorthV1x + cs_[k] * NorthV2x));
							double ckminus1y = (coef_[k].y + scal
									* (cc_[k] * NorthV1y + cs_[k] * NorthV2y));
							double ckminus1z = (coef_[k].z + scal
									* (cc_[k] * NorthV1z + cs_[k] * NorthV2z));

							basisFactor = aux * splinePrimeMFunc_[index_i];
							xPosVal += ckminus1x * basisFactor;
							yPosVal += ckminus1y * basisFactor;
							zPosVal += ckminus1z * basisFactor;
						}
					}
				}

				// l = 0
				index_j = 2 * j + DISCRETIZATIONSAMPLINGRATE * 3;
				if (index_j >= 0 && index_j < NR2_) {
					aux = spline2MFunc_[index_j];
					for (int k = 0; k < M_; k++) {
						index_i = (2 * i + DISCRETIZATIONSAMPLINGRATE
								* (3 - 2 * k) + MR2_)
								% MR2_;
						if (index_i >= 0 && index_i < NR2_) {
							// compute c[k,-1]
							double ckminus1x = (coef_[k].x + scal
									* (cc_[k] * NorthV1x + cs_[k] * NorthV2x));
							double ckminus1y = (coef_[k].y + scal
									* (cc_[k] * NorthV1y + cs_[k] * NorthV2y));
							double ckminus1z = (coef_[k].z + scal
									* (cc_[k] * NorthV1z + cs_[k] * NorthV2z));

							// compute c[k,0]
							double ck0x = (coef_[M_ * (M_ - 1)].x - spline2MFunc_[5 * DISCRETIZATIONSAMPLINGRATE]
									* (ckminus1x + coef_[k].x))
									/ spline2MFunc_[3 * DISCRETIZATIONSAMPLINGRATE];
							double ck0y = (coef_[M_ * (M_ - 1)].y - spline2MFunc_[5 * DISCRETIZATIONSAMPLINGRATE]
									* (ckminus1y + coef_[k].y))
									/ spline2MFunc_[3 * DISCRETIZATIONSAMPLINGRATE];
							double ck0z = (coef_[M_ * (M_ - 1)].z - spline2MFunc_[5 * DISCRETIZATIONSAMPLINGRATE]
									* (ckminus1z + coef_[k].z))
									/ spline2MFunc_[3 * DISCRETIZATIONSAMPLINGRATE];

							basisFactor = aux * splinePrimeMFunc_[index_i];
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
				index_j = 2 * j + DISCRETIZATIONSAMPLINGRATE * (1 - 2 * M_);
				if (index_j >= 0 && index_j < NR2_) {
					aux = spline2MFunc_[index_j];
					for (int k = 0; k < M_; k++) {
						index_i = (2 * i + DISCRETIZATIONSAMPLINGRATE
								* (3 - 2 * k) + MR2_)
								% MR2_;
						if (index_i >= 0 && index_i < NR2_) {
							// compute c[k,M+1]
							double ckMplusx = (coef_[k + (M_ - 2) * M_].x + scal
									* (cc_[k] * SouthV1x + cs_[k] * SouthV2x));
							double ckMplusy = (coef_[k + (M_ - 2) * M_].y + scal
									* (cc_[k] * SouthV1y + cs_[k] * SouthV2y));
							double ckMplusz = (coef_[k + (M_ - 2) * M_].z + scal
									* (cc_[k] * SouthV1z + cs_[k] * SouthV2z));

							basisFactor = aux * splinePrimeMFunc_[index_i];
							xPosVal += ckMplusx * basisFactor;
							yPosVal += ckMplusy * basisFactor;
							zPosVal += ckMplusz * basisFactor;
						}
					}
				}

				// l = M
				index_j = 2 * j + DISCRETIZATIONSAMPLINGRATE * (3 - 2 * M_);
				if (index_j >= 0 && index_j < NR2_) {
					aux = spline2MFunc_[index_j];
					for (int k = 0; k < M_; k++) {
						index_i = (2 * i + DISCRETIZATIONSAMPLINGRATE
								* (3 - 2 * k) + MR2_)
								% MR2_;
						if (index_i >= 0 && index_i < NR2_) {
							// compute c[k,M+1]
							double ckMplusx = (coef_[k + (M_ - 2) * M_].x + scal
									* (cc_[k] * SouthV1x + cs_[k] * SouthV2x));
							double ckMplusy = (coef_[k + (M_ - 2) * M_].y + scal
									* (cc_[k] * SouthV1y + cs_[k] * SouthV2y));
							double ckMplusz = (coef_[k + (M_ - 2) * M_].z + scal
									* (cc_[k] * SouthV1z + cs_[k] * SouthV2z));

							// compute c[k,M]
							double ckMx = (coef_[M_ * (M_ - 1) + 1].x - spline2MFunc_[5 * DISCRETIZATIONSAMPLINGRATE]
									* (ckMplusx + coef_[k + (M_ - 2) * M_].x))
									/ spline2MFunc_[3 * DISCRETIZATIONSAMPLINGRATE];
							double ckMy = (coef_[M_ * (M_ - 1) + 1].y - spline2MFunc_[5 * DISCRETIZATIONSAMPLINGRATE]
									* (ckMplusy + coef_[k + (M_ - 2) * M_].y))
									/ spline2MFunc_[3 * DISCRETIZATIONSAMPLINGRATE];
							double ckMz = (coef_[M_ * (M_ - 1) + 1].z - spline2MFunc_[5 * DISCRETIZATIONSAMPLINGRATE]
									* (ckMplusz + coef_[k + (M_ - 2) * M_].z))
									/ spline2MFunc_[3 * DISCRETIZATIONSAMPLINGRATE];

							basisFactor = aux * splinePrimeMFunc_[index_i];
							xPosVal += ckMx * basisFactor;
							yPosVal += ckMy * basisFactor;
							zPosVal += ckMz * basisFactor;
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
				for (int l = 1; l <= M_ - 1; l++) {
					index_j = 2 * j + DISCRETIZATIONSAMPLINGRATE * (3 - 2 * l);
					if (index_j >= 0 && index_j < NR2_) {
						aux = splinePrime2MFunc_[index_j];
						for (int k = 0; k < M_; k++) {
							index_i = (2 * i + DISCRETIZATIONSAMPLINGRATE
									* (3 - 2 * k) + MR2_)
									% MR2_;
							if (index_i >= 0 && index_i < NR2_) {
								basisFactor = aux * splineMFunc_[index_i];
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

				double scal = 1.0 / (M_ * splinePrime2MFunc_[5 * DISCRETIZATIONSAMPLINGRATE]);

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
				index_j = 2 * j + DISCRETIZATIONSAMPLINGRATE * 5;
				if (index_j >= 0 && index_j < NR2_) {
					aux = splinePrime2MFunc_[index_j];
					for (int k = 0; k < M_; k++) {
						index_i = (2 * i + DISCRETIZATIONSAMPLINGRATE
								* (3 - 2 * k) + MR2_)
								% MR2_;
						if (index_i >= 0 && index_i < NR2_) {
							// compute c[k,-1]
							double ckminus1x = (coef_[k].x + scal
									* (cc_[k] * NorthV1x + cs_[k] * NorthV2x));
							double ckminus1y = (coef_[k].y + scal
									* (cc_[k] * NorthV1y + cs_[k] * NorthV2y));
							double ckminus1z = (coef_[k].z + scal
									* (cc_[k] * NorthV1z + cs_[k] * NorthV2z));

							basisFactor = aux * splineMFunc_[index_i];
							xPosVal += ckminus1x * basisFactor;
							yPosVal += ckminus1y * basisFactor;
							zPosVal += ckminus1z * basisFactor;
						}
					}
				}

				// l = 0
				index_j = 2 * j + DISCRETIZATIONSAMPLINGRATE * 3;
				if (index_j >= 0 && index_j < NR2_) {
					aux = splinePrime2MFunc_[index_j];
					for (int k = 0; k < M_; k++) {
						index_i = (2 * i + DISCRETIZATIONSAMPLINGRATE
								* (3 - 2 * k) + MR2_)
								% MR2_;
						if (index_i >= 0 && index_i < NR2_) {
							// compute c[k,-1]
							double ckminus1x = (coef_[k].x + scal
									* (cc_[k] * NorthV1x + cs_[k] * NorthV2x));
							double ckminus1y = (coef_[k].y + scal
									* (cc_[k] * NorthV1y + cs_[k] * NorthV2y));
							double ckminus1z = (coef_[k].z + scal
									* (cc_[k] * NorthV1z + cs_[k] * NorthV2z));

							// compute c[k,0]
							double ck0x = (coef_[M_ * (M_ - 1)].x - spline2MFunc_[5 * DISCRETIZATIONSAMPLINGRATE]
									* (ckminus1x + coef_[k].x))
									/ spline2MFunc_[3 * DISCRETIZATIONSAMPLINGRATE];
							double ck0y = (coef_[M_ * (M_ - 1)].y - spline2MFunc_[5 * DISCRETIZATIONSAMPLINGRATE]
									* (ckminus1y + coef_[k].y))
									/ spline2MFunc_[3 * DISCRETIZATIONSAMPLINGRATE];
							double ck0z = (coef_[M_ * (M_ - 1)].z - spline2MFunc_[5 * DISCRETIZATIONSAMPLINGRATE]
									* (ckminus1z + coef_[k].z))
									/ spline2MFunc_[3 * DISCRETIZATIONSAMPLINGRATE];

							basisFactor = aux * splineMFunc_[index_i];
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
				index_j = 2 * j + DISCRETIZATIONSAMPLINGRATE * (1 - 2 * M_);
				if (index_j >= 0 && index_j < NR2_) {
					aux = splinePrime2MFunc_[index_j];
					for (int k = 0; k < M_; k++) {
						index_i = (2 * i + DISCRETIZATIONSAMPLINGRATE
								* (3 - 2 * k) + MR2_)
								% MR2_;
						if (index_i >= 0 && index_i < NR2_) {
							// compute c[k,M+1]
							double ckMplusx = (coef_[k + (M_ - 2) * M_].x + scal
									* (cc_[k] * SouthV1x + cs_[k] * SouthV2x));
							double ckMplusy = (coef_[k + (M_ - 2) * M_].y + scal
									* (cc_[k] * SouthV1y + cs_[k] * SouthV2y));
							double ckMplusz = (coef_[k + (M_ - 2) * M_].z + scal
									* (cc_[k] * SouthV1z + cs_[k] * SouthV2z));

							basisFactor = aux * splineMFunc_[index_i];
							xPosVal += ckMplusx * basisFactor;
							yPosVal += ckMplusy * basisFactor;
							zPosVal += ckMplusz * basisFactor;
						}
					}
				}

				// l = M
				index_j = 2 * j + DISCRETIZATIONSAMPLINGRATE * (3 - 2 * M_);
				if (index_j >= 0 && index_j < NR2_) {
					aux = splinePrime2MFunc_[index_j];
					for (int k = 0; k < M_; k++) {
						index_i = (2 * i + DISCRETIZATIONSAMPLINGRATE
								* (3 - 2 * k) + MR2_)
								% MR2_;
						if (index_i >= 0 && index_i < NR2_) {
							// compute c[k,M+1]
							double ckMplusx = (coef_[k + (M_ - 2) * M_].x + scal
									* (cc_[k] * SouthV1x + cs_[k] * SouthV2x));
							double ckMplusy = (coef_[k + (M_ - 2) * M_].y + scal
									* (cc_[k] * SouthV1y + cs_[k] * SouthV2y));
							double ckMplusz = (coef_[k + (M_ - 2) * M_].z + scal
									* (cc_[k] * SouthV1z + cs_[k] * SouthV2z));

							// compute c[k,M]
							double ckMx = (coef_[M_ * (M_ - 1) + 1].x - spline2MFunc_[5 * DISCRETIZATIONSAMPLINGRATE]
									* (ckMplusx + coef_[k + (M_ - 2) * M_].x))
									/ spline2MFunc_[3 * DISCRETIZATIONSAMPLINGRATE];
							double ckMy = (coef_[M_ * (M_ - 1) + 1].y - spline2MFunc_[5 * DISCRETIZATIONSAMPLINGRATE]
									* (ckMplusy + coef_[k + (M_ - 2) * M_].y))
									/ spline2MFunc_[3 * DISCRETIZATIONSAMPLINGRATE];
							double ckMz = (coef_[M_ * (M_ - 1) + 1].z - spline2MFunc_[5 * DISCRETIZATIONSAMPLINGRATE]
									* (ckMplusz + coef_[k + (M_ - 2) * M_].z))
									/ spline2MFunc_[3 * DISCRETIZATIONSAMPLINGRATE];

							basisFactor = aux * splineMFunc_[index_i];
							xPosVal += ckMx * basisFactor;
							yPosVal += ckMy * basisFactor;
							zPosVal += ckMz * basisFactor;
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
	
	/**
	 * Computes the energy associated to the distance map
	 */
	private double computedDistanceMapEnergy(){		
		
		int FACTOR = 1000000;
		
		//
		double delta_uv = 1.0 / (MR_ * MR_);
		double energy = 0.0;
		double fyLap_val;
		int x0, x1, y0, y1, z0, z1;
		double DeltaX, DeltaY, DeltaZ;

		int imageWidth = imageLUTs_.getImageWidth();
		int imageHeight = imageLUTs_.getImageHeight();
		int imageDepth = imageLUTs_.getImageDepth();
		double[][] imageDataArray = imageLUTs_.getImageDataArray();

		double deltax = -0.5;//0;
		double deltay = -0.5;
		double deltaz = -0.5;//0;
		
		for (int u = 0; u < MR_; u++) {
			for (int v = 0; v <= MR_; v++) {
				x0 = (int) Math.floor(xSnake_[u][v] + deltax);
				y0 = (int) Math.floor(ySnake_[u][v] + deltay);
				z0 = (int) Math.floor(zSnake_[u][v] + deltaz);

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

				DeltaX = xSnake_[u][v] + deltax - x0;
				DeltaY = ySnake_[u][v] + deltay - y0;
				DeltaZ = zSnake_[u][v] + deltaz - z0;

				double i1 =
						(1 - DeltaZ)* imageDataArray[z0][x0 + imageWidth * y0] 
						+ DeltaZ* imageDataArray[z1][x0 + imageWidth * y0];
				
				double i2 = 
						(1 - DeltaZ)* imageDataArray[z0][x0 + imageWidth * y1]
						+ DeltaZ * imageDataArray[z1][x0 + imageWidth * y1];
				
				double j1 = 
						(1 - DeltaZ) * imageDataArray[z0][x1 + imageWidth * y0]
						+ DeltaZ* imageDataArray[z1][x1 + imageWidth * y0];
				
				double j2 = 
						(1 - DeltaZ)* imageDataArray[z0][x1 + imageWidth * y1]
						+ DeltaZ* imageDataArray[z1][x1 + imageWidth * y1];
				
				double w1 = i1 * (1 - DeltaY) + i2 * DeltaY;
				double w2 = j1 * (1 - DeltaY) + j2 * DeltaY;
				fyLap_val = w1 * (1 - DeltaX) + w2 * DeltaX;
				energy += fyLap_val * delta_uv *  Math.sqrt((dyWedgedz_[u][v] * dyWedgedz_[u][v]
						+ dzWedgedx_[u][v] * dzWedgedx_[u][v] + dxWedgedy_[u][v]
						* dxWedgedy_[u][v]));//dzWedgedx_[u][v];
			}
		}
		
		return 1000*FACTOR * energy/surfaceArea_;		
		
	}

	
	
	// ----------------------------------------------------------------------------
	
	
	/** Computes the area of the surface of the snake. */
	private void updateArea() {
		surfaceArea_ = 0;
		double delta_uv = 1.0 / (MR_ * MR_);
		for (int i = 0; i < MR_; i++) {
			for (int j = 0; j <= MR_; j++) {
				surfaceArea_ += JACOBIANCONSTANT
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
				volume_ += ySnake_[u][v] * delta_uv * dzWedgedx_[u][v];
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
				x0 = (int) Math.floor(xSnake_[u][v] + deltax);
				y0 = (int) Math.floor(ySnake_[u][v] + deltay);
				z0 = (int) Math.floor(zSnake_[u][v] + deltaz);

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

				DeltaX = xSnake_[u][v] + deltax - x0;
				DeltaY = ySnake_[u][v] + deltay - y0;
				DeltaZ = zSnake_[u][v] + deltaz - z0;

				double i1 =
						(1 - DeltaZ)* preintegratedFilteredImageDataArray[z0][x0 + imageWidth * y0] 
						+ DeltaZ* preintegratedFilteredImageDataArray[z1][x0 + imageWidth * y0];
				
				double i2 = 
						(1 - DeltaZ)* preintegratedFilteredImageDataArray[z0][x0 + imageWidth * y1]
						+ DeltaZ * preintegratedFilteredImageDataArray[z1][x0 + imageWidth * y1];
				
				double j1 = 
						(1 - DeltaZ) * preintegratedFilteredImageDataArray[z0][x1 + imageWidth * y0]
						+ DeltaZ* preintegratedFilteredImageDataArray[z1][x1 + imageWidth * y0];
				
				double j2 = 
						(1 - DeltaZ)* preintegratedFilteredImageDataArray[z0][x1 + imageWidth * y1]
						+ DeltaZ* preintegratedFilteredImageDataArray[z1][x1 + imageWidth * y1];
				
				double w1 = i1 * (1 - DeltaY) + i2 * DeltaY;
				double w2 = j1 * (1 - DeltaY) + j2 * DeltaY;
				fyLap_val = w1 * (1 - DeltaX) + w2 * DeltaX;
				energy += fyLap_val * delta_uv * dzWedgedx_[u][v];
			}
		}
		
		return 1000 * energy;
	}

	// ----------------------------------------------------------------------------
	/** Computes the region energy in case of Brain MP2RAGE. */
	private double computeRegionEnergyBrainMP2RAGE() {
		//if (life_ > 5000)
		//	return 0;
		
		double[][] preintegratedImageDataArray = Const.LUT_MP2RAGE.getPreintegratedImageDataArray();
		double internalContribution = computeSurfaceIntegral(
				preintegratedImageDataArray, xSnake_, ySnake_, zSnake_);
		double externalContribution = CBRT
				* computeSurfaceIntegral(preintegratedImageDataArray, xShell_,
						yShell_, zShell_);
		double energy = 1000
				* (externalContribution - 2 * internalContribution) / volume_;
		
		//System.out.println("energy = " + energy);
		return -energy;
	}
	//
	

	/** Computes the region energy. */
	private double computeRegionEnergy() {	
		
		if (Const.segmentBrainMP2RAGE)			
			return computeRegionEnergyBrainMP2RAGE();
		
		double[][] preintegratedImageDataArray = imageLUTs_
				.getPreintegratedImageDataArray();
		double internalContribution = computeSurfaceIntegral(
				preintegratedImageDataArray, xSnake_, ySnake_, zSnake_);
		double externalContribution = CBRT
				* computeSurfaceIntegral(preintegratedImageDataArray, xShell_,
						yShell_, zShell_);
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
				x = xSnake_[u][v];
				y = ySnake_[u][v];
				z = zSnake_[u][v];

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
					x = xShell_[u][v];
					y = yShell_[u][v];
					z = zShell_[u][v];

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
						* Math.abs(surfaceArea_
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
	// SELF-INTERSECTION METHODS

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
					index_v = 2 * j + DISCRETIZATIONSAMPLINGRATE * (3 - 2 * l);
					if (index_v >= 0 && index_v < NR2_) {
						aux = spline2MFunc_[index_v];
						for (int k = 0; k < M_; k++) {
							index_u = (2 * i + DISCRETIZATIONSAMPLINGRATE
									* (3 - 2 * k) + MR2_)
									% MR2_;
							if (index_u >= 0 && index_u < NR2_) {
								basisFactor = aux
										* splinePrimePrimeMFunc_[index_u];
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

				double scal = 1.0 / (M_ * splinePrime2MFunc_[5 * DISCRETIZATIONSAMPLINGRATE]);

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
				index_v = 2 * j + DISCRETIZATIONSAMPLINGRATE * 5;
				if (index_v >= 0 && index_v < NR2_) {
					aux = spline2MFunc_[index_v];
					for (int k = 0; k < M_; k++) {
						index_u = (2 * i + DISCRETIZATIONSAMPLINGRATE
								* (3 - 2 * k) + MR2_)
								% MR2_;
						if (index_u >= 0 && index_u < NR2_) {
							// compute c[k,-1]
							double ckminus1x = (coef_[k].x + scal
									* (cc_[k] * NorthV1x + cs_[k] * NorthV2x));
							double ckminus1y = (coef_[k].y + scal
									* (cc_[k] * NorthV1y + cs_[k] * NorthV2y));
							double ckminus1z = (coef_[k].z + scal
									* (cc_[k] * NorthV1z + cs_[k] * NorthV2z));

							basisFactor = aux * splinePrimePrimeMFunc_[index_u];
							xPosVal += ckminus1x * basisFactor;
							yPosVal += ckminus1y * basisFactor;
							zPosVal += ckminus1z * basisFactor;
						}
					}
				}

				// l = 0
				index_v = 2 * j + DISCRETIZATIONSAMPLINGRATE * 3;
				if (index_v >= 0 && index_v < NR2_) {
					aux = spline2MFunc_[index_v];
					for (int k = 0; k < M_; k++) {
						index_u = (2 * i + DISCRETIZATIONSAMPLINGRATE
								* (3 - 2 * k) + MR2_)
								% MR2_;
						if (index_u >= 0 && index_u < NR2_) {
							// compute c[k,-1]
							double ckminus1x = (coef_[k].x + scal
									* (cc_[k] * NorthV1x + cs_[k] * NorthV2x));
							double ckminus1y = (coef_[k].y + scal
									* (cc_[k] * NorthV1y + cs_[k] * NorthV2y));
							double ckminus1z = (coef_[k].z + scal
									* (cc_[k] * NorthV1z + cs_[k] * NorthV2z));

							// compute c[k,0]
							double ck0x = (coef_[M_ * (M_ - 1)].x - spline2MFunc_[5 * DISCRETIZATIONSAMPLINGRATE]
									* (ckminus1x + coef_[k].x))
									/ spline2MFunc_[3 * DISCRETIZATIONSAMPLINGRATE];
							double ck0y = (coef_[M_ * (M_ - 1)].y - spline2MFunc_[5 * DISCRETIZATIONSAMPLINGRATE]
									* (ckminus1y + coef_[k].y))
									/ spline2MFunc_[3 * DISCRETIZATIONSAMPLINGRATE];
							double ck0z = (coef_[M_ * (M_ - 1)].z - spline2MFunc_[5 * DISCRETIZATIONSAMPLINGRATE]
									* (ckminus1z + coef_[k].z))
									/ spline2MFunc_[3 * DISCRETIZATIONSAMPLINGRATE];

							basisFactor = aux * splinePrimePrimeMFunc_[index_u];
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
				index_v = 2 * j + DISCRETIZATIONSAMPLINGRATE * (1 - 2 * M_);
				if (index_v >= 0 && index_v < NR2_) {
					aux = spline2MFunc_[index_v];
					for (int k = 0; k < M_; k++) {
						index_u = (2 * i + DISCRETIZATIONSAMPLINGRATE
								* (3 - 2 * k) + MR2_)
								% MR2_;
						if (index_u >= 0 && index_u < NR2_) {
							// compute c[k,M+1]
							double ckMplusx = (coef_[k + (M_ - 2) * M_].x + scal
									* (cc_[k] * SouthV1x + cs_[k] * SouthV2x));
							double ckMplusy = (coef_[k + (M_ - 2) * M_].y + scal
									* (cc_[k] * SouthV1y + cs_[k] * SouthV2y));
							double ckMplusz = (coef_[k + (M_ - 2) * M_].z + scal
									* (cc_[k] * SouthV1z + cs_[k] * SouthV2z));

							basisFactor = aux * splinePrimePrimeMFunc_[index_u];
							xPosVal += ckMplusx * basisFactor;
							yPosVal += ckMplusy * basisFactor;
							zPosVal += ckMplusz * basisFactor;
						}
					}
				}

				// l = M
				index_v = 2 * j + DISCRETIZATIONSAMPLINGRATE * (3 - 2 * M_);
				if (index_v >= 0 && index_v < NR2_) {
					aux = spline2MFunc_[index_v];
					for (int k = 0; k < M_; k++) {
						index_u = (2 * i + DISCRETIZATIONSAMPLINGRATE
								* (3 - 2 * k) + MR2_)
								% MR2_;
						if (index_u >= 0 && index_u < NR2_) {
							// compute c[k,M+1]
							double ckMplusx = (coef_[k + (M_ - 2) * M_].x + scal
									* (cc_[k] * SouthV1x + cs_[k] * SouthV2x));
							double ckMplusy = (coef_[k + (M_ - 2) * M_].y + scal
									* (cc_[k] * SouthV1y + cs_[k] * SouthV2y));
							double ckMplusz = (coef_[k + (M_ - 2) * M_].z + scal
									* (cc_[k] * SouthV1z + cs_[k] * SouthV2z));

							// compute c[k,M]
							double ckMx = (coef_[M_ * (M_ - 1) + 1].x - spline2MFunc_[5 * DISCRETIZATIONSAMPLINGRATE]
									* (ckMplusx + coef_[k + (M_ - 2) * M_].x))
									/ spline2MFunc_[3 * DISCRETIZATIONSAMPLINGRATE];
							double ckMy = (coef_[M_ * (M_ - 1) + 1].y - spline2MFunc_[5 * DISCRETIZATIONSAMPLINGRATE]
									* (ckMplusy + coef_[k + (M_ - 2) * M_].y))
									/ spline2MFunc_[3 * DISCRETIZATIONSAMPLINGRATE];
							double ckMz = (coef_[M_ * (M_ - 1) + 1].z - spline2MFunc_[5 * DISCRETIZATIONSAMPLINGRATE]
									* (ckMplusz + coef_[k + (M_ - 2) * M_].z))
									/ spline2MFunc_[3 * DISCRETIZATIONSAMPLINGRATE];

							basisFactor = aux * splinePrimePrimeMFunc_[index_u];
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
					index_v = 2 * j + DISCRETIZATIONSAMPLINGRATE * (3 - 2 * l);
					if (index_v >= 0 && index_v < NR2_) {
						aux = splinePrimePrime2MFunc_[index_v];
						for (int k = 0; k < M_; k++) {
							index_u = (2 * i + DISCRETIZATIONSAMPLINGRATE
									* (3 - 2 * k) + MR2_)
									% MR2_;
							if (index_u >= 0 && index_u < NR2_) {
								basisFactor = aux * splineMFunc_[index_u];
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

				double scal = 1.0 / (M_ * splinePrime2MFunc_[5 * DISCRETIZATIONSAMPLINGRATE]);

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
				index_v = 2 * j + DISCRETIZATIONSAMPLINGRATE * 5;
				if (index_v >= 0 && index_v < NR2_) {
					aux = splinePrimePrime2MFunc_[index_v];
					for (int k = 0; k < M_; k++) {
						index_u = (2 * i + DISCRETIZATIONSAMPLINGRATE
								* (3 - 2 * k) + MR2_)
								% MR2_;
						if (index_u >= 0 && index_u < NR2_) {
							// compute c[k,-1]
							double ckminus1x = (coef_[k].x + scal
									* (cc_[k] * NorthV1x + cs_[k] * NorthV2x));
							double ckminus1y = (coef_[k].y + scal
									* (cc_[k] * NorthV1y + cs_[k] * NorthV2y));
							double ckminus1z = (coef_[k].z + scal
									* (cc_[k] * NorthV1z + cs_[k] * NorthV2z));

							basisFactor = aux * splineMFunc_[index_u];
							xPosVal += ckminus1x * basisFactor;
							yPosVal += ckminus1y * basisFactor;
							zPosVal += ckminus1z * basisFactor;
						}
					}
				}

				// l = 0
				index_v = 2 * j + DISCRETIZATIONSAMPLINGRATE * 3;
				if (index_v >= 0 && index_v < NR2_) {
					aux = splinePrimePrime2MFunc_[index_v];
					for (int k = 0; k < M_; k++) {
						index_u = (2 * i + DISCRETIZATIONSAMPLINGRATE
								* (3 - 2 * k) + MR2_)
								% MR2_;
						if (index_u >= 0 && index_u < NR2_) {
							// compute c[k,-1]
							double ckminus1x = (coef_[k].x + scal
									* (cc_[k] * NorthV1x + cs_[k] * NorthV2x));
							double ckminus1y = (coef_[k].y + scal
									* (cc_[k] * NorthV1y + cs_[k] * NorthV2y));
							double ckminus1z = (coef_[k].z + scal
									* (cc_[k] * NorthV1z + cs_[k] * NorthV2z));

							// compute c[k,0]
							double ck0x = (coef_[M_ * (M_ - 1)].x - spline2MFunc_[5 * DISCRETIZATIONSAMPLINGRATE]
									* (ckminus1x + coef_[k].x))
									/ spline2MFunc_[3 * DISCRETIZATIONSAMPLINGRATE];
							double ck0y = (coef_[M_ * (M_ - 1)].y - spline2MFunc_[5 * DISCRETIZATIONSAMPLINGRATE]
									* (ckminus1y + coef_[k].y))
									/ spline2MFunc_[3 * DISCRETIZATIONSAMPLINGRATE];
							double ck0z = (coef_[M_ * (M_ - 1)].z - spline2MFunc_[5 * DISCRETIZATIONSAMPLINGRATE]
									* (ckminus1z + coef_[k].z))
									/ spline2MFunc_[3 * DISCRETIZATIONSAMPLINGRATE];

							basisFactor = aux * splineMFunc_[index_u];
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
				index_v = 2 * j + DISCRETIZATIONSAMPLINGRATE * (1 - 2 * M_);
				if (index_v >= 0 && index_v < NR2_) {
					aux = splinePrimePrime2MFunc_[index_v];
					for (int k = 0; k < M_; k++) {
						index_u = (2 * i + DISCRETIZATIONSAMPLINGRATE
								* (3 - 2 * k) + MR2_)
								% MR2_;
						if (index_u >= 0 && index_u < NR2_) {
							// compute c[k,M+1]
							double ckMplusx = (coef_[k + (M_ - 2) * M_].x + scal
									* (cc_[k] * SouthV1x + cs_[k] * SouthV2x));
							double ckMplusy = (coef_[k + (M_ - 2) * M_].y + scal
									* (cc_[k] * SouthV1y + cs_[k] * SouthV2y));
							double ckMplusz = (coef_[k + (M_ - 2) * M_].z + scal
									* (cc_[k] * SouthV1z + cs_[k] * SouthV2z));

							basisFactor = aux * splineMFunc_[index_u];
							xPosVal += ckMplusx * basisFactor;
							yPosVal += ckMplusy * basisFactor;
							zPosVal += ckMplusz * basisFactor;
						}
					}
				}

				// l = M
				index_v = 2 * j + DISCRETIZATIONSAMPLINGRATE * (3 - 2 * M_);
				if (index_v >= 0 && index_v < NR2_) {
					aux = splinePrimePrime2MFunc_[index_v];
					for (int k = 0; k < M_; k++) {
						index_u = (2 * i + DISCRETIZATIONSAMPLINGRATE
								* (3 - 2 * k) + MR2_)
								% MR2_;
						if (index_u >= 0 && index_u < NR2_) {
							// compute c[k,M+1]
							double ckMplusx = (coef_[k + (M_ - 2) * M_].x + scal
									* (cc_[k] * SouthV1x + cs_[k] * SouthV2x));
							double ckMplusy = (coef_[k + (M_ - 2) * M_].y + scal
									* (cc_[k] * SouthV1y + cs_[k] * SouthV2y));
							double ckMplusz = (coef_[k + (M_ - 2) * M_].z + scal
									* (cc_[k] * SouthV1z + cs_[k] * SouthV2z));

							// compute c[k,M]
							double ckMx = (coef_[M_ * (M_ - 1) + 1].x - spline2MFunc_[5 * DISCRETIZATIONSAMPLINGRATE]
									* (ckMplusx + coef_[k + (M_ - 2) * M_].x))
									/ spline2MFunc_[3 * DISCRETIZATIONSAMPLINGRATE];
							double ckMy = (coef_[M_ * (M_ - 1) + 1].y - spline2MFunc_[5 * DISCRETIZATIONSAMPLINGRATE]
									* (ckMplusy + coef_[k + (M_ - 2) * M_].y))
									/ spline2MFunc_[3 * DISCRETIZATIONSAMPLINGRATE];
							double ckMz = (coef_[M_ * (M_ - 1) + 1].z - spline2MFunc_[5 * DISCRETIZATIONSAMPLINGRATE]
									* (ckMplusz + coef_[k + (M_ - 2) * M_].z))
									/ spline2MFunc_[3 * DISCRETIZATIONSAMPLINGRATE];

							basisFactor = aux * splineMFunc_[index_u];
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
					index_v = 2 * j + DISCRETIZATIONSAMPLINGRATE * (3 - 2 * l);
					if (index_v >= 0 && index_v < NR2_) {
						aux = splinePrime2MFunc_[index_v];
						for (int k = 0; k < M_; k++) {
							index_u = (2 * i + DISCRETIZATIONSAMPLINGRATE
									* (3 - 2 * k) + MR2_)
									% MR2_;
							if (index_u >= 0 && index_u < NR2_) {
								basisFactor = aux * splinePrimeMFunc_[index_u];
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

				double scal = 1.0 / (M_ * splinePrime2MFunc_[5 * DISCRETIZATIONSAMPLINGRATE]);

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
				index_v = 2 * j + DISCRETIZATIONSAMPLINGRATE * 5;
				if (index_v >= 0 && index_v < NR2_) {
					aux = splinePrime2MFunc_[index_v];
					for (int k = 0; k < M_; k++) {
						index_u = (2 * i + DISCRETIZATIONSAMPLINGRATE
								* (3 - 2 * k) + MR2_)
								% MR2_;
						if (index_u >= 0 && index_u < NR2_) {
							// compute c[k,-1]
							double ckminus1x = (coef_[k].x + scal
									* (cc_[k] * NorthV1x + cs_[k] * NorthV2x));
							double ckminus1y = (coef_[k].y + scal
									* (cc_[k] * NorthV1y + cs_[k] * NorthV2y));
							double ckminus1z = (coef_[k].z + scal
									* (cc_[k] * NorthV1z + cs_[k] * NorthV2z));

							basisFactor = aux * splinePrimeMFunc_[index_u];
							xPosVal += ckminus1x * basisFactor;
							yPosVal += ckminus1y * basisFactor;
							zPosVal += ckminus1z * basisFactor;
						}
					}
				}

				// l = 0
				index_v = 2 * j + DISCRETIZATIONSAMPLINGRATE * 3;
				if (index_v >= 0 && index_v < NR2_) {
					aux = splinePrime2MFunc_[index_v];
					for (int k = 0; k < M_; k++) {
						index_u = (2 * i + DISCRETIZATIONSAMPLINGRATE
								* (3 - 2 * k) + MR2_)
								% MR2_;
						if (index_u >= 0 && index_u < NR2_) {
							// compute c[k,-1]
							double ckminus1x = (coef_[k].x + scal
									* (cc_[k] * NorthV1x + cs_[k] * NorthV2x));
							double ckminus1y = (coef_[k].y + scal
									* (cc_[k] * NorthV1y + cs_[k] * NorthV2y));
							double ckminus1z = (coef_[k].z + scal
									* (cc_[k] * NorthV1z + cs_[k] * NorthV2z));

							// compute c[k,0]
							double ck0x = (coef_[M_ * (M_ - 1)].x - spline2MFunc_[5 * DISCRETIZATIONSAMPLINGRATE]
									* (ckminus1x + coef_[k].x))
									/ spline2MFunc_[3 * DISCRETIZATIONSAMPLINGRATE];
							double ck0y = (coef_[M_ * (M_ - 1)].y - spline2MFunc_[5 * DISCRETIZATIONSAMPLINGRATE]
									* (ckminus1y + coef_[k].y))
									/ spline2MFunc_[3 * DISCRETIZATIONSAMPLINGRATE];
							double ck0z = (coef_[M_ * (M_ - 1)].z - spline2MFunc_[5 * DISCRETIZATIONSAMPLINGRATE]
									* (ckminus1z + coef_[k].z))
									/ spline2MFunc_[3 * DISCRETIZATIONSAMPLINGRATE];

							basisFactor = aux * splinePrimeMFunc_[index_u];
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
				index_v = 2 * j + DISCRETIZATIONSAMPLINGRATE * (1 - 2 * M_);
				if (index_v >= 0 && index_v < NR2_) {
					aux = splinePrime2MFunc_[index_v];
					for (int k = 0; k < M_; k++) {
						index_u = (2 * i + DISCRETIZATIONSAMPLINGRATE
								* (3 - 2 * k) + MR2_)
								% MR2_;
						if (index_u >= 0 && index_u < NR2_) {
							// compute c[k,M+1]
							double ckMplusx = (coef_[k + (M_ - 2) * M_].x + scal
									* (cc_[k] * SouthV1x + cs_[k] * SouthV2x));
							double ckMplusy = (coef_[k + (M_ - 2) * M_].y + scal
									* (cc_[k] * SouthV1y + cs_[k] * SouthV2y));
							double ckMplusz = (coef_[k + (M_ - 2) * M_].z + scal
									* (cc_[k] * SouthV1z + cs_[k] * SouthV2z));

							basisFactor = aux * splinePrimeMFunc_[index_u];
							xPosVal += ckMplusx * basisFactor;
							yPosVal += ckMplusy * basisFactor;
							zPosVal += ckMplusz * basisFactor;
						}
					}
				}

				// l = M
				index_v = 2 * j + DISCRETIZATIONSAMPLINGRATE * (3 - 2 * M_);
				if (index_v >= 0 && index_v < NR2_) {
					aux = splinePrime2MFunc_[index_v];
					for (int k = 0; k < M_; k++) {
						index_u = (2 * i + DISCRETIZATIONSAMPLINGRATE
								* (3 - 2 * k) + MR2_)
								% MR2_;
						if (index_u >= 0 && index_u < NR2_) {
							// compute c[k,M+1]
							double ckMplusx = (coef_[k + (M_ - 2) * M_].x + scal
									* (cc_[k] * SouthV1x + cs_[k] * SouthV2x));
							double ckMplusy = (coef_[k + (M_ - 2) * M_].y + scal
									* (cc_[k] * SouthV1y + cs_[k] * SouthV2y));
							double ckMplusz = (coef_[k + (M_ - 2) * M_].z + scal
									* (cc_[k] * SouthV1z + cs_[k] * SouthV2z));

							// compute c[k,M]
							double ckMx = (coef_[M_ * (M_ - 1) + 1].x - spline2MFunc_[5 * DISCRETIZATIONSAMPLINGRATE]
									* (ckMplusx + coef_[k + (M_ - 2) * M_].x))
									/ spline2MFunc_[3 * DISCRETIZATIONSAMPLINGRATE];
							double ckMy = (coef_[M_ * (M_ - 1) + 1].y - spline2MFunc_[5 * DISCRETIZATIONSAMPLINGRATE]
									* (ckMplusy + coef_[k + (M_ - 2) * M_].y))
									/ spline2MFunc_[3 * DISCRETIZATIONSAMPLINGRATE];
							double ckMz = (coef_[M_ * (M_ - 1) + 1].z - spline2MFunc_[5 * DISCRETIZATIONSAMPLINGRATE]
									* (ckMplusz + coef_[k + (M_ - 2) * M_].z))
									/ spline2MFunc_[3 * DISCRETIZATIONSAMPLINGRATE];

							basisFactor = aux * splinePrimeMFunc_[index_u];
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

						int index_i = (2 * i + DISCRETIZATIONSAMPLINGRATE
								* (3 - 2 * k) + MR2_)
								% MR2_;
						if (index_i >= 0 && index_i < NR2_) {
							basisFactor1 = splineMFunc_[index_i]; //bSpline1LUT_[index_i];
						} else {
							basisFactor1 = 0;
						}

						int index_j = 2 * j + DISCRETIZATIONSAMPLINGRATE * (3 - 2 * l);
						if (index_j >= 0 && index_j < NR2_) {
							basisFactor2 = spline2MFunc_[index_j];//bSpline2LUT_[index_j];
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
			double z = (zmax_ - zmin_) * ((double) l) / ((double) M_) + zmin_;
			for (int k = 0; k < M_; k++) {
				coef_[k + (l - 1) * M_] = new Snake3DNode(
						splineCoefs[k][l + 1].x, splineCoefs[k][l + 1].y, z);
			}
		}

		int index_0 = DISCRETIZATIONSAMPLINGRATE * 3;
		double factor0 = 0;
		if (index_0 >= 0 && index_0 < NR2_) {
			factor0 = spline2MFunc_[index_0]; //bSpline2LUT_[index_0];
		}
		int index_1 = DISCRETIZATIONSAMPLINGRATE * 5;
		double factor1 = 0;
		if (index_1 >= 0 && index_1 < NR2_) {
			factor1 = spline2MFunc_[index_1];//bSpline2LUT_[index_1];
		}
		double factorp1 = 0;
		if (index_1 >= 0 && index_1 < NR2_) {
			factorp1 = splinePrime2MFunc_[index_1];//bSpline2DerivativeLUT_[index_1];
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

			xV1N += (cs_[(k + 1) % M_]
					* (splineCoefs[k][0].x - splineCoefs[k][2].x) - cs_[k]
					* (splineCoefs[(k + 1) % M_][0].x - splineCoefs[(k + 1)
							% M_][2].x))
					/ (cs_[(k + 1) % M_] * cc_[k] - cs_[k]
							* cc_[(k + 1) % M_]);
			yV1N += (cs_[(k + 1) % M_]
					* (splineCoefs[k][0].y - splineCoefs[k][2].y) - cs_[k]
					* (splineCoefs[(k + 1) % M_][0].y - splineCoefs[(k + 1)
							% M_][2].y))
					/ (cs_[(k + 1) % M_] * cc_[k] - cs_[k]
							* cc_[(k + 1) % M_]);

			xV2N += (cc_[(k + 1) % M_]
					* (splineCoefs[k][0].x - splineCoefs[k][2].x) - cc_[k]
					* (splineCoefs[(k + 1) % M_][0].x - splineCoefs[(k + 1)
							% M_][2].x))
					/ (cc_[(k + 1) % M_] * cs_[k] - cc_[k]
							* cs_[(k + 1) % M_]);
			yV2N += (cc_[(k + 1) % M_]
					* (splineCoefs[k][0].y - splineCoefs[k][2].y) - cc_[k]
					* (splineCoefs[(k + 1) % M_][0].y - splineCoefs[(k + 1)
							% M_][2].y))
					/ (cc_[(k + 1) % M_] * cs_[k] - cc_[k]
							* cs_[(k + 1) % M_]);

			xV1S += (cs_[(k + 1) % M_]
					* (splineCoefs[k][M_ + 2].x - splineCoefs[k][M_].x) - cs_[k]
					* (splineCoefs[(k + 1) % M_][M_ + 2].x - splineCoefs[(k + 1)
							% M_][M_].x))
					/ (cs_[(k + 1) % M_] * cc_[k] - cs_[k]
							* cc_[(k + 1) % M_]);
			yV1S += (cs_[(k + 1) % M_]
					* (splineCoefs[k][M_ + 2].y - splineCoefs[k][M_].y) - cs_[k]
					* (splineCoefs[(k + 1) % M_][M_ + 2].y - splineCoefs[(k + 1)
							% M_][M_].y))
					/ (cs_[(k + 1) % M_] * cc_[k] - cs_[k]
							* cc_[(k + 1) % M_]);

			xV2S += (cc_[(k + 1) % M_]
					* (splineCoefs[k][M_ + 2].x - splineCoefs[k][M_].x) - cc_[k]
					* (splineCoefs[(k + 1) % M_][M_ + 2].x - splineCoefs[(k + 1)
							% M_][M_].x))
					/ (cc_[(k + 1) % M_] * cs_[k] - cc_[k]
							* cs_[(k + 1) % M_]);
			yV2S += (cc_[(k + 1) % M_]
					* (splineCoefs[k][M_ + 2].y - splineCoefs[k][M_].y) - cc_[k]
					* (splineCoefs[(k + 1) % M_][M_ + 2].y - splineCoefs[(k + 1)
							% M_][M_].y))
					/ (cc_[(k + 1) % M_] * cs_[k] - cc_[k]
							* cs_[(k + 1) % M_]);
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
