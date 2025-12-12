package plugins.big.vascular.utils;

import java.util.ArrayList;

import icy.gui.frame.progress.ProgressFrame;
import icy.image.IcyBufferedImage;
import icy.main.Icy;
import icy.plugin.abstract_.Plugin;
import icy.sequence.Sequence;
import icy.type.collection.array.Array1DUtil;
import plugins.big.bigsnakeutils.icy.snake3D.Snake3DNode;
import plugins.big.bigsnakeutils.icy.snake3D.Snake3DScale;
import plugins.big.vascular.core.ImageLUTContainer;
import plugins.big.vascular.keeper.SnakeKeeper;
import plugins.big.vascular.roi.Anchor3D;
import plugins.big.vascular.steerable3DFilter.*;
import vtk.vtkDataArray;
import vtk.vtkPolyData;

public class Const extends Plugin{
	public static String LOC_NORM_X = "local normalization X";
	public static String LOC_NORM_Y = "local normalization Y";
	public static String LOC_NORM_Z = "local normalization Z";
	
	public static String ATTRACTOR_IMAGE_Z = "Attractor image Z";
	
	public static String HYST_X = "";
	public static String HYST_Y = "";
	public static String HYST_Z = "";	
	
	public static String ORIG_X = "orig X";
	public static String ORIG_Y = "orig Y";
	public static String ORIG_Z = "orig Z";
	
	public static String HYST_STRIP_Z = "Hyst strip Z";
	
	public static String NMS_Z = "";		
	
	public static OrthoSlice localNormX;
	public static OrthoSlice localNormY;
	public static OrthoSlice localNormZ;
	
	public static boolean locNormXInit = false;
	public static boolean locNormYInit = false;
	public static boolean locNormZInit = false;
	
	public static OrthoSlice hystX;
	public static OrthoSlice hystY;
	public static OrthoSlice hystZ;
	
	public static boolean hystXInit = false;
	public static boolean hystYInit = false;
	
	public static OrthoSlice origX;
	public static OrthoSlice origY;
	public static OrthoSlice origZ;
	
	public static boolean origXInit = false;
	public static boolean origYInit = false;
	public static boolean origZInit = false;	
	
	public static boolean orthoViewSelected = false;
	public static boolean snakeCopiedSelected = false;
	
	public static Snake3DNode[] coef_ = null;
	public static double snakeVolume = 0;
	
	public static boolean segmentBrainMP2RAGE = false;
	
	public static Sequence LOC_NORM_Z_SEQ = null;
	public static Sequence ATTRACTOR_IMAGE_Z_SEQ = null;
	public static Sequence ORIG_Z_SEQ = null;
	
	public static Sequence DIST_TRANSFORM = null;
	public static Sequence BINARY_SEQ = null;
	
	public static String SURFACE_Z = "response Z";
	
	public static ImageLUTContainer LUT_MP2RAGE = null;
	public static ImageLUTContainer LUT_BinaryImage = null;
	public static ImageLUTContainer LUT_Binary = null;	
	
	public static FilterResponse surfaceDetection = null;
	
	public static SnakeKeeper keeper = null;
	
	public static boolean GUI2_chkSteps = false;
	
	public static Sequence binaryMask = null;
	
	public static SurfaceDetector surfaceDetectorAllInOne = null;
	
	public static double[][][] binaryThresholdingOutput = null;
	
	public static vtkPolyData mesh;
	public static vtkDataArray normals;
	
	// update snake painters on ortho slices
	public static void updateOrthoSlices(ArrayList<Snake3DScale> scales_, ArrayList<Anchor3D> nodes_){
		if (Const.hystXInit)
			Const.hystX.scalesChanged(scales_, nodes_);
		if (Const.hystYInit)
			Const.hystY.scalesChanged(scales_, nodes_);
		
		if (Const.locNormXInit)
			Const.localNormX.scalesChanged(scales_, nodes_);
		if (Const.locNormYInit)
			Const.localNormY.scalesChanged(scales_, nodes_);
		if (Const.locNormZInit)
			Const.localNormZ.scalesChanged(scales_, nodes_);
		
		// orig images
		if (Const.origXInit)
			Const.origX.scalesChanged(scales_, nodes_);
		if (Const.origYInit)
			Const.origY.scalesChanged(scales_, nodes_);
		if (Const.origZInit)
			Const.origZ.scalesChanged(scales_, nodes_);
	}
	
	public static double[][] put3DImageIn2DArray(Sequence seq){
		int depth = seq.getSizeZ();		
		
		double[][] imageDataArray = new double[depth][];
		
		for (int z = 0; z < depth; z++) {
			IcyBufferedImage singleFrameImage = seq.getImage(0, z);
			imageDataArray[z] = Array1DUtil.arrayToDoubleArray(
					singleFrameImage.getDataXY(0), singleFrameImage
							.getDataType_().isSigned());
		}
		
		return imageDataArray;
	}
	
	public static void calcLUTBrainMP2RAGE(){
		System.out.println("Const.calcLUTBrainMP2RAGE");		
		
		if (Icy.getMainInterface().getSequences("Distance Transform").get(0) != null){
			//LOC_NORM_Z_SEQ = Icy.getMainInterface().getSequences(Const.LOC_NORM_Z).get(0);
			//ATTRACTOR_IMAGE_Z_SEQ = Icy.getMainInterface().getSequences(Const.ATTRACTOR_IMAGE_Z).get(0);
			//ORIG_Z_SEQ = Icy.getMainInterface().getSequences(Const.ORIG_Z).get(0);			
			
			ProgressFrame progressFrame = new ProgressFrame("Creating extra LUT for MP2RAGE");
			
			//ImageLUTContainer imageLUTContainer = new ImageLUTContainer(LOC_NORM_Z_SEQ);	
			//ImageLUTContainer imageLUTContainer = new ImageLUTContainer(ATTRACTOR_IMAGE_Z_SEQ);	
			
			BINARY_SEQ = Icy.getMainInterface().getSequences("binaryImage").get(0);
			ImageLUTContainer imageLUTContainer2 = new ImageLUTContainer(BINARY_SEQ);		
			System.out.println("Const.buildLUT binary image");
			imageLUTContainer2.setSigma(2.0); // SIGMA
			imageLUTContainer2.buildLUTs();			
			LUT_Binary = imageLUTContainer2; // binaryImage
			System.out.println("CONST = " + LUT_Binary.getPreintegratedFilteredImageDataArray()[30][30]);
			
			DIST_TRANSFORM = Icy.getMainInterface().getSequences("Distance Transform").get(0);
			ImageLUTContainer imageLUTContainer = new ImageLUTContainer(DIST_TRANSFORM);			
			imageLUTContainer.buildLUTs();			
			LUT_MP2RAGE = imageLUTContainer; // distanceTransform			
			
			progressFrame.close();
		}
	}
	
	public static double getInterpolatedPixel(double x, double y, double z, int nx, int ny, int nz, double[][][] volume) {	

		
		if (x < 0) {
	    int periodx = 2*nx - 2;				
			while (x < 0) x += periodx;		// Periodize
			if (x >= nx)  x = periodx - x;	// Symmetrize
		}
		else if ( x >= nx) {
	    int periodx = 2*nx - 2;				
			while (x >= nx) x -= periodx;	// Periodize
			if (x < 0)  x = - x;			// Symmetrize
		}

		if (y < 0) {
	    int periody = 2*ny - 2;				
			while (y < 0) y += periody;		// Periodize
			if (y >= ny)  y = periody - y;	// Symmetrize
		}
		else if (y >= ny) {
	    int periody = 2*ny - 2;				
			while (y >= ny) y -= periody;	// Periodize
			if (y < 0)  y = - y;			// Symmetrize
		}
		
		if (z < 0) {
	    int periodz = 2*nz - 2;				
			while (z < 0) z += periodz;		// Periodize
			if (z >= nz)  z = periodz - z;	// Symmetrize
		}
		else if (z >= nz) {
	    int periodz = 2*nz - 2;				
			while (z >= nz) z -= periodz;	// Periodize
			if (z < 0)  z = - z;			// Symmetrize
		}	
		
		
		int i = (int)x;
		int j = (int)y;
		int k = (int)z;		
		
		double dx = x - (double)((int)x);
		double dy = y - (double)((int)y);
		double dz = z - (double)((int)z);
		
		
	  int di, dj, dk;
		
		if(i >= nx-1) { // border mirror condition
			di = -1;
		}	else {
			di = 1;
		}
		
		if(j >= ny-1) { // border mirror condition
			dj = -1;
		}	else {
			dj = 1;
		}	
		
		if(k >= nz-1) { // border mirror condition
			dk = -1;
		}	else {
			dk = 1;
		}	
		
		
		double v000 = volume[i][j][k];
		double v100 = volume[i+di][j][k];
		double v010 = volume[i][j+dj][k];
		double v110 = volume[i+di][j+dj][k];
		
		double v001 = volume[i][j][k+dk];
		double v101 = volume[i+di][j][k+dk];
		double v011 = volume[i][j+dj][k+dk];
		double v111 = volume[i+di][j+dj][k+dk];
		

		double interpolation = ((v000*(1.0-dx) + v100*dx)*(1-dy) + (v010*(1.0-dx) + v110*dx)*dy)*(1-dz)
												 + ((v001*(1.0-dx) + v101*dx)*(1-dy) + (v011*(1.0-dx) + v111*dx)*dy)*dz;
		
		
		return interpolation;
	}

}
