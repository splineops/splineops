package plugins.big.vascular.additionalProcessing;

import icy.canvas.Canvas3D;
import icy.canvas.IcyCanvas;
import icy.main.Icy;
import icy.painter.Overlay;
import icy.sequence.Sequence;

import java.awt.Graphics2D;

import javax.vecmath.Point3d;

import plugins.big.vascular.steerable3DFilter.ImageVolume;
import plugins.big.vascular.utils.Const;
import vtk.vtkActor;
import vtk.vtkDataArray;
import vtk.vtkExtractEdges;
import vtk.vtkIdList;
import vtk.vtkPoints;
import vtk.vtkPolyData;
import vtk.vtkPolyDataMapper;
import vtk.vtkPolyDataNormals;

public class Refine extends Overlay{
	
	Sequence binarySeq;
	private vtkPolyData mesh;
	private vtkActor finalActor = new vtkActor();
	vtkPolyDataMapper map;
	
	public Refine(Sequence binarySeq){	
		super("refine");
		this.binarySeq = binarySeq;
		mesh = Const.mesh;			
	}
	
	public void run(){	
		System.out.println("refinement start.");
		
		Sequence boundarySeq = Icy.getMainInterface().getSequences("Step 4: Hysteresis Thresholding").get(0);
		double[][][] boundaryVol = (new ImageVolume(boundarySeq)).volume;
		
		int nx = boundarySeq.getSizeX();
		int ny = boundarySeq.getSizeY();
		int nz = boundarySeq.getSizeZ();
		
		 // Generate normals
	    vtkPolyDataNormals normalGenerator = new vtkPolyDataNormals();	
	    normalGenerator.SetInputData(mesh);	
	    normalGenerator.ComputePointNormalsOn();
	    normalGenerator.ComputeCellNormalsOff();	    
	    normalGenerator.SetSplitting(0);
	    normalGenerator.AutoOrientNormalsOn();
	    normalGenerator.FlipNormalsOn();
	    normalGenerator.ConsistencyOn();	    
	    normalGenerator.Update();	    
	    /*
	    // Optional settings
	    normalGenerator->SetFeatureAngle(0.1);
	    normalGenerator->SetSplitting(1);
	    normalGenerator->SetConsistency(0);
	    normalGenerator->SetAutoOrientNormals(0);
	    normalGenerator->SetComputePointNormals(1);
	    normalGenerator->SetComputeCellNormals(0);
	    normalGenerator->SetFlipNormals(0);
	    normalGenerator->SetNonManifoldTraversal(1);
	    */
	 
	    vtkPolyData pointNormals = normalGenerator.GetOutput();
	    
	    //vtkDataArray normalData = pointNormals.GetPointData().GetNormals();
	    vtkDataArray normalData = mesh.GetPointData().GetNormals();
	    
		
		int nPts = mesh.GetNumberOfPoints();
		System.out.println("nPts = " + nPts + ", ptNormals = " + pointNormals.GetNumberOfPoints());				
		
		
		Point3d pt, ptN, meanP, s;
		double theta;
		boolean[] stop = new boolean[nPts];
		int nIts = 2;
		double MU = 4;//0.1;//0.02;		
		double mu = MU;
		
		/*
		for (int i=0; i<10; i++){
			pt = new Point3d(mesh.GetPoint(i));
			ptN = new Point3d(normalData.GetComponent(0, i), normalData.GetComponent(1, i), normalData.GetComponent(2, i));
			
			System.out.println("pt.x = " + pt.x + ", pt.y = " + pt.y + ", pt.z = " + pt.z);
			System.out.println("ptN.x = " + ptN.x + ", ptN.y = " + ptN.y + ", ptN.z = " + ptN.z);
			
			System.out.println("***");
		}
		*/
		
		
		for (int p = 0; p < nPts; p++) 			
			stop[p] = false;
		
		vtkPolyData meshUpdated = mesh;
		vtkPolyData origMesh = mesh;
		
		vtkPoints pointsUpdated = mesh.GetPoints();
		vtkIdList[] verticesConnections = getConnectedVertices();	
		
		
		Point3d center = new Point3d(nx/2, ny/2, nz/2);		
		
		
		for (int i=1; i <= nIts; i++){			
			
			
			mesh = meshUpdated;
			
			/*
			normalGenerator.RemoveAllInputs();
			normalGenerator.SetInput(mesh);
			
			//
			normalGenerator.ComputePointNormalsOn();
			normalGenerator.ComputeCellNormalsOff();	    
			normalGenerator.SetSplitting(0);
			normalGenerator.AutoOrientNormalsOn();
			normalGenerator.FlipNormalsOn();
			normalGenerator.ConsistencyOn();
			//
			
			normalGenerator.Update();
			pointNormals = normalGenerator.GetOutput();
			normalData = pointNormals.GetPointData().GetNormals();
			*/
			
					
			
			
			for (int p=0; p<nPts; p++)
			if (!stop[p]){				
				
				pt = new Point3d(mesh.GetPoint(p));
				ptN = new Point3d(normalData.GetComponent(0, p), normalData.GetComponent(1, p), normalData.GetComponent(2, p));
								
				meanP = meanPoint(verticesConnections[p]);				
				s = minus(pt, meanP);				
				
				double normS = Math.sqrt(s.x*s.x + s.y*s.y + s.z*s.z);
				Point3d sNormal = new Point3d(s.x/normS, s.y/normS, s.z/normS);
				
				
				
				
				if (p== 0){
					System.out.println("ptN = " + ptN.x + ", " + ptN.y + ", " + ptN.z);
					System.out.println("mesh p = " + pt.x + ", " + pt.y + ", " + pt.z);
					System.out.println("s = " + sNormal.x + ", " + sNormal.y + ", " + sNormal.z);
				}
				
				
				
				// local curvature
				theta = Math.acos((ptN.x*sNormal.x + ptN.y*sNormal.y + ptN.z*sNormal.z));					
				
				/*
				double angle = 0.2*Math.PI; //0.2				
				if (theta > angle){
					mu = 0.01*mu;
				}	
				*/				
				
				/*
				if (theta < 0.5*Math.PI)
					mu *= -1;
				*/
							
				
				double[] ptUpdated = {pt.x - mu*ptN.x, pt.y - mu*ptN.y, pt.z - mu*ptN.z};
				
				//
				if (center.distance(new Point3d(ptUpdated)) < center.distance(pt)){					
					ptUpdated[0] = pt.x + mu*ptN.x;
					ptUpdated[1] = pt.y + mu*ptN.y;
					ptUpdated[2] = pt.z + mu*ptN.z;				
				}					
				//				
				
				pointsUpdated.InsertPoint(p, ptUpdated);
				
				mu = MU;					
				
				double interP = Const.getInterpolatedPixel(ptUpdated[0], ptUpdated[1], ptUpdated[2],
						nx, ny, nz, boundaryVol);
				
				double totalDist = (new Point3d(origMesh.GetPoint(p))).distance((new Point3d(ptUpdated)));						
				
				if (interP != 0 || totalDist > 20)
					stop[p] = true;				
				
			}	
			
			
			System.out.println("it = " + i);// + ", NPtsUpd = " + pointsUpdated.GetActualMemorySize());
			//System.out.println("ptU[100] = " + pointsUpdated.GetPoint(100)[0] + ", " + pointsUpdated.GetPoint(100)[1] +
			//		", " + pointsUpdated.GetPoint(100)[2]);
			
			meshUpdated.SetPoints(pointsUpdated);
		}	
		
		
		binarySeq.addOverlay(this);
		System.out.println("refinement done.");
	}
	
	private Point3d minus(Point3d p1, Point3d p2){
		return new Point3d(p1.x-p2.x, p1.y-p2.y, p1.z-p2.z);
	}
	
	private Point3d meanPoint(vtkIdList verticeConnections){
		double x = 0;
		double y = 0;
		double z = 0;
		
		int nIds = verticeConnections.GetNumberOfIds();
		for (int i=0; i<nIds; i++){
			int id = verticeConnections.GetId(i);
			double[] p =  mesh.GetPoint(id);
			x += p[0];
			y += p[1];
			z += p[2];
		}
		
		return new Point3d(x/nIds, y/nIds, z/nIds);
	}
	
	
	private vtkIdList[] getConnectedVertices(){	
		
		vtkExtractEdges extractEdges = new vtkExtractEdges();
		extractEdges.SetInputData(mesh);
		extractEdges.Update();	
		
		vtkPolyData edgeMesh = extractEdges.GetOutput();
		
		System.out.println("edgeMesh nPoints = " + edgeMesh.GetNumberOfPoints());
		
		int nPts = mesh.GetNumberOfPoints();
		System.out.println("nPts = " + nPts);
		
		vtkIdList[] verticesConnections = new vtkIdList[nPts];
		for (int i=0; i<nPts; i++)
			verticesConnections[i] = getConnectedVertices(edgeMesh, i);			
			
		return verticesConnections;		
	}

	private vtkIdList getConnectedVertices(vtkPolyData edgeMesh, int id){
		vtkIdList connectedVertices = new vtkIdList();
		
		// get all cells that vertex 'id' is part of 
		vtkIdList cellIdList = new vtkIdList();
		edgeMesh.GetPointCells(id, cellIdList);
		
		for (int i=0; i<cellIdList.GetNumberOfIds(); i++){
			vtkIdList pointIdList = new vtkIdList();
			edgeMesh.GetCellPoints(cellIdList.GetId(i), pointIdList);
			
			if (pointIdList.GetId(0) != id)
				connectedVertices.InsertNextId(pointIdList.GetId(0));
			else
				connectedVertices.InsertNextId(pointIdList.GetId(1));
		}
		return connectedVertices;
	}
	
	@Override
	public void paint(Graphics2D g, Sequence sequence, IcyCanvas canvas){
		
		if (canvas instanceof Canvas3D){
			
			Canvas3D c3d = (Canvas3D) canvas;			
			c3d.getRenderer().SetGlobalWarningDisplay(0);			
			c3d.getRenderer().RemoveActor(finalActor);		
			
			map = new vtkPolyDataMapper(); 
			map.SetInputData(mesh); 
			
			finalActor = new vtkActor();
			finalActor.SetMapper(map);	
			
			finalActor.GetProperty().SetRepresentationToWireframe();			
			
			c3d.getRenderer().AddActor(finalActor);			
		}
	}
	
}
