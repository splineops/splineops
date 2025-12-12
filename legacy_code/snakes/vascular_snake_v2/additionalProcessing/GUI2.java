package plugins.big.vascular.additionalProcessing;

import icy.main.Icy;
import icy.sequence.Sequence;
import ij.IJ;

import java.awt.Frame;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JDialog;

import plugins.big.vascular.steerable3DFilter.SurfaceDetector;
import plugins.big.vascular.utils.Const;
import plugins.big.vascular.utils.DistanceTransform2;
import additionaluserinterface.GridPanel;
import additionaluserinterface.WalkBar;

public class GUI2 extends JDialog implements Runnable, ActionListener{
	
	private WalkBar			walk					= new WalkBar("(c) 2013 EPFL, Biomedical Imaging Group", true, false, true);
		
	private Thread			thread					= null;
	private JButton			job;	
	
	private JButton 		bnSkullStrip1			= new JButton("Skull strip 1");
	private JButton 		bnDistBorder			= new JButton("Adjust");
	private JButton 		bnSkullStrip2			= new JButton("Skull strip 2");
	private JButton 		bnSkullStrip3			= new JButton("Skull strip 3");
	private JButton 		bnAttractorImage		= new JButton("Attractor Image");
	private JButton 		bnHystStrip				= new JButton("HystImage Strip");
	private JButton			bnFinalize				= new JButton("Finalize");
	private JButton			bnAllInOne 				= new JButton("All in One");
	private JButton			bnApplyMask				= new JButton("Apply Mask");
	private JButton			bnCalcMesh				= new JButton("Calc Mesh");
	private JButton			bnDT					= new JButton("Triangulate");
	private JButton			bnRefine				= new JButton("Refine");
	private JButton			bnNMSLarge				= new JButton("NMSLarge");
	private JButton 		bnLevelSet 				= new JButton("Level Set");
	private JButton			bnBinarizeDist			= new JButton("Binarize Dist-Map");
	private JButton			bnNifti					= new JButton("Save Nifti");
	private JButton			bnNormalize				= new JButton("Normalize");
	
	private JCheckBox		chkSteps				= new JCheckBox("show steps", false);
	
	private GridPanel 		pnMain	 				= new GridPanel(false, 2);
	
	
	
	/**
	 * Constructor
	 */
	public GUI2(){
		super(new Frame(), "Additional Processing");			
		doDialog();				
	}
	
	/**
	 * doDialog
	 */
	private void doDialog(){
		pnMain.place(0, 0, bnSkullStrip1);
		pnMain.place(1, 0, bnDistBorder);
		pnMain.place(2, 0, bnSkullStrip2);
		pnMain.place(3, 0, bnSkullStrip3);
		pnMain.place(4, 0, bnAttractorImage);
		pnMain.place(5, 0, bnHystStrip);
		pnMain.place(6, 0, bnFinalize);
		pnMain.place(7, 0, bnAllInOne);
		pnMain.place(7, 1, chkSteps);
		pnMain.place(8, 0, bnApplyMask);
		pnMain.place(9, 0, bnCalcMesh);
		pnMain.place(10, 0, bnDT);
		pnMain.place(11,  0, bnRefine);
		pnMain.place(12, 0, bnNMSLarge);
		pnMain.place(13, 0, bnLevelSet);
		pnMain.place(14, 0, bnBinarizeDist);
		pnMain.place(15, 0, bnNifti);
		pnMain.place(16, 0, bnNormalize);
		
		bnSkullStrip1.addActionListener(this);
		bnDistBorder.addActionListener(this);
		bnSkullStrip2.addActionListener(this);
		bnSkullStrip3.addActionListener(this);
		bnAttractorImage.addActionListener(this);
		bnHystStrip.addActionListener(this);
		bnFinalize.addActionListener(this);
		bnAllInOne.addActionListener(this);
		bnApplyMask.addActionListener(this);
		bnCalcMesh.addActionListener(this);
		bnDT.addActionListener(this);
		bnRefine.addActionListener(this);
		bnNMSLarge.addActionListener(this);
		bnLevelSet.addActionListener(this);
		bnBinarizeDist.addActionListener(this);
		bnNifti.addActionListener(this);
		bnNormalize.addActionListener(this);
		
		chkSteps.addActionListener(this);		
		
		add(pnMain);
		setResizable(true);
		pack();
		setVisible(true);
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		
		if (e.getSource() == chkSteps)
			Const.GUI2_chkSteps = chkSteps.isSelected();
		
		if (e.getSource() == bnBinarizeDist){			
			
			if (thread == null) {
				job = (JButton)e.getSource();
				thread = new Thread(this);
				thread.setPriority(Thread.MIN_PRIORITY);
				thread.start();
			}
		}	
		
		if (e.getSource() == bnNifti){			
			
			if (thread == null) {
				job = (JButton)e.getSource();
				thread = new Thread(this);
				thread.setPriority(Thread.MIN_PRIORITY);
				thread.start();
			}
		}
		
		if (e.getSource() == bnNormalize){			
			
			if (thread == null) {
				job = (JButton)e.getSource();
				thread = new Thread(this);
				thread.setPriority(Thread.MIN_PRIORITY);
				thread.start();
			}
		}
		
		if (e.getSource() == bnLevelSet){			
			
			if (thread == null) {
				job = (JButton)e.getSource();
				thread = new Thread(this);
				thread.setPriority(Thread.MIN_PRIORITY);
				thread.start();
			}
		}
		
		if (e.getSource() == bnNMSLarge){			
			
			if (thread == null) {
				job = (JButton)e.getSource();
				thread = new Thread(this);
				thread.setPriority(Thread.MIN_PRIORITY);
				thread.start();
			}
		}	
		
		if (e.getSource() == bnRefine){			
			
			if (thread == null) {
				job = (JButton)e.getSource();
				thread = new Thread(this);
				thread.setPriority(Thread.MIN_PRIORITY);
				thread.start();
			}
		}	
		
		
		if (e.getSource() == bnDT){			
			
			if (thread == null) {
				job = (JButton)e.getSource();
				thread = new Thread(this);
				thread.setPriority(Thread.MIN_PRIORITY);
				thread.start();
			}
		}		
		
		if (e.getSource() == bnCalcMesh){			
			
			if (thread == null) {
				job = (JButton)e.getSource();
				thread = new Thread(this);
				thread.setPriority(Thread.MIN_PRIORITY);
				thread.start();
			}
		}
		
		if (e.getSource() == bnApplyMask){			
			
			if (thread == null) {
				job = (JButton)e.getSource();
				thread = new Thread(this);
				thread.setPriority(Thread.MIN_PRIORITY);
				thread.start();
			}
		}
		
		if (e.getSource() == bnAllInOne){			
			
			if (thread == null) {
				job = (JButton)e.getSource();
				thread = new Thread(this);
				thread.setPriority(Thread.MIN_PRIORITY);
				thread.start();
			}
		}
		
		
		if (e.getSource() == bnSkullStrip1){			
			
			if (thread == null) {
				job = (JButton)e.getSource();
				thread = new Thread(this);
				thread.setPriority(Thread.MIN_PRIORITY);
				thread.start();
			}
		}
		
		if (e.getSource() == bnSkullStrip2){			
			
			if (thread == null) {
				job = (JButton)e.getSource();
				thread = new Thread(this);
				thread.setPriority(Thread.MIN_PRIORITY);
				thread.start();
			}
		}
		
		if (e.getSource() == bnSkullStrip3){			
			
			if (thread == null) {
				job = (JButton)e.getSource();
				thread = new Thread(this);
				thread.setPriority(Thread.MIN_PRIORITY);
				thread.start();
			}
		}
		
		if (e.getSource() == bnAttractorImage){			
			
			if (thread == null) {
				job = (JButton)e.getSource();
				thread = new Thread(this);
				thread.setPriority(Thread.MIN_PRIORITY);
				thread.start();
			}
		}
		
		if (e.getSource() == bnHystStrip){			
			
			if (thread == null) {
				job = (JButton)e.getSource();
				thread = new Thread(this);
				thread.setPriority(Thread.MIN_PRIORITY);
				thread.start();
			}
		}
		
		if (e.getSource() == bnFinalize){			
			
			if (thread == null) {
				job = (JButton)e.getSource();
				thread = new Thread(this);
				thread.setPriority(Thread.MIN_PRIORITY);
				thread.start();
			}
		}
		
		
		
		if (e.getSource() == bnDistBorder){			
			
			if (thread == null) {
				job = (JButton)e.getSource();
				thread = new Thread(this);
				thread.setPriority(Thread.MIN_PRIORITY);
				thread.start();
			}
		}
	}

	@Override
	public void run() {
		
		//bnBinarizeDist
		if (job == bnBinarizeDist){
			System.out.println("bn Binarize Distance");	
			
			BinarizeDistMap bdm = new BinarizeDistMap();
			bdm.process();
		}
		
		//bnLevelSet
		if (job == bnLevelSet){
			System.out.println("bn Level Set");				
			IJ.error("not implemented in this version");
			
			//LevelSet levelSet = new LevelSet();
			//levelSet.process();
		
		}	
		
		//bnNMSLarge
		if (job == bnNMSLarge){
			System.out.println("bn NMS large");				
			
			double sigma = 2;
			
			SurfaceDetector surfaceDetector = new SurfaceDetector(sigma, walk, null);
			surfaceDetector.detect();		
		}	
		
		
		
		//bnCalcMesh
		if (job == bnCalcMesh){
			System.out.println("bn Calc Mesh");				
			
			//Sequence inputSeq = Icy.getMainInterface().getSequences("Binary_PointMask_DS").get(0);	
			//Triangulation triangulation = new Triangulation(inputSeq);
			//triangulation.run();			
		}
		
		//bnDT
		if (job == bnDT){
			System.out.println("bn triangulation");
			Sequence inputSeq = Icy.getMainInterface().getSequences("Binary_PointMask_DS").get(0);
			//Sequence inputSeq = Icy.getMainInterface().getSequences("SurfaceMask_DS").get(0);
			
			//DT dT = new DT(inputSeq);
			
			IJ.error("not implemented in this version");
			/*
			Triangulate tri = new Triangulate(inputSeq);		
			tri.surfaceReconstruction();	
			*/
			
			
			//tri.saveAnalyze();
			
			//Triangulate tri = new Triangulate();
			//tri.surfaceMesher();			
			//tri.surfaceReconstruction();		
			
			//tri.delaunay3D();	
			
			//tri.CGAL_regularTriangulation();
		}
		
		//bnRefine
		if (job == bnRefine){
			System.out.println("bn Refine");
			
			//Sequence inputSeq = Icy.getMainInterface().getSequences("Binary_PointMask_DS").get(0);
			//Refine refine = new Refine(inputSeq);
			
			Refine2 refine2 = new Refine2();
			refine2.run();
		}
		
		//bnNifti
		if (job == bnNifti){
			System.out.println("bn Nifti");
			Nifti_Writer niftiWriter = new Nifti_Writer();
			niftiWriter.run(null);
		}	
		
		
		//bnApplyMask
		if (job == bnApplyMask){
			System.out.println("bn Apply Mask");
			ApplyMask applyMask = new ApplyMask();
			applyMask.process();
		}
		
		//bnAllInOne
		if (job == bnAllInOne){
			System.out.println("bn AllInOne");
			AllInOne allInOne = new AllInOne(false);
			allInOne.process();
		}
		
		
		//bnFinalize
		if (job == bnFinalize){
			System.out.println("bn Finalize");
			Finalize finalize = new Finalize();
			finalize.process();
			
		}
		
		//bnSkullStrip1
		if (job == bnSkullStrip1){
			System.out.println("bnSS 1");
			//SkullStrip1  skullStrip1 = new SkullStrip1();
			//skullStrip1.process();
			HystStrip2 hystStrip2 = new HystStrip2();
			hystStrip2.process();
		}
		
		//bnSkullStrip2
		if (job == bnSkullStrip2){
			System.out.println("bnSS 2");
			//SkullStrip2  skullStrip2 = new SkullStrip2();
			//skullStrip2.process();
			HystStrip3 hystStrip3 = new HystStrip3();
			hystStrip3.process();
		}
		
		//bnSkullStrip3
		if (job == bnSkullStrip3){
			System.out.println("bnSS 3");
			SkullStrip3  skullStrip3 = new SkullStrip3();
			skullStrip3.process();
		}
		
		//bnAttractorImage
		if (job == bnAttractorImage){
			System.out.println("bn Attractor Image");
			//AttractorImage attractorImage = new AttractorImage();
			//attractorImage.process();
			
			//Sequence inputSeq = Icy.getMainInterface().getSequences("binaryImage_DS").get(0);
			DistanceTransform2 dT = new DistanceTransform2();//new DistanceTransform();
			dT.run();
		}
		
		//bnHystStrip2
		if (job == bnHystStrip){
			System.out.println("bn HystImage Strip");
			HystStrip3 hystStrip3 = new HystStrip3();
			hystStrip3.process();
		}
		
		//bnDistBorder
		if (job == bnDistBorder){
			System.out.println("bnAdjust");
			Adjust adjust = new Adjust();
			adjust.process();			
		}
		
		//bnNormalize
		if (job == bnNormalize){
			System.out.println("bnNormalize");
			Normalize normalize = new Normalize();
			normalize.process();			
		}
		
		
		
		thread = null;
	}

	

}
