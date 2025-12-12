package plugins.big.vascular.vascularGui;

//import icy.gui.dialog.MessageDialog;
//import icy.main.Icy;
import icy.plugin.PluginDescriptor;
import icy.plugin.PluginLauncher;
import icy.plugin.PluginLoader;
import icy.sequence.Sequence;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Frame;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JDialog;
import javax.swing.JLabel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;










import plugins.big.vascular.BIGVascular;
import plugins.big.vascular.additionalProcessing.GUI2;
import plugins.big.vascular.steerable3DFilter.ImageVolume;
import plugins.big.vascular.steerable3DFilter.SurfaceDetector;
import plugins.big.vascular.utils.ActiveSequenceDisplay;
import plugins.big.vascular.utils.Const;
import plugins.big.vascular.utils.HysteresisThresholding;
import plugins.big.vascular.utils.LocalNormalization;
//import plugins.schmitter.brainsegmentation.steerable3Dfilter.ImageVolume;
import additionaluserinterface.GridPanel;
import additionaluserinterface.WalkBar;

public class Gui extends JDialog implements Runnable, ActionListener, ChangeListener{
	
	private WalkBar			walk					= new WalkBar("(c) 2013 EPFL, Biomedical Imaging Group", true, false, true);
	private JTextArea		logWindow				= new JTextArea("", 10, 30);	
	
	private Thread			thread					= null;
	private JButton			job;	
	
	private GridPanel 		pnMain	 				= new GridPanel(false, 2);
	private GridPanel 		pnSteerableFilter 		= new GridPanel();
	private GridPanel 		pnDisplay		 		= new GridPanel();
	private GridPanel 		pnLogWindow 			= new GridPanel();
	private GridPanel 		pnSnake3D	 			= new GridPanel();
	
	private JComboBox		cmbShowResponse			= new JComboBox(new String[] {"Local normalization", "3D steerable filter", "NMS", "Hysteresis thresholding", 
			"Active Sequence"});
	
	private SpinnerDouble 	spnLocMean				= new SpinnerDouble(80, 0.1, 500, 10);
	private SpinnerDouble 	spnLocVariance			= new SpinnerDouble(0.1, 0.001, 500, 10);
	
	private JButton 		bnGui2					= new JButton("Additional Processing");
	private JButton			bnSnake3D				= new JButton("Snake 3D");
	private JButton			bnRotate				= new JButton("Rotate");
	private JButton			bnEqualize				= new JButton("Equalize");
	private JButton			bnLocNorm				= new JButton("Normalize");
	private JButton			bnRunSteerableFilter	= new JButton("RUN 3D filter");
	private JButton			bnHyst					= new JButton("Hysteresis");
	private JButton			bnXResponse				= new JButton("show X Response");
	private JButton			bnYResponse				= new JButton("show Y Response");
	private JButton			bnZResponse				= new JButton("show Z Response");
	
	private JCheckBox		chkHyst					= new JCheckBox();
	
	private SpinnerDouble 	spnSigma				= new SpinnerDouble(2, 0, 100, 1);
	private SpinnerDouble 	spnHystLow				= new SpinnerDouble(10, 0, 1000, 10);
	private SpinnerDouble 	spnHystHigh				= new SpinnerDouble(45, 0, 1000, 10);
	
	//private boolean boolSteerable = false;	
	private HysteresisThresholding hysteresisThresholding;
	private LocalNormalization localNormalization;
	private SurfaceDetector surfaceDetector;
	
	
	
	/**
	 * Constructor
	 */
	public Gui(Sequence sequence){
		super(new Frame(), "Vascular Segmentation");			
		doDialog();		
		logWindow.setBackground(new Color(139,26,26));
		logWindow.setForeground(new Color(85,26,139));
	}
	
	
	
	/**
	 * doDialog
	 */
	private void doDialog(){		
		
		// ***************************
		// steerable filter panel (process)
		// ***************************		
		
		pnSteerableFilter.setBackground(new Color(198,226,255));		
		
		bnLocNorm.setBackground(Color.CYAN);
		bnLocNorm.setForeground(Color.BLUE);
		bnLocNorm.setOpaque(true);
		
		JLabel lbLocNorm = new JLabel();
		lbLocNorm.setText("<html> <u>1. Local Normalization</u>");
		pnSteerableFilter.place(0, 0, lbLocNorm);
		pnSteerableFilter.place(1, 0, new JLabel("local mean:"));
		pnSteerableFilter.place(1, 1, spnLocMean);
		pnSteerableFilter.place(2, 0, new JLabel("local variance:"));
		pnSteerableFilter.place(2, 1, spnLocVariance);	
		pnSteerableFilter.place(2, 2, bnLocNorm);		
		
		bnRunSteerableFilter.setPreferredSize(new Dimension(130, 50));
		bnRunSteerableFilter.setBackground(Color.CYAN);
		bnRunSteerableFilter.setForeground(Color.BLUE);
		bnRunSteerableFilter.setOpaque(true);		
		
		JLabel lbProcess = new JLabel();
		lbProcess.setText("<html> <u>2. 3D steerable filter</u>");
		pnSteerableFilter.place(3, 0, lbProcess);
		pnSteerableFilter.place(4, 0, new JLabel("sigma:"));		
		pnSteerableFilter.place(4, 1, spnSigma);
		pnSteerableFilter.place(4, 2, bnRunSteerableFilter);
		
		bnHyst.setBackground(Color.CYAN);
		bnHyst.setForeground(Color.BLUE);
		bnHyst.setOpaque(true);
		
		JLabel lbHist = new JLabel();
		lbHist.setText("<html> <u>3. Hysteresis thresholding</u>");
		pnSteerableFilter.place(5, 0, lbHist);
		pnSteerableFilter.place(6, 0, new JLabel("Threshold low:"));
		pnSteerableFilter.place(6, 1, spnHystLow);	
		pnSteerableFilter.place(6, 2, new JLabel("active sequence"));
		pnSteerableFilter.place(6, 3, chkHyst);
		pnSteerableFilter.place(7, 0, new JLabel("Threshold high:"));
		pnSteerableFilter.place(7, 1, spnHystHigh);	
		pnSteerableFilter.place(7, 2, bnHyst);	
	
		// ***************************
		// Snake3D panel
		// ***************************	
		bnSnake3D.setPreferredSize(new Dimension(150, 50));
		bnSnake3D.setBackground(Color.BLACK);
		bnSnake3D.setForeground(Color.GRAY);
		bnSnake3D.setOpaque(true);		
		
		bnGui2.setPreferredSize(new Dimension(200, 50));
		bnGui2.setBackground(Color.BLACK);
		bnGui2.setForeground(Color.GRAY);
		bnGui2.setOpaque(true);	
		
		pnSnake3D.place(0, 0, bnSnake3D);
		pnSnake3D.place(0,  2,  bnGui2);
		
		
		// ***************************
		// Display panel
		// ***************************
		JLabel lbDisplay = new JLabel();
		lbDisplay.setText("<html> <u>Display</u>");
		pnDisplay.place(0, 0, lbDisplay);
		pnDisplay.place(1, 0, new JLabel("Response:"));
		pnDisplay.place(1, 1, cmbShowResponse);
		pnDisplay.place(1, 2, bnXResponse);
		pnDisplay.place(2, 2, bnYResponse);
		pnDisplay.place(3, 2, bnZResponse);	
		pnDisplay.place(2, 1, bnRotate);
		pnDisplay.place(3, 1, bnEqualize);
		
		// ***************************
		// log window panel
		// ***************************		
		JScrollPane logWindowScrollPane = new JScrollPane(logWindow);		
		pnLogWindow.place(2, 0, 5, 1, logWindowScrollPane);
		
		
		// ***************************
		// main panel
		// ***************************			
		pnMain.place(2, 0, pnSteerableFilter);
		pnMain.place(3, 0, pnSnake3D);		
		pnMain.place(4, 0, pnDisplay);
		pnMain.place(5, 0, logWindowScrollPane);
		pnMain.place(6, 0, walk);		
		
		
		// ***************************
		// Listeners
		// ***************************
		
		bnGui2.addActionListener(this);
		bnSnake3D.addActionListener(this);
		bnRotate.addActionListener(this);
		bnEqualize.addActionListener(this);
		bnLocNorm.addActionListener(this);
		bnRunSteerableFilter.addActionListener(this);
		bnHyst.addActionListener(this);
		spnSigma.addChangeListener(this);
		bnXResponse.addActionListener(this);
		bnYResponse.addActionListener(this);
		bnZResponse.addActionListener(this);
		cmbShowResponse.addActionListener(this);
		spnHystLow.addChangeListener(this);
		spnHystHigh.addChangeListener(this);
		spnLocMean.addChangeListener(this);
		spnLocVariance.addChangeListener(this);
		
		walk.getButtonClose().addActionListener(this);

		
		
		// ***************************
		// build main panel
		// ***************************				
		add(pnMain);
		setResizable(true);
		pack();
		setVisible(true);	
		
	}

	@Override
	public void run() {
		
		//bnHist
		if (job == bnHyst){
			
			System.out.println("" + chkHyst.isSelected());
			
			if (!chkHyst.isSelected())
				hysteresisThresholding = new HysteresisThresholding(surfaceDetector.nmsResult, spnHystLow.get(), spnHystHigh.get(), walk, logWindow);
			else				
				hysteresisThresholding = new HysteresisThresholding((new ImageVolume(true)), spnHystLow.get(), spnHystHigh.get(), walk, logWindow);
			
			hysteresisThresholding.process();
			
		}
		
		//bnLocNorm
		if (job == bnLocNorm){
			localNormalization = new LocalNormalization(logWindow, walk, spnLocMean.get(), spnLocVariance.get());
			localNormalization.normalize();
		}
		
		//bnRunSteerableFilter
		if (job == bnRunSteerableFilter){
			surfaceDetector = new SurfaceDetector(spnSigma.get(), walk, logWindow);
			surfaceDetector.detect();
			
			//boolSteerable = true;
		}
		
		// bnXResponse
		if (job == bnXResponse){			
						
			if (cmbShowResponse.getSelectedIndex() == 0)
				localNormalization.showX("local normalization X");
			if (cmbShowResponse.getSelectedIndex() == 1)
				surfaceDetector.showXFilter();
			if (cmbShowResponse.getSelectedIndex() == 2)
				surfaceDetector.showXNMS();
			if (cmbShowResponse.getSelectedIndex() == 3)
				hysteresisThresholding.showX("hysteresis thresholding X, tl=" + spnHystLow.get() + ", th=" + spnHystHigh.get());
			if (cmbShowResponse.getSelectedIndex() == 4)
				(new ActiveSequenceDisplay()).showX(Const.ORIG_X);
		}
		
		// bnYResponse
		if (job == bnYResponse){
			if (cmbShowResponse.getSelectedIndex() == 0)
				localNormalization.showY("local normalization Y");			
			if (cmbShowResponse.getSelectedIndex() == 1)
				surfaceDetector.showYFilter();
			if (cmbShowResponse.getSelectedIndex() == 2)
				surfaceDetector.showYNMS();	
			if (cmbShowResponse.getSelectedIndex() == 3)
				hysteresisThresholding.showY("hysteresis thresholding Y, tl=" + spnHystLow.get() + ", th=" + spnHystHigh.get());
			if (cmbShowResponse.getSelectedIndex() == 4)
				(new ActiveSequenceDisplay()).showY(Const.ORIG_Y);
		}
		
		// bnZResponse
		if (job == bnZResponse){
			
			/*
			if (!boolSteerable){
				MessageDialog.showDialog("Filtering must be done first!");
				thread = null;
				return;
			}
			*/
			if (cmbShowResponse.getSelectedIndex() == 0)
				localNormalization.showZ("local normalization Z");
			if (cmbShowResponse.getSelectedIndex() == 1)
				surfaceDetector.showZFilter();
			if (cmbShowResponse.getSelectedIndex() == 2)
				surfaceDetector.showZNMS();		
			if (cmbShowResponse.getSelectedIndex() == 3)
				hysteresisThresholding.showZ("hysteresis thresholding Z, tl=" + spnHystLow.get() + ", th=" + spnHystHigh.get());
			if (cmbShowResponse.getSelectedIndex() == 4)
				(new ActiveSequenceDisplay()).showZ(Const.ORIG_Z);
		}		
		
		thread = null;
	}

	@Override
	public void stateChanged(ChangeEvent e) {
		if (e.getSource() == spnSigma){
			logWindow.append("\nsource = spnSigma");
			logWindow.setCaretPosition(logWindow.getDocument().getLength());
		}
		
	}
	

	@Override
	public void actionPerformed(ActionEvent e) {		
		
		
		if (e.getSource() == bnGui2){
			GUI2 gui2 = new GUI2();
		}
		
		if (e.getSource() == bnSnake3D){
					
			BIGVascular bigSnake3D = new BIGVascular();
			bigSnake3D.run();								
		}
		
		if (e.getSource() == bnRotate){		
			
			PluginDescriptor pluginDescriptor = PluginLoader.getPlugin("plugins.spop.rotation3D.StackRotationByAngle");			
			PluginLauncher.start(pluginDescriptor);				
		}
		
		if (e.getSource() == bnEqualize){		
			
			PluginDescriptor pluginDescriptor = PluginLoader.getPlugin("plugins.spop.clahe.Clahe");			
			PluginLauncher.start(pluginDescriptor);			
		}
		
		
		if (e.getSource() == bnRunSteerableFilter){
			logWindow.append("\nsource = bnRunSteerableFilter");
			logWindow.setCaretPosition(logWindow.getDocument().getLength());
			
			if (thread == null) {
				job = (JButton)e.getSource();
				thread = new Thread(this);
				thread.setPriority(Thread.MIN_PRIORITY);
				thread.start();
			}
		}
		
		
		if (e.getSource() == bnHyst){
			logWindow.append("\nsource = bnHysteresis");
			logWindow.setCaretPosition(logWindow.getDocument().getLength());
			
			if (thread == null) {
				job = (JButton)e.getSource();
				thread = new Thread(this);
				thread.setPriority(Thread.MIN_PRIORITY);
				thread.start();
			}
		}
		
		if (e.getSource() == bnLocNorm){
			logWindow.append("\nsource = bnNormalization");
			logWindow.setCaretPosition(logWindow.getDocument().getLength());
			
			if (thread == null) {
				job = (JButton)e.getSource();
				thread = new Thread(this);
				thread.setPriority(Thread.MIN_PRIORITY);
				thread.start();
			}
		}
		
		if (e.getSource() == bnXResponse){
			logWindow.append("\nsource = bnXResponse");
			logWindow.setCaretPosition(logWindow.getDocument().getLength());
			
			if (thread == null) {
				job = (JButton)e.getSource();
				thread = new Thread(this);
				thread.setPriority(Thread.MIN_PRIORITY);
				thread.start();
			}
		}
		
		if (e.getSource() == bnYResponse){
			logWindow.append("\nsource = bnYResponse");
			logWindow.setCaretPosition(logWindow.getDocument().getLength());
			
			if (thread == null) {
				job = (JButton)e.getSource();
				thread = new Thread(this);
				thread.setPriority(Thread.MIN_PRIORITY);
				thread.start();
			}
		}
		
		if (e.getSource() == bnZResponse){
			logWindow.append("\nsource = bnZResponse");
			logWindow.setCaretPosition(logWindow.getDocument().getLength());
			
			if (thread == null) {
				job = (JButton)e.getSource();
				thread = new Thread(this);
				thread.setPriority(Thread.MIN_PRIORITY);
				thread.start();
			}
		}

		
	}

}
