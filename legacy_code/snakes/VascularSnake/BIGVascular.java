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
package plugins.big.vascular;

import icy.gui.component.BorderedPanel;
import icy.gui.frame.IcyFrameAdapter;
import icy.gui.frame.IcyFrameEvent;
import icy.gui.frame.progress.AnnounceFrame;
import icy.gui.frame.progress.ProgressFrame;
import icy.gui.main.ActiveSequenceListener;
import icy.gui.main.ActiveViewerListener;
import icy.gui.main.GlobalSequenceListener;
import icy.gui.main.GlobalViewerListener;
import icy.gui.viewer.Viewer;
import icy.gui.viewer.ViewerEvent;
import icy.main.Icy;
import icy.plugin.abstract_.PluginActionable;
import icy.resource.ResourceUtil;
import icy.roi.ROI;
import icy.roi.ROI3D;
import icy.sequence.Sequence;
import icy.sequence.SequenceEvent;
import icy.sequence.SequenceUtil;
import icy.type.rectangle.Rectangle3D;
import icy.type.rectangle.Rectangle5D;

import java.awt.CardLayout;
import java.awt.Container;
import java.awt.GridLayout;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

import javax.swing.AbstractAction;
import javax.swing.BoxLayout;
import javax.swing.Icon;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JComponent;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.KeyStroke;
import javax.swing.SwingConstants;
import javax.vecmath.Point3d;

import jxl.write.WriteException;
import jxl.write.biff.RowsExceededException;

import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Workbook;

import plugins.adufour.blocks.lang.Block;
import plugins.adufour.blocks.lang.BlockDescriptor;
import plugins.adufour.blocks.util.VarList;
import plugins.adufour.vars.lang.Var;
import plugins.adufour.vars.lang.VarBoolean;
import plugins.adufour.vars.lang.VarDouble;
import plugins.adufour.vars.lang.VarEnum;
import plugins.adufour.vars.lang.VarInteger;
import plugins.adufour.vars.lang.VarSequence;
import plugins.adufour.vars.lang.VarWorkbook;
import plugins.big.vascular.io.IOXML;
import plugins.big.bigsnakeutils.icy.ellipsoid.Ellipsoid3D;
import plugins.big.bigsnakeutils.icy.ellipsoid.EllipsoidROI3D;
import plugins.big.bigsnakeutils.icy.snake3D.Snake3DNode;
import plugins.big.bigsnakeutils.icy.snake3D.Snake3DOptimizer;
import plugins.big.bigsnakeutils.icy.snake3D.Snake3DPowellOptimizer;
import plugins.big.vascular.core.DisplaySettings;
import plugins.big.vascular.core.ImageLUTContainer;
import plugins.big.vascular.core.Snake3DClipboard;
import plugins.big.vascular.gui.DisplaySettingsPane;
import plugins.big.vascular.gui.ImageSettingsPane;
import plugins.big.vascular.gui.SnakeInteractionPane;
import plugins.big.vascular.gui.PlugInFrame;
import plugins.big.vascular.gui.SnakeSettingsPane;
import plugins.big.vascular.gui.ToolTipsMessages;
import plugins.big.vascular.io.InputFormat;
import plugins.big.vascular.keeper.KeepersList;
import plugins.big.vascular.keeper.SnakeKeeper;
import plugins.big.vascular.roi.ActionPlane;
import plugins.big.vascular.roi.Snake3DROI;
import plugins.big.vascular.roi.SnakeEditMode;
import plugins.big.vascular.snake.CylinderSnake;
import plugins.big.vascular.snake.CylinderSnakeEnergyType;
import plugins.big.vascular.snake.CylinderSnakeParameters;
import plugins.big.vascular.snake.CylinderSnakeTargetType;

/**
 * Snake plug-in.
 * 
 * @version October 30, 2013
 * 
 * @author Nicolas Chenouard (nicolas.chenouard@gmail.com)
 * @author Ricard Delgado-Gonzalo (ricard.delgado@gmail.com)
 * @author Emrah Bostan (emrah.bostan@gmail.com)
 * @author Clement Marti (clement.marti@centraliens.net)
 * @author Christophe Gaudet-Blavignac (chrisgaubla@gmail.com)
 * 
 */
public class BIGVascular extends PluginActionable implements Block,
		ActionListener, ActiveSequenceListener, ActiveViewerListener,
		GlobalSequenceListener, GlobalViewerListener, KeyListener {

	/** Reference to the main plug-in class. */
	private BIGVascular mainPlugin_ = null;

	// ----------------------------------------------------------------------------
	// SNAKES

	/** Map that associates each open image with a list of snakes. */
	private final HashMap<Sequence, KeepersList> keepersListTable_ = new HashMap<Sequence, KeepersList>();
	/** Lock used to preserve synchronization among different open images. */
	private final Lock keepersListLock_ = new ReentrantLock();
	/**
	 * Index that counts the number of snakes added since the plug-in was
	 * created.
	 */
	private int nSnakes_ = 0;

	// ----------------------------------------------------------------------------
	// IMAGES AND VIEWERS

	/** Map that associates each open image with its corresponding LUTs. */
	private final HashMap<Sequence, ImageLUTContainer> imageLUTs_ = new HashMap<Sequence, ImageLUTContainer>();
	/** Map that associates each viewer with its corresponding KeyListeners. */
	private final HashMap<Viewer, Boolean> hasKeyListenerTable_ = new HashMap<Viewer, Boolean>();

	// ----------------------------------------------------------------------------
	// GUI

	/** Main frame of the plug-in. */
	private final PlugInFrame plugInMainFrame_ = new PlugInFrame(this);
	/** Pane that contains snake interactions settings */
	private SnakeInteractionPane snakeInteractionPane_ = null;
	/** Panel that contains the settings of snake interactions. */
	private final JPanel snakeInteractionPanel_ = new BorderedPanel();
	/** Pane that contains the image related settings. */
	private ImageSettingsPane imageSettingsPane_ = null;
	/** Panel that contains the settings of the focused image. */
	private final JPanel imageSettingsPanel_ = new BorderedPanel();
	/** Pane that contains the snake related settings. */
	private SnakeSettingsPane sphereSnakeSettingsPane_ = null;
	/** Panel that contains the settings of the active snake. */
	private final JPanel snakeParametersPanel_ = new BorderedPanel();
	/** Panel that contains the display settings of the snakes in the viewers. */
	private final JPanel displaySettingsPanel_ = new BorderedPanel();
	/** Pane that contains the display related settings. */
	private DisplaySettingsPane displaySettingsPane_ = null;
	/** Label in which the tooltips are shown. */
	private final JLabel toolTipMessagesLabel_ = new JLabel(
			ToolTipsMessages.getToolTipMessage(0), SwingConstants.CENTER);

	// ----------------------------------------------------------------------------
	// ACTION BUTTONS

	/**
	 * Button that sets the interaction mode with the snake to translation mode
	 * (control points or whole snake).
	 */
	private final JButton moveSnakeButton_ = new JButton("");
	/**
	 * Button that sets the interaction mode with the snake to resizing/scaling
	 * mode.
	 */
	private final JButton resizeSnakeButton_ = new JButton("");
	/** Button that sets the interaction mode with the snake to rotation mode. */
	private final JButton rotateSnakeButton_ = new JButton("");
	/**
	 * Button that fires the process to create a new snake from the interface
	 * parameters.
	 */
	private final JButton createSnakeButton_ = new JButton("");
	/**
	 * Button that fires the process to optimize the active snake of the focused
	 * image.
	 */
	private final JButton optimizeSnakeButton_ = new JButton("");
	/**
	 * Button that fires the process to optimize all of the snakes of the
	 * focused image.
	 */
	private final JButton optimizeAllSnakesButton_ = new JButton("");
	/**
	 * Button that fires the process of deletion of the active snake of the
	 * focused image.
	 */
	private final JButton deleteSnakeButton_ = new JButton("");

	private final JComboBox activePlaneComboBox_ = new JComboBox();

	/** Size of the icons in the menu. */
	private static int ICONSIZE = 20;

	// ----------------------------------------------------------------------------
	// OTHER

	/** Determines the way the user can interact with the snake. */
	private SnakeEditMode editingMode_ = null;
	/**
	 * Determines the plane in which the scaling action is applied, and the
	 * plane perpendicular to the axis in which the rotation action is applied.
	 */
	private ActionPlane actionPlane_ = null;

	/** Stores the information of the snake when using the COPY command. */
	private final Snake3DClipboard clipboard_ = new Snake3DClipboard();

	/** List of running threads. */
	private final ArrayList<Thread> threadList_ = new ArrayList<Thread>();

	// ============================================================================
	// BLOCK FIELDS

	// ----------------------------------------------------------------------------
	// INPUT

	/** Input image for the snake plug-in within the protocol environment. */
	private final VarSequence inputSequenceBlock_ = new VarSequence("Input",
			null);
	private final VarDouble sigmaBlock_ = new VarDouble("Smoothing", 10);
	private final VarDouble gammaBlock_ = new VarDouble("Elasticity", 1);
	private final VarEnum<CylinderSnakeTargetType> targetBrightnessBlock_ = new VarEnum<CylinderSnakeTargetType>(
			"Target brightness", CylinderSnakeTargetType.DARK);
	private final VarInteger MBlock_ = new VarInteger("Control points", 3);
	private final VarEnum<CylinderSnakeEnergyType> energyTypeBlock_ = new VarEnum<CylinderSnakeEnergyType>(
			"Energy type", CylinderSnakeEnergyType.REGION);
	private final VarDouble alphaBlock_ = new VarDouble("Alpha", 0);
	private final VarInteger maxIterationsBlock_ = new VarInteger(
			"Max iterations", 100);
	private final VarBoolean isImmortalBlock_ = new VarBoolean("Immortal",
			false);
	private final VarEnum<InputFormat> inputFormatROI_ = new VarEnum<InputFormat>(
			"Input snake format", InputFormat.ROI_Array);
	private final VarWorkbook inputSnakeWorkbook_ = new VarWorkbook(
			"Snake input workbook", (Workbook) null);
	private final Var<ROI[]> roiArray = new Var<ROI[]>("Array of ROIs",
			ROI[].class);

	// ----------------------------------------------------------------------------
	// OUTPUT

	private final VarWorkbook outputSnakeWorkbook_ = new VarWorkbook(
			"Snake output workbook", "outputSnakeWorkbook_");

	// ============================================================================
	// PUBLIC METHODS

	/** Method executed when launching the plug-in. */
	@Override
	public void run() {
		if (Thread.currentThread().getStackTrace()[2].getClassName().equals(
				BlockDescriptor.class.getName())) {
			runBlock();
		} else {
			runStandalone();
		}
	}

	// ----------------------------------------------------------------------------

	/**
	 * Method executed when launching the plug-in as a block within the
	 * protocols.
	 */
	private void runBlock() {
		nSnakes_++;
		inputSequenceBlock_.getValue().removeAllROI();
		// fill columns names in outputSnakeWorkbook_ inputSnakeWorkbook_
		String[] snakeParamsName = { "Snake ID", "Area", "Time", "Energy ype",
				"Energy value", "Centroid x", "Centroid y", "Centroid z",
				"Number of Nodes" };
		Row columnTitles = outputSnakeWorkbook_.getValue()
				.getSheet("outputSnakeWorkbook_").createRow(0);
		for (int j = 0; j < snakeParamsName.length; j++) {
			columnTitles.createCell(j);
			columnTitles.getCell(j).setCellValue(snakeParamsName[j]);
		}

		// FIXME it assumes that only the first channel is used
		ImageLUTContainer imageLUTContainer = new ImageLUTContainer();
		imageLUTContainer.setSigma(sigmaBlock_.getValue());
		imageLUTContainer.setSequence(SequenceUtil.getSubSequence(
				inputSequenceBlock_.getValue(), new Rectangle5D.Integer(0, 0,
						0, 0, 0, inputSequenceBlock_.getValue().getWidth(),
						inputSequenceBlock_.getValue().getHeight(),
						inputSequenceBlock_.getValue().getSizeZ(), 1, 1)));
		imageLUTContainer.buildLUTs();

		CylinderSnakeParameters snakeParameters = new CylinderSnakeParameters(
				maxIterationsBlock_.getValue(), MBlock_.getValue(),
				alphaBlock_.getValue(), gammaBlock_.getValue(),
				isImmortalBlock_.getValue(), targetBrightnessBlock_.getValue(),
				energyTypeBlock_.getValue());

		double r = Math.min(Math.min(imageLUTContainer.getImageWidth() / 5.0,
				imageLUTContainer.getImageHeight() / 5.0), imageLUTContainer
				.getImageDepth() / 5.0);

		int length = 0;
		int startIndex = 0;
		if (inputFormatROI_.getValue().equals(InputFormat.ROI_Array)) {
			length = roiArray.getValue().length;
		} else {
			length = inputSnakeWorkbook_
					.getValue()
					.getSheetAt(
							inputSnakeWorkbook_.getValue().getFirstVisibleTab())
					.getLastRowNum() + 1;
			startIndex = 1;
		}
		int previoustime = 0;
		int timeROI = 0;
		int index = 1;
		// loop over ROIs
		for (int i = startIndex; i < length; i++) {
			boolean inTime = false;
			double minX = 0;
			double maxX = 0;
			double minY = 0;
			double maxY = 0;
			double minZ = 0;
			double maxZ = 0;
			double a = 0;
			double b = 0;
			double c = 0;
			Point3d roiCenter = new Point3d();
			if (inputFormatROI_.getValue().equals(
					InputFormat.Workbook_of_ROI_parameters)) {
				
			} else if (inputFormatROI_.getValue().equals(InputFormat.ROI_Array)) {
				if (roiArray.getValue()[i] instanceof EllipsoidROI3D) {

					Ellipsoid3D descriptor = ((EllipsoidROI3D) roiArray
							.getValue()[i]).getDescriptor();
					timeROI = descriptor.getT();
					inTime = timeROI == previoustime;
					if (!inTime) {
						previoustime = timeROI;
						imageLUTContainer.setSequence(SequenceUtil
								.getSubSequence(inputSequenceBlock_.getValue(),
										new Rectangle5D.Integer(0, 0, 0,
												previoustime, 0,
												inputSequenceBlock_.getValue()
														.getWidth(),
												inputSequenceBlock_.getValue()
														.getHeight(),
												inputSequenceBlock_.getValue()
														.getSizeZ(), 1, 1)));
						imageLUTContainer.buildLUTs();
					}
					roiCenter = new Point3d(descriptor.x0, descriptor.y0,
							descriptor.z0);
					a = descriptor.a;
					b = descriptor.b;
					c = descriptor.c;

				} else {
					inTime = timeROI == previoustime;
					ROI3D preOpRoi = (ROI3D) roiArray.getValue()[i];
					timeROI = preOpRoi.getT();
					inTime = timeROI == previoustime;
					if (!inTime) {
						previoustime = timeROI;
						imageLUTContainer.setSequence(SequenceUtil
								.getSubSequence(inputSequenceBlock_.getValue(),
										new Rectangle5D.Integer(0, 0, 0,
												previoustime, 0,
												inputSequenceBlock_.getValue()
														.getWidth(),
												inputSequenceBlock_.getValue()
														.getHeight(),
												inputSequenceBlock_.getValue()
														.getSizeZ(), 1, 1)));
						imageLUTContainer.buildLUTs();
					}
					Rectangle3D bounds = preOpRoi.getBounds3D();

					a = bounds.getSizeX();
					b = bounds.getSizeY();
					c = bounds.getSizeZ();
					minX = bounds.getX();
					maxX = minX + a;
					minY = bounds.getY();
					maxY = minY + b;
					minZ = bounds.getZ();
					maxZ = minZ + c;
					roiCenter = new Point3d((maxX + minX) / 2,
							(maxY + minY) / 2, (maxZ + minZ) / 2);
					Rectangle3D boundaries = preOpRoi.getBounds3D();

					minX = boundaries.getX();
					maxX = minX + boundaries.getSizeX();
					minY = boundaries.getY();
					maxY = minY + boundaries.getSizeY();
					minY = boundaries.getZ();
					maxZ = minZ + boundaries.getSizeZ();
					roiCenter = new Point3d((maxX + minX) / 2.0,
							(maxY + minY) / 2.0, (maxZ + minZ) / 2.0);
					
				}
			}
			// define a new snake
			CylinderSnake snake = new CylinderSnake(imageLUTContainer,
					snakeParameters);

			Snake3DNode[] nodes = snake.getNodes();
			Point3d centroid = snake.getCentroid();

			// adjust parameters
			snake.dilateX(a / r);
			snake.dilateX(b / r);
			snake.dilateX(c / r);
			for (int j = 0; j < nodes.length; j++) {
				nodes[j].x += roiCenter.x - centroid.x;
				nodes[j].y += roiCenter.y - centroid.y;
				nodes[j].z += roiCenter.z - centroid.z;
			}
			snake.setNodes(nodes);
			// optimizing snake
			Snake3DOptimizer optimizer = new Snake3DPowellOptimizer();
			final Snake3DNode[] youngSnake = snake.getNodes();
			final Snake3DNode[] X = new Snake3DNode[youngSnake.length];
			for (int k = 0; k < X.length; k++) {
				X[k] = new Snake3DNode(youngSnake[k].x, youngSnake[k].y,
						youngSnake[k].z, youngSnake[k].isFrozen(),
						youngSnake[k].isHidden());
			}
			snake.setNodes(X);
			optimizer.optimize(snake, X);
			nodes = snake.getNodes();

			Object[] snakeParams = { nSnakes_, snake.getArea(), previoustime,
					snake.getSnakeParameters().getEnergyType().toString(),
					snake.energy(), snake.getCentroid().x,
					snake.getCentroid().y, snake.getCentroid().z,
					2 * snake.getNumNodes() };
			// check if snake parameters are all well defined (non null or
			// NaN or infinity)
			boolean validSnake = true;
			for (Object snakeParam : snakeParams) {
				if (snakeParam instanceof Double) {
					validSnake = validSnake && (snakeParam != null)
							&& (!((Double) snakeParam).isNaN())
							&& (!((Double) snakeParam).isInfinite());
				}
			}
			for (int j = 0; j < snake.getNumNodes(); j++) {
				validSnake = validSnake && (!((Double) nodes[j].x).isNaN())
						&& (!((Double) nodes[j].x).isInfinite());
				validSnake = validSnake && (!((Double) nodes[j].y).isNaN())
						&& (!((Double) nodes[j].y).isInfinite());
				validSnake = validSnake && (!((Double) nodes[j].z).isNaN())
						&& (!((Double) nodes[j].z).isInfinite());

			}
			if (validSnake) {
				Row resultRow = outputSnakeWorkbook_.getValue()
						.getSheet("outputSnakeWorkbook_").createRow(index);
				index++;
				nSnakes_++;
				for (int j = 0; j < snakeParams.length; j++) {
					resultRow.createCell(j);
					if (snakeParams[j] instanceof String) {
						resultRow.getCell(j).setCellValue(
								(String) snakeParams[j]);
					} else if (snakeParams[j] instanceof Double) {
						resultRow.getCell(j).setCellValue(
								(int) ((Double) snakeParams[j]).doubleValue());
					} else if (snakeParams[j] instanceof Integer) {
						resultRow.getCell(j).setCellValue(
								(Integer) snakeParams[j]);
					} else {
						System.out.println("unknown type");
					}
				}
				for (int j = 0; j < snake.getNumNodes(); j++) {
					int cellNum = 3 * j + snakeParams.length;
					resultRow.createCell(cellNum);
					resultRow.getCell(cellNum).setCellValue(nodes[j].x);
					resultRow.createCell(cellNum + 1);
					resultRow.getCell(cellNum + 1).setCellValue(nodes[j].y);
					resultRow.createCell(cellNum + 2);
					resultRow.getCell(cellNum + 2).setCellValue(nodes[j].z);
				}
				new Snake3DROI(snake, new SnakeKeeper(
						inputSequenceBlock_.getValue(), snake, this)); //

			}
		}
	}

	// ----------------------------------------------------------------------------

	/** Method executed when launching the plug-in in standalone mode. */
	private void runStandalone() {
		mainPlugin_ = this;
		snakeInteractionPane_ = new SnakeInteractionPane("Snake interactions",
				this);
		imageSettingsPane_ = new ImageSettingsPane("Image", imageLUTs_);
		sphereSnakeSettingsPane_ = new SnakeSettingsPane("Snake", this);
		displaySettingsPane_ = new DisplaySettingsPane("Dislay options", this);

		Viewer existedViewer = getActiveViewer();
		if (existedViewer != null) {
			addKeyListenerToViewer(existedViewer);
		}

		Container mainPane = plugInMainFrame_.getContentPane();
		mainPane.setLayout(new BoxLayout(mainPane, BoxLayout.PAGE_AXIS));

		snakeInteractionPanel_.setLayout(new CardLayout());
		snakeInteractionPanel_.add("snakeInteractionPane_",
				snakeInteractionPane_);
		mainPane.add(snakeInteractionPanel_);

		imageSettingsPanel_.setLayout(new CardLayout());
		imageSettingsPanel_.add("imageSettingsPane_", imageSettingsPane_);
		mainPane.add(imageSettingsPanel_);

		snakeParametersPanel_.setLayout(new CardLayout());
		snakeParametersPanel_.add("sphereSnakeSettingsPane_",
				sphereSnakeSettingsPane_);
		mainPane.add(snakeParametersPanel_);

		displaySettingsPanel_.setLayout(new CardLayout());
		displaySettingsPanel_.add("displaySettingsPane_", displaySettingsPane_);
		mainPane.add(displaySettingsPanel_);

		JPanel interactionPanelHeader = new JPanel();
		interactionPanelHeader.setLayout(new BoxLayout(interactionPanelHeader,
				BoxLayout.LINE_AXIS));
		JLabel snakeEditLabel = new JLabel("Snake editing");
		interactionPanelHeader.add(snakeEditLabel);
		mainPane.add(interactionPanelHeader);

		// create the bar of interaction buttons
		JPanel interactionPanel = new JPanel();

		Icon newSnakeIcon = ResourceUtil.getAlphaIcon("plus.png", ICONSIZE);
		createSnakeButton_.setIcon(newSnakeIcon);
		createSnakeButton_.setToolTipText("Add a new snake");

		Icon imgiconMoveSnake = ResourceUtil.getAlphaIcon("cursor_arrow.png",
				ICONSIZE);
		moveSnakeButton_.setIcon(imgiconMoveSnake);
		moveSnakeButton_.setToolTipText("Move the whole snake");
		moveSnakeButton_.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				setEditingMode(SnakeEditMode.MOVE_SNAKE);
			}
		});

		Icon imgiconDilateSnake = ResourceUtil.getAlphaIcon("expand.png",
				ICONSIZE);
		resizeSnakeButton_.setIcon(imgiconDilateSnake);
		resizeSnakeButton_.setToolTipText("Dilate snake");
		resizeSnakeButton_.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				setEditingMode(SnakeEditMode.DILATE_SNAKE);
			}
		});

		Icon imgiconRotateSnake = ResourceUtil.getAlphaIcon("rot_unclock.png",
				ICONSIZE);
		rotateSnakeButton_.setIcon(imgiconRotateSnake);
		rotateSnakeButton_.setToolTipText("Rotate snake");
		rotateSnakeButton_.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				setEditingMode(SnakeEditMode.ROTATE_SNAKE);
			}
		});

		activePlaneComboBox_.addItem("XY");
		activePlaneComboBox_.addItem("YZ");
		activePlaneComboBox_.addItem("XZ");
		activePlaneComboBox_.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				if (e.getSource() == activePlaneComboBox_) {
					switch (activePlaneComboBox_.getSelectedIndex()) {
					case 0:
						setActionPlane(ActionPlane.XY);
						break;
					case 1:
						setActionPlane(ActionPlane.YZ);
						break;
					case 2:
						setActionPlane(ActionPlane.XZ);
						break;
					}
				}
			}
		});

		// Set defaults
		setEditingMode(SnakeEditMode.MOVE_SNAKE);
		setActionPlane(ActionPlane.XY);

		interactionPanel.setLayout(new GridLayout(1, 4));
		interactionPanel.add(createSnakeButton_);
		interactionPanel.add(moveSnakeButton_);
		interactionPanel.add(resizeSnakeButton_);
		interactionPanel.add(rotateSnakeButton_);
		interactionPanel.add(activePlaneComboBox_);

		mainPane.add(interactionPanel);

		// create the action panel
		JPanel actionPanelHeader = new JPanel();
		actionPanelHeader.setLayout(new BoxLayout(actionPanelHeader,
				BoxLayout.LINE_AXIS));
		JLabel actionLabel = new JLabel("Snake actions");
		actionPanelHeader.add(actionLabel);
		mainPane.add(actionPanelHeader);

		JPanel actionPane = new JPanel();
		actionPane.setLayout(new GridLayout(1, 3));

		Icon optimizeActiveSnakeIcon = ResourceUtil.getAlphaIcon(
				"playback_play.png", ICONSIZE);
		optimizeSnakeButton_.setIcon(optimizeActiveSnakeIcon);
		optimizeSnakeButton_.setToolTipText("Optimize active snake");

		Icon optimizeAllSnakesIcon = ResourceUtil.getAlphaIcon(
				"playback_ff.png", ICONSIZE);
		optimizeAllSnakesButton_.setIcon(optimizeAllSnakesIcon);
		optimizeAllSnakesButton_.setToolTipText("Optimize all snakes");

		Icon deleteSnakeIcon = ResourceUtil.getAlphaIcon("trash.png", ICONSIZE);
		deleteSnakeButton_.setIcon(deleteSnakeIcon);
		deleteSnakeButton_.setToolTipText("Remove the active snake");

		actionPane.add(optimizeSnakeButton_);
		actionPane.add(optimizeAllSnakesButton_);
		actionPane.add(deleteSnakeButton_);

		createSnakeButton_.addActionListener(this);
		optimizeSnakeButton_.addActionListener(this);
		optimizeAllSnakesButton_.addActionListener(this);
		deleteSnakeButton_.addActionListener(this);

		// assign DELETE as key binding for deleteSnakeButton
		deleteSnakeButton_.getInputMap(JComponent.WHEN_IN_FOCUSED_WINDOW)
				.put(KeyStroke.getKeyStroke(KeyEvent.VK_DELETE, 0),
						"doDeleteAction");
		deleteSnakeButton_.getActionMap().put("doDeleteAction",
				new AbstractAction() {
					private static final long serialVersionUID = -7619717973444534482L;

					@Override
					public void actionPerformed(ActionEvent e) {
						deleteSnakeButton_.doClick();
					}
				});

		// assign BACKSPACE as key binding for deleteSnakeButton (for MAC users)
		deleteSnakeButton_.getInputMap(JComponent.WHEN_IN_FOCUSED_WINDOW).put(
				KeyStroke.getKeyStroke(KeyEvent.VK_BACK_SPACE, 0),
				"doDeleteAction");
		deleteSnakeButton_.getActionMap().put("doDeleteAction",
				new AbstractAction() {
					private static final long serialVersionUID = -7619717973444534482L;

					@Override
					public void actionPerformed(ActionEvent e) {
						deleteSnakeButton_.doClick();
					}
				});
		mainPane.add(actionPane);

		// create a pane for the selected sequence display
		JPanel sequencePane = new JPanel();
		sequencePane.add(toolTipMessagesLabel_);

		toolTipMessagesLabel_.addMouseListener(new MouseAdapter() {
			@Override
			public void mouseClicked(MouseEvent e) {
				toolTipMessagesLabel_.setText(ToolTipsMessages
						.getToolTipMessage());
			}
		});

		mainPane.add(sequencePane);

		Icy.getMainInterface().addActiveSequenceListener(this);
		Icy.getMainInterface().addActiveViewerListener(this);
		Icy.getMainInterface().addGlobalSequenceListener(this);
		Icy.getMainInterface().addGlobalViewerListener(this);

		plugInMainFrame_.addFrameListener(new IcyFrameAdapter() {
			@Override
			public void icyFrameClosed(IcyFrameEvent e) {
				Icy.getMainInterface()
						.removeActiveSequenceListener(mainPlugin_);
				Icy.getMainInterface().removeActiveViewerListener(mainPlugin_);
				Icy.getMainInterface()
						.removeGlobalSequenceListener(mainPlugin_);
				Icy.getMainInterface().removeGlobalViewerListener(mainPlugin_);
			}
		});

		plugInMainFrame_.pack();

		plugInMainFrame_.addFrameListener(new IcyFrameAdapter() {
			@Override
			public void icyFrameInternalized(IcyFrameEvent e) {
				plugInMainFrame_.pack();
			}

			@Override
			public void icyFrameExternalized(IcyFrameEvent e) {
				plugInMainFrame_.pack();
			}
		});

		addIcyFrame(plugInMainFrame_);
		plugInMainFrame_.center();
		plugInMainFrame_.setVisible(true);
	}

	// ----------------------------------------------------------------------------

	/**
	 * Prepares the plug-in to be terminated. All memory is freed, and auxiliary
	 * plug-ins are closed.
	 */
	public void terminatePlugin() {
		removeAllSnakes();
		removeAllKeyListeners();
	}

	// ----------------------------------------------------------------------------
	// SNAKE CREATION/DESTRUCTION MANAGEMENT

	/**
	 * Returns <code>true</code> if <code>SnakeKeeper</code> passed is the
	 * active one.
	 */
	public boolean isActiveSnake(SnakeKeeper keeper) {
		Sequence focusedSequence = getActiveSequence();
		if (focusedSequence == null) {
			return false;
		}
		keepersListLock_.lock();
		try {
			if (keepersListTable_.containsKey(focusedSequence)) {
				KeepersList keepersList = keepersListTable_
						.get(focusedSequence);
				if (keepersList != null) {
					return keepersList.isActiveSnakeKeeper(keeper);
				}
			}
		} finally {
			keepersListLock_.unlock();
		}
		return false;
	}

	// ----------------------------------------------------------------------------

	/** Makes a particular snake active and responsive to interaction. */
	public void activateSnake(SnakeKeeper keeper) {
		Sequence focusedSequence = getActiveSequence();
		if (focusedSequence == null) {
			return;
		}
		keepersListLock_.lock();
		try {
			if (keepersListTable_.containsKey(focusedSequence)) {
				KeepersList keepersList = keepersListTable_
						.get(focusedSequence);
				if (keepersList != null) {
					boolean success = keepersList.activateSnakeKeeper(keeper);
					sphereSnakeSettingsPane_.setSnakeParameters(keeper
							.getESnakeParameters());
					if (!success) {
						System.err
								.println("activateSnake: The SphereSnakeKeeper could not be activated.");
					}
				}
			}
		} finally {
			keepersListLock_.unlock();
		}
	}

	// ----------------------------------------------------------------------------

	/**
	 * Adds a new snake to the active <code>Sequence</code> and makes it active
	 * and responsive to interactions.
	 */
	private void addAndActivateSnake() {
		Thread thr = new Thread() {
			@Override
			public void run() {
				Sequence focusedSequence = getActiveSequence();
				if (focusedSequence == null) {
					new AnnounceFrame(
							"No image detected. Please, open an image first.");
					return;
				} else if (focusedSequence.getSizeZ() <= 1) {
					new AnnounceFrame(
							"This plugin works only with 3D images. For 2D images, you can use the 2D Active Cells segmentation plug-in.");
					return;
				}

				ProgressFrame progressFrame = new ProgressFrame(
						"Creating a new 3D snake");

				ImageLUTContainer imageLUTContainer = imageLUTs_
						.get(focusedSequence);

				if (imageLUTContainer == null) {
					System.err.println("LUT not found");
					return;
				}

				if (!imageLUTContainer.isLUTUpToDate()) {
					imageLUTContainer.buildLUTs();
				}

				CylinderSnakeParameters snakeParameters = new CylinderSnakeParameters(
						sphereSnakeSettingsPane_.getMaxIterations(),
						sphereSnakeSettingsPane_.getNumControlPoints(),
						sphereSnakeSettingsPane_.getAlpha(),
						sphereSnakeSettingsPane_.getGamma(),
						sphereSnakeSettingsPane_.isImmortal(),
						sphereSnakeSettingsPane_.getTargetBrightness(),
						sphereSnakeSettingsPane_.getEnergyType());

				CylinderSnake snake = new CylinderSnake(imageLUTContainer,
						snakeParameters);
				SnakeKeeper keeper = new SnakeKeeper(
						imageLUTContainer.getOriginalSequence(), snake,
						mainPlugin_);
				nSnakes_++;
				keeper.setID("" + nSnakes_);
				keeper.setSnakeEditMode(getEditingMode());

				DisplaySettings ds = displaySettingsPane_.getDisplaySettings();
				if (ds != null) {
					keeper.setDisplaySettings(ds);
				}

				keepersListLock_.lock();
				try {
					if (keepersListTable_.containsKey(focusedSequence)) {
						KeepersList list = keepersListTable_
								.get(focusedSequence);
						list.addAndActivateKeeper(keeper);
					} else {
						KeepersList list = new KeepersList();
						list.addAndActivateKeeper(keeper);
						keepersListTable_.put(focusedSequence, list);
					}
					snakeInteractionPane_.updateComboBox(
							keepersListTable_.get(focusedSequence)
									.getNumKeepers());

				} finally {
					keepersListLock_.unlock();
				}

				progressFrame.close();
				threadList_.remove(this);
			}
		};
		threadList_.add(thr);
		thr.start();
	}

	// ----------------------------------------------------------------------------

	/** Removes the active snake from the active <code>Sequence</code>. */
	private void removeActiveSnake() {
		Sequence focusedSequence = getActiveSequence();
		if (focusedSequence == null) {
			new AnnounceFrame("No image detected. Please, open an image first.");
			return;
		}
		keepersListLock_.lock();
		try {
			if (keepersListTable_.containsKey(focusedSequence)) {
				KeepersList keepersList = keepersListTable_
						.get(focusedSequence);
				if (keepersList != null) {
					keepersList.removeActiveSnakeKeeper();
				}
			}
			snakeInteractionPane_.updateComboBox(
					keepersListTable_.get(focusedSequence).getNumKeepers());

		} finally {
			keepersListLock_.unlock();
		}
	}

	// ----------------------------------------------------------------------------

	/** Clears the snakes from memory and detaches the ROI's from the image. */
	private void removeAllSnakes() {
		keepersListLock_.lock();
		try {
			for (Entry<Sequence, KeepersList> entry : keepersListTable_
					.entrySet()) {
				KeepersList keepersList = entry.getValue();
				if (keepersList != null) {
					if (!keepersList.isEmpty()) {
						keepersList.removeAllSnakeKeepers();
					}
				}
			}
			keepersListTable_.clear();
		} finally {
			keepersListLock_.unlock();
		}
	}

	// ----------------------------------------------------------------------------

	/**
	 * Retrieves the snake parameters from the GUI and sets them to the active
	 * snake.
	 */
	public void loadSnakeParametersFromInterface() {
		Sequence focusedSequence = getActiveSequence();
		if (focusedSequence == null) {
			return;
		}
		keepersListLock_.lock();
		try {
			if (keepersListTable_.containsKey(focusedSequence)) {
				KeepersList keepersList = keepersListTable_
						.get(focusedSequence);
				if (keepersList != null) {
					SnakeKeeper activeSnakeKeeper = keepersList
							.getActiveSnakeKeeper();
					if (activeSnakeKeeper != null) {
						CylinderSnakeParameters snakeParameters = new CylinderSnakeParameters(
								sphereSnakeSettingsPane_.getMaxIterations(),
								sphereSnakeSettingsPane_.getNumControlPoints(),
								sphereSnakeSettingsPane_.getAlpha(),
								sphereSnakeSettingsPane_.getGamma(),
								sphereSnakeSettingsPane_.isImmortal(),
								sphereSnakeSettingsPane_.getTargetBrightness(),
								sphereSnakeSettingsPane_.getEnergyType());
						activeSnakeKeeper.setSnakeParameters(snakeParameters);
					}
				}
			}
		} finally {
			keepersListLock_.unlock();
		}
	}

	// ----------------------------------------------------------------------------
	// I/O

	public void loadSnakesFromBinaryImage() {
		Thread thr = new Thread() {
			@Override
			public void run() {
				Sequence focusedSequence = getActiveSequence();
				if (focusedSequence == null) {
					new AnnounceFrame(
							"No image detected. Please, open an image first.");
					return;
				} else if (focusedSequence.getSizeZ() <= 1) {
					new AnnounceFrame(
							"This plugin works only with 3D images. For 2D images, you can use the 2D Active Cells segmentation plug-in.");
					return;
				}

				ProgressFrame progressFrame = new ProgressFrame(
						"Creating a new 3D snake");

				ImageLUTContainer imageLUTContainer = imageLUTs_
						.get(focusedSequence);

				if (imageLUTContainer == null) {
					System.err.println("LUT not found");
					return;
				}

				if (!imageLUTContainer.isLUTUpToDate()) {
					imageLUTContainer.buildLUTs();
				}
				CylinderSnakeParameters snakeParameters = new CylinderSnakeParameters(
						sphereSnakeSettingsPane_.getMaxIterations(),
						sphereSnakeSettingsPane_.getNumControlPoints(),
						sphereSnakeSettingsPane_.getAlpha(),
						sphereSnakeSettingsPane_.getGamma(),
						sphereSnakeSettingsPane_.isImmortal(),
						sphereSnakeSettingsPane_.getTargetBrightness(),
						sphereSnakeSettingsPane_.getEnergyType());

				CylinderSnake snake = new CylinderSnake(imageLUTContainer,
						snakeParameters);

				Snake3DNode[] snakeNodesCopy = snake.setShape(focusedSequence);
				snake.setNodes(snakeNodesCopy);

				SnakeKeeper keeper = new SnakeKeeper(
						imageLUTContainer.getOriginalSequence(), snake,
						mainPlugin_);
				nSnakes_++;
				keeper.setID("" + nSnakes_);
				keeper.setSnakeEditMode(getEditingMode());

				DisplaySettings ds = displaySettingsPane_.getDisplaySettings();
				if (ds != null) {
					keeper.setDisplaySettings(ds);
				}

				keepersListLock_.lock();
				try {
					if (keepersListTable_.containsKey(focusedSequence)) {
						KeepersList list = keepersListTable_
								.get(focusedSequence);
						list.addAndActivateKeeper(keeper);
					} else {
						KeepersList list = new KeepersList();
						list.addAndActivateKeeper(keeper);
						keepersListTable_.put(focusedSequence, list);
					}
				} finally {
					keepersListLock_.unlock();
				}
				progressFrame.close();
				threadList_.remove(this);
			}
		};
		threadList_.add(thr);
		thr.start();
	}

	// ----------------------------------------------------------------------------

	public void loadSnakesFromXML() {
		Thread thr = new Thread() {
			@Override
			public void run() {
				try {
					Sequence focusedSequence = getActiveSequence();
					if (focusedSequence == null) {
						new AnnounceFrame(
								"No sequence loaded. Please open an image first.");
						return;
					}

					keepersListLock_.lock();
					try {
						if (keepersListTable_.containsKey(focusedSequence)) {
							KeepersList keepersList = keepersListTable_
									.get(focusedSequence);
							// keepersList.deactivateActiveSnakeKeeper();
						}
					} finally {
						keepersListLock_.unlock();
					}

					try {
						ImageLUTContainer imageLUTContainer = imageLUTs_
								.get(focusedSequence);

						if (imageLUTContainer == null) {
							System.err.println("LUT not found");
							return;
						}

						ArrayList<SnakeKeeper> loadedKeepers = IOXML
								.loadSnakesFromXML(imageLUTContainer,
										mainPlugin_);
						if (loadedKeepers != null) {
							imageSettingsPane_
									.sequenceFocused(imageLUTContainer
											.getOriginalSequence());
							for (SnakeKeeper loadedKeeper : loadedKeepers) {
								if (loadedKeeper != null) {
									CylinderSnakeParameters snakeParameters = loadedKeeper
											.getESnakeParameters();
									sphereSnakeSettingsPane_
											.setSnakeParameters(snakeParameters);
									nSnakes_++;
									loadedKeeper.setID("" + nSnakes_);
									loadedKeeper
											.setSnakeEditMode(getEditingMode());

									keepersListLock_.lock();
									try {
										if (keepersListTable_
												.containsKey(focusedSequence)) {
											KeepersList list = keepersListTable_
													.get(focusedSequence);
											list.addAndActivateKeeper(loadedKeeper);
										} else {
											KeepersList list = new KeepersList();
											list.addAndActivateKeeper(loadedKeeper);
											keepersListTable_.put(
													focusedSequence, list);
										}
									} finally {
										keepersListLock_.unlock();
									}
									snakeInteractionPane_.updateComboBox(keepersListTable_.get(focusedSequence).getNumKeepers());
									// deactivateSnake(loadedKeeper);
								}
							}
						}
					} catch (Exception e) {
						new AnnounceFrame(
								"The snakes could not be loaded. XML file is corrupt or non-valid.");
						System.err
								.println("The snakes could not be loaded. XML file is corrupt or non-valid.");
						e.printStackTrace();
					}
				} finally {
					threadList_.remove(this);
				}
			}
		};
		threadList_.add(thr);
		thr.start();
	}

	// ----------------------------------------------------------------------------

	public void saveSnakesToBinaryImage() {
		System.out.println("Feature not implemented yet.");
		new AnnounceFrame("Feature not implemented yet.");
	}

	// ----------------------------------------------------------------------------
	/** Saves the snakes to an XML file. */
	public void saveSnakesToXML() {
		Thread thr = new Thread() {
			@Override
			public void run() {
				try {
					Sequence focusedSequence = getActiveSequence();
					if (focusedSequence == null) {
						new AnnounceFrame(
								"No image detected. Please, open an image first.");
						return;
					}

					if (nSnakes_ <= 0) {
						new AnnounceFrame(
								"No snake curves detected. Please, add one first.");
						return;
					}
					try {
						IOXML.saveSnakesToXML(imageLUTs_.get(focusedSequence));
					} catch (RowsExceededException e) {
						new AnnounceFrame(
								"An error occurred while saving the snakes. XML file is corrupt or non-valid.");
						System.err
								.println("An error occurred while saving the snakes. XML file is corrupt or non-valid.");
						e.printStackTrace();
					} catch (WriteException e) {
						new AnnounceFrame(
								"An error occurred while saving the snakes. XML file is corrupt or non-valid.");
						System.err
								.println("An error occurred while saving the snakes. XML file is corrupt or non-valid.");
						e.printStackTrace();
					} catch (Exception e) {
						new AnnounceFrame(
								"An error occurred while saving the snakes. XML file is corrupt or non-valid.");
						System.err
								.println("An error occurred while saving the snakes. XML file is corrupt or non-valid.");
						e.printStackTrace();
					}
				} finally {
					threadList_.remove(this);
				}
			}
		};
		threadList_.add(thr);
		thr.start();
	}

	// ----------------------------------------------------------------------------
	// SNAKE INTERACTION
	
	public void activateSnakeFromIndex(int index){

		Sequence focusedSequence = getActiveSequence();
		if (focusedSequence == null) {
			return;
		}
		keepersListLock_.lock();
		try {
				KeepersList keepersList = keepersListTable_.get(focusedSequence);
				if (keepersList != null) {
					boolean success = keepersList.activateSnakeKeeper(index);
					if (!success) {
						System.err
								.println("activateSnake: The SphereSnakeKeeper could not be activated.");
					}
				
			}
		} finally {
			keepersListLock_.unlock();
		}
	
	}

	// ----------------------------------------------------------------------------
	// SNAKE ANALYSIS

	public void rasterizeActiveSnake() {
		Sequence focusedSequence = getActiveSequence();
		if (focusedSequence == null) {
			new AnnounceFrame("No image detected. Please, open an image first.");
			return;
		}
		keepersListLock_.lock();
		try {
			if (keepersListTable_.containsKey(focusedSequence)) {
				KeepersList keepersList = keepersListTable_
						.get(focusedSequence);
				if (keepersList != null) {
					keepersList.rasterizeActiveSnake();
				}
			} else {
				new AnnounceFrame(
						"No image detected. Please, open an image first and add a snake.");
				return;
			}
		} finally {
			keepersListLock_.unlock();
		}
	}

	// ----------------------------------------------------------------------------
	// CLIPBOARD METHODS

	private void copySnake() {
		Sequence focusedSequence = getActiveSequence();
		if (focusedSequence == null) {
			return;
		}
		keepersListLock_.lock();
		try {
			if (keepersListTable_.containsKey(focusedSequence)) {
				SnakeKeeper keeper = keepersListTable_.get(focusedSequence)
						.getActiveSnakeKeeper();
				if (keeper != null) {
					clipboard_.setSnakeParameters(keeper.getESnakeParameters());
					clipboard_.setSnakeNodes(keeper.getNodesCopy());
				}
			}
		} finally {
			keepersListLock_.unlock();
		}
	}

	// ----------------------------------------------------------------------------

	private void pasteSnake() {
		if (clipboard_.isEmpty()) {
			new AnnounceFrame("The clipboard is empty. Copy a snake before.");
			return;
		}

		Thread thr = new Thread() {
			@Override
			public void run() {
				Sequence focusedSequence = getActiveSequence();
				if (focusedSequence == null) {
					new AnnounceFrame(
							"No image detected. Please, open an image first.");
					return;
				} else if (focusedSequence.getSizeZ() <= 1) {
					new AnnounceFrame(
							"This plugin works only with 3D images. For 2D images, you can use the 2D Active Cells segmentation plug-in.");
					return;
				}

				ProgressFrame progressFrame = new ProgressFrame(
						"Pasting a 3D snake");

				ImageLUTContainer imageLUTContainer = imageLUTs_
						.get(focusedSequence);

				if (imageLUTContainer == null) {
					System.err.println("LUT not found");
					return;
				}
				if (!imageLUTContainer.isLUTUpToDate()) {
					imageLUTContainer.buildLUTs();
				}

				CylinderSnake snake = new CylinderSnake(imageLUTContainer,
						clipboard_.getSnakeParameters());
				Snake3DNode[] nodes = clipboard_.getSnakeNodes();
				for (int i = 0; i < nodes.length; i++) {
					nodes[i].x += 20;
					nodes[i].y += 20;
					nodes[i].z += 0;
				}
				snake.setNodes(nodes);

				SnakeKeeper keeper = new SnakeKeeper(
						imageLUTContainer.getOriginalSequence(), snake,
						mainPlugin_);
				nSnakes_++;
				keeper.setID("" + nSnakes_);
				keeper.setSnakeEditMode(getEditingMode());

				DisplaySettings ds = displaySettingsPane_.getDisplaySettings();
				if (ds != null) {
					keeper.setDisplaySettings(ds);
				}

				keepersListLock_.lock();
				try {
					if (keepersListTable_.containsKey(focusedSequence)) {
						KeepersList list = keepersListTable_
								.get(focusedSequence);
						list.addAndActivateKeeper(keeper);
					} else {
						KeepersList list = new KeepersList();
						list.addAndActivateKeeper(keeper);
						keepersListTable_.put(focusedSequence, list);
					}
				} finally {
					keepersListLock_.unlock();
				}
				progressFrame.close();
				threadList_.remove(this);
			}
		};
		threadList_.add(thr);
		thr.start();
	}

	// ----------------------------------------------------------------------------
	// SNAKE OPTIMIZATION METHODS

	/**
	 * Launches the optimization routine for the active snake in the active
	 * <code>Sequence</code>.
	 */
	private void optimizeActiveSnake() {
		long start = System.currentTimeMillis();
		System.out.println("Start "+ start);
		Thread thr = new Thread() {
			@Override
			public void run() {
				Sequence focusedSequence = getActiveSequence();
				if (focusedSequence == null) {
					new AnnounceFrame(
							"No image detected. Please, open an image first.");
					return;
				}

				ImageLUTContainer imageLUTContainer = imageLUTs_
						.get(focusedSequence);

				if (imageLUTContainer == null) {
					System.err.println("LUT not found");
					return;
				}
				if (!imageLUTContainer.isLUTUpToDate()) {
					imageLUTContainer.buildLUTs();
				}

				SnakeKeeper activeKeeper = null;
				keepersListLock_.lock();
				try {
					if (keepersListTable_.get(focusedSequence) != null) {
						KeepersList keepersList = keepersListTable_
								.get(focusedSequence);
						if (keepersList != null) {
							activeKeeper = keepersList.getActiveSnakeKeeper();
						}
					} else {
						new AnnounceFrame(
								"No snakes detected. Add a snake first.");
					}
				} finally {
					keepersListLock_.unlock();
				}
				if (activeKeeper != null) {
					ProgressFrame pFrame = new ProgressFrame(
							"Optimizing snake position");
					try {
						activeKeeper.startOptimization();
					} finally {
						threadList_.remove(this);
					}
					pFrame.close();
				}
				
			}
		};
		threadList_.add(thr);
		thr.start();
		
	}

	// ----------------------------------------------------------------------------

	/** Launched the optimization routine for all the snakes. */
	private void optimizeAllSnakes() {
		Thread thr = new Thread() {
			@Override
			public void run() {
				Sequence focusedSequence = getActiveSequence();
				if (focusedSequence == null) {
					new AnnounceFrame(
							"No image detected. Please, open an image first.");
					return;
				}

				ImageLUTContainer imageLUTContainer = imageLUTs_
						.get(focusedSequence);

				if (imageLUTContainer == null) {
					System.err.println("LUT not found");
					return;
				}
				if (!imageLUTContainer.isLUTUpToDate()) {
					imageLUTContainer.buildLUTs();
				}

				SnakeKeeper activeKeeper = null;
				keepersListLock_.lock();
				try {
					if (keepersListTable_.get(focusedSequence) != null) {
						KeepersList keepersList = keepersListTable_
								.get(focusedSequence);
						if (keepersList != null) {
							for (int i = 0; i < keepersList.getNumKeepers(); i++) {
								if (keepersList.activateSnakeKeeper(i)) {
									activeKeeper = keepersList
											.getActiveSnakeKeeper();
									if (activeKeeper != null) {
										activeKeeper.startOptimization();
									}
								}
							}
						}
					} else {
						new AnnounceFrame(
								"No snakes detected. Add a snake first.");
					}
				} finally {
					keepersListLock_.unlock();
				}
			}
		};
		threadList_.add(thr);
		thr.start();
	}

	// ----------------------------------------------------------------------------
	// GUI METHODS

	/** Retrieves the interaction method specified on the GUI. */
	private SnakeEditMode getEditingMode() {
		if (moveSnakeButton_.isSelected()) {
			return SnakeEditMode.MOVE_SNAKE;
		}
		if (resizeSnakeButton_.isSelected()) {
			return SnakeEditMode.DILATE_SNAKE;
		}
		if (rotateSnakeButton_.isSelected()) {
			return SnakeEditMode.ROTATE_SNAKE;
		}
		return null;
	}

	// ----------------------------------------------------------------------------

	private void setActionPlane(ActionPlane actionPlane) {
		if (actionPlane == actionPlane_) {
			return;
		}
		actionPlane_ = actionPlane;
		keepersListLock_.lock();
		try {
			for (KeepersList keeperList : keepersListTable_.values()) {
				if (keeperList != null) {
					keeperList.setActionPlane(actionPlane);
				}
			}
		} finally {
			keepersListLock_.unlock();
		}
	}

	// ----------------------------------------------------------------------------

	public void setDisplaySettings(DisplaySettings displaySettings) {
		keepersListLock_.lock();
		try {
			for (Map.Entry<Sequence, KeepersList> keeperLists : keepersListTable_
					.entrySet()) {
				KeepersList keeperList = keeperLists.getValue();
				if (keeperList != null) {
					keeperList.setDisplaySettings(displaySettings);
				}
			}
		} finally {
			keepersListLock_.unlock();
		}
	}

	// ----------------------------------------------------------------------------

	/** Sets an interaction method to the GUI. */
	private void setEditingMode(SnakeEditMode editingMode) {
		switch (editingMode) {
		case DILATE_SNAKE:
			resizeSnakeButton_.setSelected(true);
			moveSnakeButton_.setSelected(false);
			rotateSnakeButton_.setSelected(false);
			break;
		case MOVE_SNAKE:
			moveSnakeButton_.setSelected(true);
			resizeSnakeButton_.setSelected(false);
			rotateSnakeButton_.setSelected(false);
			break;
		case ROTATE_SNAKE:
			rotateSnakeButton_.setSelected(true);
			resizeSnakeButton_.setSelected(false);
			moveSnakeButton_.setSelected(false);
			break;
		default:
			break;
		}
		if (editingMode == editingMode_) {
			return;
		}
		editingMode_ = editingMode;
		keepersListLock_.lock();
		try {
			for (KeepersList keeperList : keepersListTable_.values()) {
				if (keeperList != null) {
					keeperList.setSnakeEditMode(editingMode);
				}
			}
		} finally {
			keepersListLock_.unlock();
		}
	}

	// ----------------------------------------------------------------------------
	// SNAKE EDITING

	private void translateActiveSnake(int dx, int dy, int dz) {
		Sequence focusedSequence = getActiveSequence();
		if (focusedSequence == null) {
			return;
		}

		ImageLUTContainer imageLUTContainer = imageLUTs_.get(focusedSequence);

		if (imageLUTContainer == null) {
			System.err.println("LUT not found");
			return;
		}
		if (!imageLUTContainer.isLUTUpToDate()) {
			imageLUTContainer.buildLUTs();
		}

		keepersListLock_.lock();
		try {
			if (keepersListTable_.get(focusedSequence) != null) {
				KeepersList keepersList = keepersListTable_
						.get(focusedSequence);
				if (keepersList != null) {
					SnakeKeeper activeKeeper = keepersList
							.getActiveSnakeKeeper();
					activeKeeper.shiftSnake(dx, dy, dz);
				}
			}
		} finally {
			keepersListLock_.unlock();
		}
	}

	// ----------------------------------------------------------------------------
	// LISTENER MANAGEMENT

	/**
	 * Adds a <code>KeyListener</code> to a <code>Viewer</code> if it has not
	 * one already.
	 */
	private void addKeyListenerToViewer(Viewer viewer) {
		if (!hasKeyListenerTable_.containsKey(viewer)) {
			hasKeyListenerTable_.put(viewer, true);
			viewer.addKeyListener(this);
		}
	}

	// ----------------------------------------------------------------------------

	/** Removes a <code>KeyListener</code> from a <code>>Viewer</code>. */
	private void removeKeyListenerFromViewer(Viewer viewer) {
		viewer.removeKeyListener(this);
	}

	// ----------------------------------------------------------------------------

	/** Removes all instances of <code>KeyListener</code> from all images. */
	private void removeAllKeyListeners() {
		ArrayList<Sequence> sequences = getSequences();
		for (Sequence sequence : sequences) {
			ArrayList<Viewer> viewers = sequence.getViewers();
			for (Viewer viewer : viewers) {
				removeKeyListenerFromViewer(viewer);
			}
		}
	}

	// ============================================================================
	// LISTENER METHODS

	// ----------------------------------------------------------------------------
	// ACTIONLISTENER METHODS

	@Override
	public void actionPerformed(ActionEvent e) {
		if (e.getSource() == createSnakeButton_) {
			addAndActivateSnake();
		} else if (e.getSource() == optimizeSnakeButton_) {
			optimizeActiveSnake();
		} else if (e.getSource() == optimizeAllSnakesButton_) {
			optimizeAllSnakes();
		} else if (e.getSource() == deleteSnakeButton_) {
			removeActiveSnake();
		}
	}

	// ----------------------------------------------------------------------------
	// ACTIVESEQUENCELISTENER METHODS

	@Override
	public void activeSequenceChanged(SequenceEvent event) {
	}

	// ----------------------------------------------------------------------------

	@Override
	public void sequenceActivated(Sequence sequence) {
		if (sequence != null) {
			imageSettingsPane_.sequenceFocused(sequence);
		}
	}

	// ----------------------------------------------------------------------------

	@Override
	public void sequenceDeactivated(Sequence sequence) {
	}

	// ----------------------------------------------------------------------------
	// ACTIVEVIEWERLISTENER METHODS

	@Override
	public void viewerActivated(Viewer viewer) {
		if (viewer != null) {
			addKeyListenerToViewer(viewer);
		}
	}

	// ----------------------------------------------------------------------------

	@Override
	public void viewerDeactivated(Viewer viewer) {
	}

	// ----------------------------------------------------------------------------

	@Override
	public void activeViewerChanged(ViewerEvent event) {
	}

	// ----------------------------------------------------------------------------
	// GLOBALSEQUENCELISTENER METHODS

	@Override
	public void sequenceOpened(Sequence sequence) {
	}

	// ----------------------------------------------------------------------------

	@Override
	public void sequenceClosed(Sequence sequence) {
		if (sequence != null) {
			imageSettingsPane_.sequenceClosed(sequence);
		}
	}

	// ----------------------------------------------------------------------------
	// GLOBALVIEWERLISTENER METHODS

	@Override
	public void viewerOpened(Viewer viewer) {
	}

	// ----------------------------------------------------------------------------

	@Override
	public void viewerClosed(Viewer viewer) {
		if (viewer != null) {
			removeKeyListenerFromViewer(viewer);
		}
	}

	// ----------------------------------------------------------------------------
	// KEYLISTENER METHODS

	/**
	 * Interactions with respect to <code>KeyEvent</code> to detect the COPY and
	 * PASTE commands, and to translate the snake using the keyboard arrows.
	 */
	@Override
	public void keyPressed(KeyEvent e) {
		int key = e.getKeyCode();
		int keymod = e.getModifiers();

		// get the default modifier key (CTRL for Windows, COMMAND for MacOS)
		int defmod = Toolkit.getDefaultToolkit().getMenuShortcutKeyMask();

		if (key == KeyEvent.VK_DELETE || key == KeyEvent.VK_BACK_SPACE) {
			removeActiveSnake();
		} else if (key == KeyEvent.VK_C && (keymod & defmod) != 0) {
			copySnake();
		} else if (key == KeyEvent.VK_V && (keymod & defmod) != 0) {
			pasteSnake();
		} else if (key == KeyEvent.VK_UP && (keymod & defmod) != 0) {
			translateActiveSnake(0, 0, 1);
		} else if (key == KeyEvent.VK_DOWN && (keymod & defmod) != 0) {
			translateActiveSnake(0, 0, -1);
		} else if (key == KeyEvent.VK_LEFT) {
			translateActiveSnake(-1, 0, 0);
		} else if (key == KeyEvent.VK_RIGHT) {
			translateActiveSnake(1, 0, 0);
		} else if (key == KeyEvent.VK_UP) {
			translateActiveSnake(0, -1, 0);
		} else if (key == KeyEvent.VK_DOWN) {
			translateActiveSnake(0, 1, 0);
		}
	}

	// ----------------------------------------------------------------------------

	@Override
	public void keyReleased(KeyEvent e) {
	}

	// ----------------------------------------------------------------------------

	@Override
	public void keyTyped(KeyEvent e) {
	}

	// ============================================================================
	// BLOCK METHODS

	@Override
	public void declareInput(VarList inputMap) {
		inputMap.add(inputSequenceBlock_);
		inputMap.add(sigmaBlock_);
		inputMap.add(gammaBlock_);
		inputMap.add(targetBrightnessBlock_);
		inputMap.add(MBlock_);
		inputMap.add(energyTypeBlock_);
		inputMap.add(alphaBlock_);
		inputMap.add(maxIterationsBlock_);
		inputMap.add(inputFormatROI_);
		inputMap.add(inputSnakeWorkbook_);
		inputMap.add(roiArray);
	}

	// ----------------------------------------------------------------------------

	@Override
	public void declareOutput(VarList outputMap) {
		outputMap.add(outputSnakeWorkbook_);
	}
}
