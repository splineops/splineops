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
package plugins.big.vascular.io;

import icy.file.FileUtil;
import icy.gui.frame.progress.AnnounceFrame;
import icy.roi.ROI;
import icy.sequence.Sequence;
import icy.util.XMLUtil;

import java.io.File;
import java.util.ArrayList;

import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.filechooser.FileNameExtensionFilter;

import org.w3c.dom.Document;
import org.w3c.dom.Node;

import plugins.big.vascular.BIGVascular;
import plugins.big.vascular.core.ImageLUTContainer;
import plugins.big.vascular.keeper.SnakeKeeper;
import plugins.big.vascular.roi.Snake3DROI;
import plugins.big.vascular.snake.CylinderSnake;
import plugins.big.vascular.snake.CylinderSnakeParameters;
import plugins.big.bigsnakeutils.icy.snake3D.Snake3DNode;
import plugins.big.bigsnakeutils.shape.priorshapes.shapes.Custom3D;

/**
 * Class that provides loading and exporting capabilities to the snake.
 * 
 * @version October 31, 2014
 * 
 * @author Nicolas Chenouard (nicolas.chenouard@gmail.com)
 * @author Ricard Delgado-Gonzalo (ricard.delgado@gmail.com)
 * @authoer Daniel Schmitter (daniel.schmitter@epfl.ch)
 */
public class IOXML {

	/**
	 * Label of the XML tag the metadata of the image from which the XML was
	 * generated.
	 */
	private final static String ROOT_META = "meta";
	/**
	 * Label of the XML tag containing the name of the image from which the XML
	 * was generated.
	 */
	private final static String ID_NAME = "name";
	/**
	 * Label of the XML tag containing the the information associated to the ROI
	 * of a collection of snakes.
	 */
	private final static String ROOT_ROIS = "rois";
	/**
	 * Label of the XML tag containing the the information associated to the ROI
	 * of a single snake.
	 */
	private final static String ID_ROI = "roi";

	// ============================================================================
	// PUBLIC METHODS

	/**
	 * Opens a dialog and saves all the snakes in the active
	 * <code>Sequence</code> in an XML file.
	 */
	public static void saveSnakesToXML(ImageLUTContainer imageLUT)
			throws Exception {
		if (imageLUT == null) {
			throw new Exception("imageLUT is null in saveSnakesToXML.");
		}

		Sequence sequence = imageLUT.getOriginalSequence();
		if (sequence == null) {
			throw new Exception(
					"sequence is null within imageLUT in saveSnakesToXML.");
		}

		//TODO: point to last chosen path when opening a new FileChooser
		// --> take care of Fileseparator (i.e. Windows / Mac / Unix compatibility)
		// For the moment the path is customized and not generic
		String pathname = "/Users/schmitter/Documents/ISBI_backup/ISBI15/testData/shapes";
		File filePointer = new File(pathname); 
		if (!filePointer.exists())
			filePointer = null;
		
		
		

		JFileChooser fc = new JFileChooser(filePointer);
		int returnVal = fc.showSaveDialog(new JFrame());
		if (returnVal == JFileChooser.APPROVE_OPTION) {
			String filename = fc.getSelectedFile().getAbsolutePath();
			String xmlFilename = filename + ".xml";
			Document document = XMLUtil.createDocument(true);
			XMLUtil.setElementValue(document.getDocumentElement(), ID_NAME,
					sequence.getName());
			addMetaDataToXML(document.getDocumentElement(), imageLUT);
			addROIsToXML(document.getDocumentElement(), sequence);
			XMLUtil.saveDocument(document, xmlFilename);
		}
	}

	// ----------------------------------------------------------------------------

	/** Opens a dialog to select an XML file, and loads all a prior shape. */
	public static Custom3D loadPriorShapeFromXML() throws Exception {

		JFrame frame = new JFrame();
		JFileChooser fc = new JFileChooser();
		FileNameExtensionFilter filter = new FileNameExtensionFilter(
				"XML Files", "xml");
		fc.setFileFilter(filter);
		int returnVal = fc.showOpenDialog(frame);
		if (returnVal == JFileChooser.APPROVE_OPTION) {
			String xmlFilename = fc.getSelectedFile().getAbsolutePath();
			Document document = null;
			if (xmlFilename != null) {
				if (FileUtil.exists(xmlFilename)) {
					document = XMLUtil.loadDocument(xmlFilename, true);
				} else {
					new AnnounceFrame("Selected file does not exist.");
					throw new Exception(
							"xmlFilename is null in loadPriorShapeFromXML.");
				}
			} else {
				throw new Exception(
						"xmlFilename is null in loadPriorShapeFromXML.");
			}

			final Node nodeROIs = XMLUtil.getElement(
					document.getDocumentElement(), ROOT_ROIS);

			if (nodeROIs != null) {
				ArrayList<Snake3DNode[]> controlPointsList = loadSnakeNodesFromXML(nodeROIs);
				return new Custom3D(controlPointsList.get(0));
			}
		}
		return null;
	}

	// ----------------------------------------------------------------------------

	/**
	 * Opens a dialog to select an XML file, and loads all the snakes in the
	 * active <code>Sequence</code>.
	 */
	public static ArrayList<SnakeKeeper> loadSnakesFromXML(
			ImageLUTContainer imageLUT, BIGVascular mainPlugin) throws Exception {

		if (imageLUT == null) {
			throw new Exception("imageLUT is null in loadSnakesFromXML.");
		}

		Sequence sequence = imageLUT.getOriginalSequence();

		if (sequence == null) {
			throw new Exception("sequence is null in loadSnakesFromXML.");
		}

		//TODO: point to last chosen path when opening a new FileChooser
		// --> take care of Fileseparator (i.e. Windows / Mac / Unix compatibility)
		// For the moment the path is customized and not generic
		String pathname = "/Users/schmitter/Documents/ISBI_backup/ISBI15/testData/shapes";
		File filePointer = new File(pathname); 
		if (!filePointer.exists())
			filePointer = null;
		
		//
		
		JFrame frame = new JFrame();
		JFileChooser fc = new JFileChooser(filePointer);
		FileNameExtensionFilter filter = new FileNameExtensionFilter(
				"XML Files", "xml");
		fc.setFileFilter(filter);
		int returnVal = fc.showOpenDialog(frame);
		if (returnVal == JFileChooser.APPROVE_OPTION) {
			String xmlFilename = fc.getSelectedFile().getAbsolutePath();
			Document document = null;
			if (xmlFilename != null) {
				if (FileUtil.exists(xmlFilename)) {
					document = XMLUtil.loadDocument(xmlFilename, true);
				} else {
					new AnnounceFrame("Selected file does not exist.");
					throw new Exception(
							"xmlFilename is null in loadSnakesFromXML.");
				}
			} else {
				throw new Exception("xmlFilename is null in loadSnakesFromXML.");
			}

			loadMetaDataFromXML(document.getDocumentElement(), imageLUT);

			ArrayList<SnakeKeeper> loadedKeepers = new ArrayList<SnakeKeeper>();

			final Node nodeROIs = XMLUtil.getElement(
					document.getDocumentElement(), ROOT_ROIS);

			if (nodeROIs != null) {
				ArrayList<Snake3DNode[]> controlPointsList = loadSnakeNodesFromXML(nodeROIs);
				ArrayList<CylinderSnakeParameters> snakeParametersList = loadSnakeParametersFromXML(nodeROIs);

				int detectedSnakes = controlPointsList.size();
				for (int i = 0; i < detectedSnakes; i++) {
					CylinderSnake snake = new CylinderSnake(imageLUT,
							snakeParametersList.get(i));
					snake.setNodes(controlPointsList.get(i));
					loadedKeepers.add(new SnakeKeeper(sequence, snake,
							mainPlugin));
				}
				return loadedKeepers;
			}
		}
		return null;
	}

	// ============================================================================
	// PRIVATE METHODS

	private static void addMetaDataToXML(Node node, ImageLUTContainer imageLUT)
			throws Exception {
		if (node == null) {
			throw new Exception("XML node is null in addMetaDataToXML.");
		}
		if (imageLUT == null) {
			throw new Exception("imageLUT is null in addMetaDataToXML.");
		}

		Sequence sequence = imageLUT.getOriginalSequence();
		if (sequence == null) {
			throw new Exception(
					"sequence is null within imageLUT in addMetaDataToXML.");
		}

		final Node nodeMeta = XMLUtil.setElement(node, ROOT_META);
		if (nodeMeta != null) {
			XMLUtil.setElementDoubleValue(nodeMeta, Sequence.ID_PIXEL_SIZE_X,
					sequence.getPixelSizeX());
			XMLUtil.setElementDoubleValue(nodeMeta, Sequence.ID_PIXEL_SIZE_Y,
					sequence.getPixelSizeY());
			XMLUtil.setElementDoubleValue(nodeMeta, Sequence.ID_PIXEL_SIZE_Z,
					sequence.getPixelSizeZ());
			XMLUtil.setElementDoubleValue(nodeMeta, Sequence.ID_TIME_INTERVAL,
					sequence.getTimeInterval());
			XMLUtil.setElementDoubleValue(nodeMeta, ImageLUTContainer.ID_SIGMA,
					imageLUT.getSigma());
			XMLUtil.setElementDoubleValue(nodeMeta,
					ImageLUTContainer.ID_ACTIVECHANNEL,
					imageLUT.getChannelNumber());
			for (int c = 0; c < sequence.getSizeC(); c++) {
				XMLUtil.setElementValue(nodeMeta, Sequence.ID_CHANNEL_NAME + c,
						sequence.getChannelName(c));
			}
		}
	}

	// ----------------------------------------------------------------------------

	private static void addROIsToXML(Node node, Sequence sequence)
			throws Exception {
		if (node == null) {
			throw new Exception("XML node is null in addROIsToXML.");
		}
		if (sequence == null) {
			throw new Exception("sequence is null in addROIsToXML.");
		}

		final Node nodeROIs = XMLUtil.setElement(node, ROOT_ROIS);
		if (nodeROIs != null) {
			XMLUtil.removeAllChildren(nodeROIs);
			for (ROI roi : sequence.getROIs()) {
				final Node nodeROI = XMLUtil.addElement(nodeROIs, ID_ROI);
				if (nodeROI != null) {
					if (!roi.saveToXML(nodeROI)) {
						XMLUtil.removeNode(nodeROIs, nodeROI);
					}
				}
			}
		}
	}

	// ----------------------------------------------------------------------------

	private static void loadMetaDataFromXML(Node node,
			ImageLUTContainer imageLUT) throws Exception {

		if (node == null) {
			throw new Exception("XML node is null in loadMetaDataFromXML.");
		}

		if (imageLUT == null) {
			throw new Exception("imageLUT is null in loadMetaDataFromXML.");
		}

		Sequence sequence = imageLUT.getOriginalSequence();

		if (sequence == null) {
			throw new Exception("Sequence is null in loadMetaDataFromXML.");
		}

		final Node nodeMeta = XMLUtil.getElement(node, ROOT_META);
		if (nodeMeta != null) {
			imageLUT.setChannelNumber(XMLUtil.getElementIntValue(nodeMeta,
					ImageLUTContainer.ID_ACTIVECHANNEL, 0));
			imageLUT.setSigma(XMLUtil.getElementDoubleValue(nodeMeta,
					ImageLUTContainer.ID_SIGMA, 10d));
		}
		if (!imageLUT.isLUTUpToDate()) {
			imageLUT.buildLUTs();
		}
	}

	// ----------------------------------------------------------------------------

	private static ArrayList<Snake3DNode[]> loadSnakeNodesFromXML(Node nodeROIs)
			throws Exception {

		if (nodeROIs == null) {
			throw new Exception("XML node is null in loadSnakeNodesFromXML.");
		}
		final ArrayList<Node> nodesROI = XMLUtil.getChildren(nodeROIs, ID_ROI);
		ArrayList<Snake3DNode[]> controlPointsList = new ArrayList<Snake3DNode[]>();
		if (nodesROI != null) {
			for (Node n : nodesROI) {
				final Node snakeParametersNode = XMLUtil.getElement(n,
						Snake3DROI.ID_SNAKE_PARAMETERS);
				final Node controlPointsNode = XMLUtil.getElement(
						snakeParametersNode, CylinderSnake.ID_CONTROL_POINTS);
				final ArrayList<Node> controlPointNodeList = XMLUtil
						.getChildren(controlPointsNode, CylinderSnake.ID_CONTROL_POINT);
				int M = controlPointNodeList.size();
				Snake3DNode[] controlPoints = new Snake3DNode[M];
				for (int i = 0; i < M; i++) {
					controlPoints[i] = new Snake3DNode(0, 0, 0);
					controlPoints[i].loadFromXML(controlPointNodeList.get(i));
				}
				controlPointsList.add(controlPoints);
			}
			return controlPointsList;
		}
		return null;
	}

	// ----------------------------------------------------------------------------

	private static ArrayList<CylinderSnakeParameters> loadSnakeParametersFromXML(
			Node nodeROIs) throws Exception {

		if (nodeROIs == null) {
			throw new Exception(
					"XML node is null in loadSnakeParametersFromXML.");
		}
		final ArrayList<Node> nodesROI = XMLUtil.getChildren(nodeROIs, ID_ROI);
		ArrayList<CylinderSnakeParameters> snakeParametersList = new ArrayList<CylinderSnakeParameters>();
		if (nodesROI != null) {
			for (Node n : nodesROI) {
				final Node snakeParametersNode = XMLUtil.getElement(n,
						Snake3DROI.ID_SNAKE_PARAMETERS);
				CylinderSnakeParameters snakeParameters = new CylinderSnakeParameters();
				snakeParameters.loadFromToXML(snakeParametersNode);
				snakeParametersList.add(snakeParameters);
			}
			return snakeParametersList;
		}
		return null;
	}
}
