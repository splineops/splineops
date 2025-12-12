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
package plugins.big.bigsnake3d.roi;

import icy.canvas.Canvas3D;
import icy.canvas.IcyCanvas;
import icy.canvas.IcyCanvas2D;
import icy.canvas.IcyCanvas3D;
import icy.gui.viewer.Viewer;
import icy.painter.OverlayEvent;
import icy.painter.OverlayEvent.OverlayEventType;
import icy.painter.OverlayListener;
import icy.roi.ROI3D;
import icy.sequence.Sequence;
import icy.type.point.Point5D;
import icy.type.rectangle.Rectangle3D;
import icy.util.XMLUtil;
import icy.vtk.VtkUtil;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Component;
import java.awt.Cursor;
import java.awt.Graphics2D;
import java.awt.event.MouseEvent;
import java.awt.geom.Line2D;
import java.awt.geom.Point2D;
import java.util.ArrayList;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

import javax.vecmath.Point3d;

import org.w3c.dom.Element;
import org.w3c.dom.Node;

import plugins.big.bigsnake3d.core.SettingsMSBrain;
import plugins.big.bigsnake3d.keeper.SnakeKeeperMSBrain;
import plugins.big.bigsnake3d.snake.SphereSnakeMSBrain;
import plugins.big.bigsnakeutils.icy.snake3D.Snake3DNode;
import plugins.big.bigsnakeutils.icy.snake3D.Snake3DScale;
import plugins.kernel.canvas.VtkCanvas;
import vtk.vtkActor;
import vtk.vtkCellArray;
import vtk.vtkDoubleArray;
import vtk.vtkPointPicker;
import vtk.vtkPoints;
import vtk.vtkPolyData;
import vtk.vtkPolyDataMapper;
import vtk.vtkRenderer;
import vtk.vtkSphereSource;

/**
 * Class that handles the interaction and the display of the snake.
 * 
 * @version May 3, 2014
 * 
 * @author Ricard Delgado-Gonzalo (ricard.delgado@gmail.com)
 * @author Nicolas Chenouard (nicolas.chenouard@gmail.com)
 */
public class ROI3DSnakeMSBrain extends ROI3D implements OverlayListener {

	/** Snake associated to the ROI. */
	private SphereSnakeMSBrain snake_ = null;
	/** Keeper associated to the ROI */
	private SnakeKeeperMSBrain keeper_ = null;
	/** Group of all polylines forming the skin of the snake. */
	private final ArrayList<Snake3DScale> scales_ = new ArrayList<Snake3DScale>();
	/** List with the control points. */
	private final ArrayList<Anchor3DMSBrain> nodes_ = new ArrayList<Anchor3DMSBrain>();

	// ----------------------------------------------------------------------------
	// STATUS FIELDS

	/** Denotes the current state of the snake for mouse interaction. */
	private SnakeEditModeMSBrain snakeEditMode_ = SnakeEditModeMSBrain.MOVE_SNAKE;
	/** Flag that denotes if the snake can be modified. */
	private boolean isEditable_;
	/** Denotes the current state of the snake for mouse interaction. */
	private ActionPlaneMSBrain actionPlane_ = ActionPlaneMSBrain.XY;

	// ----------------------------------------------------------------------------
	// COLORS

	/** Default ROI color for the snake. */
	private static final Color DEFAULT_NORMAL_COLOR = Color.RED;
	/** Default ROI color for the snake when selected. */
	private static final Color DEFAULT_SELECTED_COLOR = Color.GREEN;

	// ----------------------------------------------------------------------------
	// DISPLAY

	/** Stroke thickness of the snake's skin. */
	private float strokeMultiplier_ = SettingsMSBrain.STROKE_THICKNESS_DEFAULT;
	/** Controls the number of scales shown on the screen. */
	private int scaleSubsamplingFactor_ = 4;
	/** Range (in depth) of the scales shown in the 2D viewer. */
	private int displayRange_ = SettingsMSBrain.DEPTH_TRANSPARENCY_DEFAULT;

	private double pixelSizeX_ = 1d;
	private double pixelSizeY_ = 1d;
	private double pixelSizeZ_ = 1d;

	// ----------------------------------------------------------------------------
	// CURSORS

	/** Cursor shown on the screen when moving a control point. */
	private final Cursor defaultCursor_ = Cursor
			.getPredefinedCursor(Cursor.DEFAULT_CURSOR);
	/** Cursor shown on the screen when moving a control point. */
	private final Cursor moveControlPointCursor_ = Cursor
			.getPredefinedCursor(Cursor.CROSSHAIR_CURSOR);
	/** Cursor shown on the screen when moving the snake. */
	private final Cursor moveSnakeCursor_ = Cursor
			.getPredefinedCursor(Cursor.HAND_CURSOR);
	/** Cursor shown on the screen when dragging the whole snake. */
	private final Cursor dragSnakeCursor_ = Cursor
			.getPredefinedCursor(Cursor.HAND_CURSOR);
	/** Cursor shown on the screen when rotating the whole snake. */
	private final Cursor rotateSnakeCursor_ = Cursor
			.getPredefinedCursor(Cursor.HAND_CURSOR);
	/** Cursor shown on the screen when dilating the whole snake. */
	private final Cursor dilateSnakeCursor_ = Cursor
			.getPredefinedCursor(Cursor.HAND_CURSOR);

	// ----------------------------------------------------------------------------
	// LOCKS

	private final Lock scalesLock_ = new ReentrantLock();
	private final Lock nodesLock_ = new ReentrantLock();
	private final Lock pixelSizeLock_ = new ReentrantLock();

	// ----------------------------------------------------------------------------
	// 3D VTK INTERACTION: FIELDS

	private vtkPointPicker nodePicker = new vtkPointPicker();
	private int coloredPointIndex = 0;
	private int selectedPointIndex = -1;
	private double[] mouseOrigin = { 0, 0, 0 };
	private boolean snakeMoved = false;
	private boolean pointMoved = false;

	// ----------------------------------------------------------------------------
	// XML TAGS

	/** XML tag for the snake parameters. */
	public static final String ID_SNAKE_PARAMETERS = "snake_parameters";

	// ============================================================================
	// SNAKE3DROI METHODS

	// ----------------------------------------------------------------------------
	// PUBLIC METHODS

	/** Default constructor. */
	public ROI3DSnakeMSBrain(SphereSnakeMSBrain snake, SnakeKeeperMSBrain keeper) {
		super();
		isEditable_ = true;
		snake_ = snake;
		keeper_ = keeper;
		setEditMode(SnakeEditModeMSBrain.MOVE_SNAKE);
	}

	// ----------------------------------------------------------------------------

	public void changePosition(ArrayList<double[]> newPoints,
			Snake3DScale[] newScales) {
		try {
			boolean changedPoints = false;
			nodesLock_.lock();
			if (nodes_.isEmpty()) {
				for (double[] point : newPoints) {
					addControlPoint(point[0], point[1], point[2]);
				}
			} else {
				int i = 0;
				for (Anchor3DMSBrain a : nodes_) {
					double[] coords = newPoints.get(i);
					changedPoints = a.setPositionNoUpdate(coords[0], coords[1],
							coords[2]) || changedPoints;
					i++;
				}
			}
			if (changedPoints) {
				changeScales(newScales);
			}
		} finally {
			nodesLock_.unlock();
		}
	}

	// ----------------------------------------------------------------------------

	public void changeScales(Snake3DScale[] newScales) {
		try {
			scalesLock_.lock();
			scales_.clear();
			for (Snake3DScale scale : newScales) {
				scales_.add(scale);
			}
		} finally {
			scalesLock_.unlock();
		}
		if (this.painter != null && this.painter instanceof Snake3DPainter) {
			((Snake3DPainter) this.painter).updateScales3DRendererThreaded();
		}
	}

	// ----------------------------------------------------------------------------

	public void delete3DPainters() {
		if (painter instanceof Snake3DPainter) {
			for (Sequence seq : painter.getSequences()) {
				((Snake3DPainter) painter).remove3DPainters(seq);
				for (Anchor3DMSBrain a3d : nodes_) {
					a3d.remove3DPainters(seq);
				}
			}
		}
	}

	// ----------------------------------------------------------------------------

	public double[] getControlPoint(int i) {
		double[] pos = null;
		try {
			nodesLock_.lock();
			pos = new double[] { nodes_.get(i).getX(), nodes_.get(i).getY(),
					nodes_.get(i).getZ() };
		} finally {
			nodesLock_.unlock();
		}
		return pos;
	}

	// ----------------------------------------------------------------------------

	public void setActionPlane(ActionPlaneMSBrain actionPlane) {
		actionPlane_ = actionPlane;
	}

	// ----------------------------------------------------------------------------

	public void setDisplayRange(int displayRange) {
		if (displayRange_ != displayRange) {
			displayRange_ = displayRange;
			roiChanged();
		}
	}

	// ----------------------------------------------------------------------------

	public void setEditMode(SnakeEditModeMSBrain editMode) {
		snakeEditMode_ = editMode;
		if (editMode == SnakeEditModeMSBrain.MOVE_SNAKE) {
			try {
				nodesLock_.lock();
				for (Anchor3DMSBrain a : nodes_) {
					a.setSelected(true);
				}
			} finally {
				nodesLock_.unlock();
			}
		} else {
			try {
				nodesLock_.lock();
				for (Anchor3DMSBrain a : nodes_) {
					a.setSelected(false);
				}
			} finally {
				nodesLock_.unlock();
			}
		}
	}

	// ----------------------------------------------------------------------------

	public void setPixelSize(double pixelSizeX, double pixelSizeY,
			double pixelSizeZ) {
		try {
			pixelSizeLock_.lock();
			if (pixelSizeX != pixelSizeX_ || pixelSizeY != pixelSizeY_
					|| pixelSizeZ != pixelSizeZ_) {
				pixelSizeX_ = pixelSizeX;
				pixelSizeY_ = pixelSizeY;
				pixelSizeZ_ = pixelSizeZ;
				nodesLock_.lock();
				try {
					for (Anchor3DMSBrain node : nodes_) {
						node.setPixelSize(pixelSizeX, pixelSizeY, pixelSizeZ);
					}
				} finally {
					nodesLock_.unlock();
				}
				if (painter != null) {
					if (this.painter instanceof Snake3DPainter) {
						Snake3DPainter painter3D = ((Snake3DPainter) this.painter);
						painter3D.setPixelSize(pixelSizeX_, pixelSizeY_,
								pixelSizeZ_);
					}
					painter.painterChanged();
				}
			}
		} finally {
			pixelSizeLock_.unlock();
		}
	}

	// ----------------------------------------------------------------------------

	public void setScaleSubsamplingFactor(int subsamplingFactor) {
		if (scaleSubsamplingFactor_ != subsamplingFactor) {
			scaleSubsamplingFactor_ = subsamplingFactor;
			if (this.painter != null && this.painter instanceof Snake3DPainter) {
				((Snake3DPainter) this.painter).reinitScaleRendering();
			}
			roiChanged();
		}
	}

	// ----------------------------------------------------------------------------

	public void setStrokeThicknessMultiplier(int strokeMultiplier) {
		if (strokeMultiplier_ != strokeMultiplier) {
			strokeMultiplier_ = strokeMultiplier;
			roiChanged();
		}
	}

	// ----------------------------------------------------------------------------

	public void translate2D(double dx, double dy, double dz) {
		beginUpdate();
		try {
			nodesLock_.lock();
			for (Anchor3DMSBrain anchor : nodes_) {
				anchor.setPosition(anchor.getX() + dx, anchor.getY() + dy,
						anchor.getZ() + dz);
			}
		} finally {
			nodesLock_.unlock();
			endUpdate();
		}
	}

	// ============================================================================
	// PRIVATE METHODS

	private void addControlPoint(double x, double y, double z) {
		Anchor3DMSBrain point = new Anchor3DMSBrain(x, y, z, DEFAULT_NORMAL_COLOR,
				DEFAULT_SELECTED_COLOR);
		point.setPixelSize(pixelSizeX_, pixelSizeY_, pixelSizeZ_);
		// point.addAnchorListener(this);

		point.addOverlayListener(this);

		nodes_.add(point);
		roiChanged();
	}

	// ----------------------------------------------------------------------------

	private boolean sectionContains(Point2D imagePoint) {
		for (Snake3DScale scale : scales_) {
			if (scale.containsInXYProjection(imagePoint)) {
				return true;
			}
		}
		return false;
	}

	// ============================================================================
	// ROI3D METHODS

	// ----------------------------------------------------------------------------
	// PUBlIC METHODS

	/**
	 * Returns the bounding box aligned with the axis containing the snake
	 * surface.
	 */
	@Override
	public Rectangle3D computeBounds3D() {
		return snake_.getBounds();
	}

	// ----------------------------------------------------------------------------

	@Override
	public boolean contains(double x, double y, double z) {
		// TODO Auto-generated method stub
		return false;
	}

	// ----------------------------------------------------------------------------

	@Override
	public boolean contains(double x, double y, double z, double sizeX,
			double sizeY, double sizeZ) {
		// TODO Auto-generated method stub
		return false;
	}

	// ----------------------------------------------------------------------------

	@Override
	public boolean contains(double x, double y, double z, double t, double c) {
		return contains(x, y, z);
	}

	// ----------------------------------------------------------------------------

	/** Retrieves the area of the surface of the snake. */
	@Override
	public double getSurfaceArea() {
		return snake_.getArea();
	}

	// ----------------------------------------------------------------------------

	/** Retrieves the volume inside the surface determined by the snake. */
	@Override
	public double getVolume() {
		return snake_.getVolume();
	}

	// ----------------------------------------------------------------------------

	@Override
	public boolean hasSelectedPoint() {
		// TODO Auto-generated method stub
		return false;
	}

	// ----------------------------------------------------------------------------

	@Override
	public boolean intersects(double x, double y, double z, double sizeX,
			double sizeY, double sizeZ) {
		// TODO Auto-generated method stub
		return false;
	}

	// ----------------------------------------------------------------------------

	@Override
	public boolean intersects(double x, double y, double z, double t, double c,
			double sizeX, double sizeY, double sizeZ, double sizeT, double sizeC) {
		// TODO Auto-generated method stub
		return false;
	}

	// ----------------------------------------------------------------------------

	@Override
	public void overlayChanged(OverlayEvent event) {
		if (event.getType() == OverlayEventType.PAINTER_CHANGED) {
			roiChanged();
			painter.painterChanged();
		}
	}

	// ----------------------------------------------------------------------------

	/** Saves a snake from an XML node. */
	@Override
	public boolean saveToXML(Node node) {
		if (snake_ != null) {
			Element snakeParametersElement = XMLUtil.setElement(node,
					ID_SNAKE_PARAMETERS);
			snake_.saveToXML(snakeParametersElement);
		}
		return true;
	}

	// ----------------------------------------------------------------------------

	@Override
	public void setEditable(boolean isEditable) {
		isEditable_ = isEditable;
		try {
			nodesLock_.lock();
			for (Anchor3DMSBrain a : nodes_) {
				a.setEditable(isEditable);
			}
		} finally {
			nodesLock_.unlock();
		}
	}

	// ----------------------------------------------------------------------------
	// PROTECTED METHODS

	@Override
	protected ROIPainter createPainter() {
		return new Snake3DPainter(1, 1, 1);
	}

	// ============================================================================
	// INNER CLASSES

	/**
	 * Class that draws the ROI forming the snake.
	 * 
	 * @version May 1, 2013
	 * 
	 * @author Nicolas Chenouard (nicolas.chenouard@gmail.com)
	 * @author Ricard Delgado-Gonzalo (ricard.delgado@gmail.com)
	 */
	private class Snake3DPainter extends ROIPainter {

		/**
		 * Determines if the mouse pointer is within the snake or a control
		 * point.
		 */
		private SnakeInteractionMSBrain snakeInteraction_ = SnakeInteractionMSBrain.NOTHING;

		/** Point in image coordinates where the user clicked. */
		private Point5D.Double pressedImagePoint_ = null;
		/** Point in image coordinates where the is clicking clicked. */
		private Point2D firstControlPointInitPosition_ = null;
		/** Point with the center of gravity of the snake. */
		private Point3d snakeMassCenter_ = null;

		// 3d display
		private final ArrayList<vtkActor> scaleActors = new ArrayList<vtkActor>();
		private final ArrayList<vtkActor> nodesActors = new ArrayList<vtkActor>();

		private final ArrayList<vtkPolyData> scaleDataLists = new ArrayList<vtkPolyData>();

		private double pixelSizeX = 1d;
		private double pixelSizeY = 1d;
		private double pixelSizeZ = 1d;

		private boolean painter3Dinitialized = false;

		private final ReentrantLock lockUpdate = new ReentrantLock();
		private final Lock screenUpdateLock = new ReentrantLock();
		private final Condition updateCondition = screenUpdateLock
				.newCondition();

		private final Thread updateScalesThread = new Thread() {
			@Override
			public void run() {
				while (isAlive() && !this.isInterrupted()) {
					screenUpdateLock.lock();
					try {
						try {
							updateCondition.await();
							updateScales3DRenderer();
						} catch (InterruptedException e) {
							e.printStackTrace();
						}
					} finally {
						screenUpdateLock.unlock();
					}
				}
			}
		};

		// ============================================================================
		// PUBLIC METHODS

		public Snake3DPainter(double pixelSizeX, double pixelSizeY,
				double pixelSizeZ) {
			updateScalesThread.start();
			this.pixelSizeX = pixelSizeX;
			this.pixelSizeY = pixelSizeY;
			this.pixelSizeZ = pixelSizeZ;
		}

		// ----------------------------------------------------------------------------

		public void remove3DPainters(Sequence sequence) {
			if (painter3Dinitialized) {
				for (Viewer v : sequence.getViewers()) {
					IcyCanvas canvas = v.getCanvas();
					if (canvas instanceof VtkCanvas) {
						final VtkCanvas canvas3d = (VtkCanvas) canvas;
						vtkRenderer renderer = canvas3d.getRenderer();
						for (vtkActor actor : scaleActors) {
							renderer.RemoveActor(actor);
						}
					}
				}
				scaleActors.clear();
				scaleDataLists.clear();
			}
		}

		// ----------------------------------------------------------------------------

		@Override
		public void paint(Graphics2D g, Sequence sequence, IcyCanvas canvas) {
			nodesLock_.lock();
			try {
				if (canvas instanceof IcyCanvas2D) {
					final Graphics2D g2 = (Graphics2D) g.create();
					float stroke = (float) getAdjustedStroke(canvas)
							* strokeMultiplier_;
					// draw scales
					if (scales_ != null) {
						double minZnodes = sequence.getSizeZ();
						double maxZnodes = 0;
						for (Anchor3DMSBrain node : nodes_) {
							if (node.getZ() > maxZnodes) {
								maxZnodes = node.getZ();
							}
							if (node.getZ() < minZnodes) {
								minZnodes = node.getZ();
							}
						}
						int zSlide = canvas.getPositionZ();
						float maxDist = Math.max(zSlide, sequence.getSizeZ()
								- zSlide)
								/ displayRange_;
						g2.setStroke(new BasicStroke(stroke / 2));
						for (int i = 0; i < scales_.size(); i += scaleSubsamplingFactor_) {
							Snake3DScale scale = scales_.get(i);
							if (scale != null) {
								Color c = scale.getColor();
								if (c != null) {
									g2.setColor(c);
								}
								double[] prevPos = new double[3];
								boolean firstOne = true;
								for (double[] pos : scale.getCoordinates()) {
									if (firstOne) {
										firstOne = false;
									} else {
										float meanZ = (float) (pos[2] + prevPos[2]) / 2.f;
										float transparency = Math.max(0, 1
												- Math.abs(meanZ - zSlide)
												/ maxDist);
										float green = (float) Math
												.abs((meanZ - minZnodes)
														/ (maxZnodes - minZnodes));
										green = Math.max(green, 0);
										green = Math.min(green, 1);
										g2.setColor(new Color(1.f, green,
												1.f - green, transparency));
										g2.draw(new Line2D.Double(
												prevPos[0] + 0.5,
												prevPos[1] + 0.5, pos[0] + 0.5,
												pos[1] + 0.5));
									}
									prevPos[0] = pos[0];
									prevPos[1] = pos[1];
									prevPos[2] = pos[2];
								}
								if (scale.isClosed()) {
									double[] pos = scale.getCoordinates()[0];
									float meanZ = (float) (pos[2] + prevPos[2]) / 2.f;
									float transparency = Math.max(0,
											1 - Math.abs(meanZ - zSlide)
													/ maxDist);
									float green = (float) Math
											.abs((meanZ - minZnodes)
													/ (maxZnodes - minZnodes));
									green = Math.max(green, 0);
									green = Math.min(green, 1);
									g2.setColor(new Color(1.f, green,
											1.f - green, transparency));
									g2.draw(new Line2D.Double(prevPos[0] + 0.5,
											prevPos[1] + 0.5, pos[0] + 0.5,
											pos[1] + 0.5));
								}
							}
						}
					}
					// paint the nodes
					for (Anchor3DMSBrain node : nodes_) {
						node.setStroke(stroke);
						node.paint(g2, sequence, canvas);
					}
					g2.dispose();
				} else if (canvas instanceof VtkCanvas) {
					if (scales_ != null) {
						final VtkCanvas canvas3d = (VtkCanvas) canvas;
						final vtkRenderer renderer = canvas3d.getRenderer();
						// paint the scales
						if (!painter3Dinitialized) {
							for (vtkActor actor : scaleActors) {
								renderer.RemoveActor(actor);
							}
							scaleActors.clear();
							scaleDataLists.clear();
							init3DRenderer(renderer);
						}
					}
				}
			} finally {
				nodesLock_.unlock();
			}
		}

		// ----------------------------------------------------------------------------

		public void updateScales3DRenderer() {
			if (painter3Dinitialized) {
				final ArrayList<Snake3DScale> list = new ArrayList<Snake3DScale>();
				list.addAll(scales_);
				if (list != null) {
					if (list.size() == scaleDataLists.size()) {
						try {
							for (int i = 0; i < list.size(); i += scaleSubsamplingFactor_) {
								final Snake3DScale scale = list.get(i);
								final vtkPolyData data = scaleDataLists.get(i);
								double[][] scalePoints = scale.getCoordinates();
								data.SetPoints(getScaledPoints(scalePoints,
										this.pixelSizeX, this.pixelSizeY,
										this.pixelSizeZ));
							}
						} finally {
						}
					} else {
						reinitScaleRendering();
					}
				}
			}
		}

		// ----------------------------------------------------------------------------

		public void reinitScaleRendering() {
			painter3Dinitialized = false;
		}

		// ----------------------------------------------------------------------------

		public void setPixelSize(double pixelSizeX2, double pixelSizeY2,
				double pixelSizeZ2) {
			this.pixelSizeX = pixelSizeX2;
			this.pixelSizeY = pixelSizeY2;
			this.pixelSizeZ = pixelSizeZ2;

			if (painter3Dinitialized) {
				final ArrayList<Snake3DScale> list = new ArrayList<Snake3DScale>();
				list.addAll(scales_);
				if (list != null) {
					if (list.size() == scaleDataLists.size()) {
						lockUpdate.lock();
						try {
							for (int i = 0; i < list.size(); i += scaleSubsamplingFactor_) {
								final Snake3DScale scale = list.get(i);
								final vtkPolyData data = scaleDataLists.get(i);
								data.SetPoints(getScaledPoints(
										scale.getCoordinates(),
										this.pixelSizeX, this.pixelSizeY,
										this.pixelSizeZ));
							}
						} finally {
							lockUpdate.unlock();
						}
					} else {
						reinitScaleRendering();
					}
				}
			}
		}

		// ----------------------------------------------------------------------------

		/**
		 * Invoked when the mouse cursor has been moved onto the Icy canvas but
		 * no buttons have been pushed.
		 */
		@Override
		public void mouseMove(MouseEvent e, Point5D.Double imagePoint,
				IcyCanvas canvas) {
			if (canvas instanceof IcyCanvas2D) {
				if (isEditable_) {
					switch (snakeEditMode_) {
					case DILATE_SNAKE:
						mouseMoveDilate(e, imagePoint, canvas);
						break;
					case ROTATE_SNAKE:
						mouseMoveRotate(e, imagePoint, canvas);
						break;
					case MOVE_SNAKE:
						mouseMoveDefault(e, imagePoint, canvas);
						break;
					}
				}
			}

			// 3D interaction on vtk renderer
			else if (canvas instanceof IcyCanvas3D) {

				// if mouse is over a point, it is colored in red

				Canvas3D canvas3D = (Canvas3D) canvas;
				if (!snakeMoved)
					resetControlPointColor();
				if (pickControlPoint(canvas3D, e) != null) {
					vtkActor point = pickControlPoint(canvas3D, e);
					colorControlPoint(point);
					coloredPointIndex = indexOfControlPoint(point);
				}

			}

		}

		// ----------------------------------------------------------------------------

		/** Invoked when a mouse button has been pressed on the Icy canvas. */
		@Override
		public void mousePressed(MouseEvent e, Point5D.Double imagePoint,
				IcyCanvas canvas) {
			if (canvas instanceof IcyCanvas2D) {
				if (isEditable_) {
					pressedImagePoint_ = imagePoint;
					switch (snakeEditMode_) {
					case DILATE_SNAKE:
						mousePressedDilate(e, imagePoint, canvas);
						break;
					case ROTATE_SNAKE:
						mousePressedRotate(e, imagePoint, canvas);
						break;
					case MOVE_SNAKE:
						mousePressedDefault(e, imagePoint, canvas);
						break;
					}
				}
			} else if (canvas instanceof IcyCanvas3D) {
				// if mouse is over a point select the point otherwise select
				// the all snake
				Canvas3D canvas3D = (Canvas3D) canvas;
				if (pickControlPoint(canvas3D, e) != null) {
					selectedPointIndex = indexOfControlPoint(pickControlPoint(
							canvas3D, e));
					pointMoved = true;
				} else if (e.isShiftDown()) {
					mouseOrigin[0] = e.getX();
					mouseOrigin[1] = e.getY();
					mouseOrigin[2] = 0;
					snakeMoved = true;

				}
			}
		}

		// ----------------------------------------------------------------------------

		/**
		 * Invoked when a mouse button is pressed on the Icy canvas and then
		 * dragged.
		 */
		@Override
		public void mouseDrag(MouseEvent e, Point5D.Double imagePoint,
				IcyCanvas canvas) {
			if (canvas instanceof IcyCanvas2D) {
				if (isEditable_) {
					switch (snakeEditMode_) {
					case DILATE_SNAKE:
						mouseDragDilate(e, imagePoint, canvas);
						break;
					case ROTATE_SNAKE:
						mouseDragRotate(e, imagePoint, canvas);
						break;
					case MOVE_SNAKE:
						mouseDragDefault(e, imagePoint, canvas);
						break;
					}
				}
			} else if (canvas instanceof IcyCanvas3D) {
				Canvas3D canvas3D = (Canvas3D) canvas;

				// move the controlPoint along with the mouse movement
				// vtkActor controlPoint = pickControlPoint(canvas3D, e);

				if (isEditable_) {

					if (!e.isConsumed() && pointMoved) {
						setPositionFromMouseEvent(canvas3D, e);
						e.consume();
					}
					// if shift is pressed move the whole snake
					else if (!e.isConsumed() && snakeMoved) {

						e.consume();
					}
				}
			}
		}

		// ----------------------------------------------------------------------------

		/**
		 * Invoked when a mouse button has been pressed on the Icy canvas.
		 */
		@Override
		public void mouseReleased(MouseEvent e, Point5D.Double imagePoint,
				IcyCanvas canvas) {
			if (canvas instanceof IcyCanvas2D) {
				if (!isEditable_) {
					if (!isActiveFor(canvas)) {
						return;
					}
					ROI3DSnakeMSBrain.this.beginUpdate();
					try {
						if (ROI3DSnakeMSBrain.this
								.sectionContains(new Point2D.Double(imagePoint
										.getX(), imagePoint.getY()))) {
							snakeInteraction_ = SnakeInteractionMSBrain.SCALE;
						} else {
							snakeInteraction_ = SnakeInteractionMSBrain.NOTHING;
						}
						if (snakeInteraction_ == SnakeInteractionMSBrain.SCALE
								&& imagePoint.equals(pressedImagePoint_)) {
							keeper_.activateSnake();
						}
					} finally {
						ROI3DSnakeMSBrain.this.endUpdate();
					}
				}
			} else if (canvas instanceof IcyCanvas3D) {
				Canvas3D canvas3D = (Canvas3D) canvas;
				if (snakeMoved&& isEditable_) {
					for (int i = 0; i < nodes_.size(); i++) {
						double[] vector = { e.getX() - mouseOrigin[0],
								e.getY() - mouseOrigin[1], 0.0 };
						setPositionFromVector(canvas3D, vector, i);
					}
					e.consume();
				}
				// release the point, or the whole snake
				pointMoved = false;
				snakeMoved = false;
				mouseOrigin[0] = 0.0;
				mouseOrigin[1] = 0.0;
				mouseOrigin[2] = 0.0;

			}
		}

		// ============================================================================
		// PRIVATE METHODS

		private void mouseMoveDefault(MouseEvent e, Point5D.Double imagePoint,
				IcyCanvas canvas) {
			if (!isActiveFor(canvas)) {
				return;
			}
			ROI3DSnakeMSBrain.this.beginUpdate();
			try {
				snakeInteraction_ = getSnakeMouseInteraction(imagePoint, canvas);

				Anchor3DMSBrain selectedControlPoint = null;
				for (Anchor3DMSBrain pt : nodes_) {
					if (pt.isOver(canvas, imagePoint.getX(), imagePoint.getY())) {
						selectedControlPoint = pt;
						break;
					}
				}
				Component component = canvas.getViewComponent();
				switch (snakeInteraction_) {
				case NOTHING:
					if (component != null) {
						component.setCursor(defaultCursor_);
					}
					unselectAllControlPoints();
					break;
				case CONTROL_POINT:
					if (component != null) {
						component.setCursor(moveControlPointCursor_);
					}
					try {
						nodesLock_.lock();
						for (Anchor3DMSBrain pt : nodes_) {
							pt.setSelected(pt.equals(selectedControlPoint));
						}
					} finally {
						nodesLock_.unlock();
					}
					break;
				case SCALE:
					if (component != null) {
						component.setCursor(moveSnakeCursor_);
					}
					selectAllControlPoints();
					break;
				}
			} finally {
				ROI3DSnakeMSBrain.this.endUpdate();
			}
		}

		// ----------------------------------------------------------------------------

		private void mouseMoveDilate(MouseEvent e, Point5D.Double imagePoint,
				IcyCanvas canvas) {
			if (!isActiveFor(canvas)) {
				return;
			}
			ROI3DSnakeMSBrain.this.beginUpdate();
			try {
				snakeInteraction_ = getSnakeMouseInteraction(imagePoint, canvas);
				Component component = canvas.getViewComponent();
				switch (snakeInteraction_) {
				case NOTHING:
					if (component != null) {
						component.setCursor(defaultCursor_);
					}
					unselectAllControlPoints();
					snakeMassCenter_ = null;
					firstControlPointInitPosition_ = null;
					break;
				case CONTROL_POINT:
					if (component != null) {
						component.setCursor(dilateSnakeCursor_);
					}
					selectAllControlPoints();
					break;
				case SCALE:
					if (component != null) {
						component.setCursor(dilateSnakeCursor_);
					}
					selectAllControlPoints();
					break;
				}
			} finally {
				ROI3DSnakeMSBrain.this.endUpdate();
			}
		}

		// ----------------------------------------------------------------------------

		private void mouseMoveRotate(MouseEvent e, Point5D.Double imagePoint,
				IcyCanvas canvas) {
			if (!isActiveFor(canvas)) {
				return;
			}
			ROI3DSnakeMSBrain.this.beginUpdate();
			try {
				snakeInteraction_ = getSnakeMouseInteraction(imagePoint, canvas);
				Component component = canvas.getViewComponent();
				switch (snakeInteraction_) {
				case NOTHING:
					if (component != null) {
						component.setCursor(defaultCursor_);
					}
					unselectAllControlPoints();
					snakeMassCenter_ = null;
					firstControlPointInitPosition_ = null;
					break;
				case CONTROL_POINT:
					if (component != null) {
						component.setCursor(rotateSnakeCursor_);
					}
					selectAllControlPoints();
					break;
				case SCALE:
					if (component != null) {
						component.setCursor(rotateSnakeCursor_);
					}
					selectAllControlPoints();
					break;
				}
			} finally {
				ROI3DSnakeMSBrain.this.endUpdate();
			}
		}

		// ----------------------------------------------------------------------------

		/**
		 * Default actions when a mouse button has been pressed on the Icy
		 * canvas.
		 */
		private void mousePressedDefault(MouseEvent e,
				Point5D.Double imagePoint, IcyCanvas canvas) {
			if (!isActiveFor(canvas)) {
				return;
			}
			ROI3DSnakeMSBrain.this.beginUpdate();
			try {
				Component component = canvas.getViewComponent();
				switch (snakeInteraction_) {
				case NOTHING:
					if (component != null) {
						component.setCursor(defaultCursor_);
					}
					for (Anchor3DMSBrain pt : nodes_) {
						pt.mousePressed(e, imagePoint, canvas);
					}
					e.consume();
					break;
				case CONTROL_POINT:
					if (component != null) {
						component.setCursor(moveControlPointCursor_);
					}
					for (Anchor3DMSBrain pt : nodes_) {
						pt.mousePressed(e, imagePoint, canvas);
					}
					break;
				case SCALE:
					if (component != null) {
						component.setCursor(dragSnakeCursor_);
					}
					e.consume();
					break;
				}
			} finally {
				ROI3DSnakeMSBrain.this.endUpdate();
			}
		}

		// ----------------------------------------------------------------------------

		private void mousePressedDilate(MouseEvent e,
				Point5D.Double imagePoint, IcyCanvas canvas) {

			if (!isActiveFor(canvas)) {
				return;
			}
			ROI3DSnakeMSBrain.this.beginUpdate();
			try {
				Component component = canvas.getViewComponent();
				switch (snakeInteraction_) {
				case NOTHING:
					snakeMassCenter_ = null;
					firstControlPointInitPosition_ = null;
					break;
				case CONTROL_POINT:
					snakeMassCenter_ = null;
					firstControlPointInitPosition_ = null;
					break;
				case SCALE:

					if (component != null) {
						component.setCursor(dilateSnakeCursor_);
					}
					pressedImagePoint_ = imagePoint;
					snakeMassCenter_ = snake_.getCentroid();
					double[] pos = ROI3DSnakeMSBrain.this.getControlPoint(0);
					firstControlPointInitPosition_ = new Point2D.Double(pos[0],
							pos[1]);

					e.consume();
					break;
				}
			} finally {
				ROI3DSnakeMSBrain.this.endUpdate();
			}
		}

		// ----------------------------------------------------------------------------

		private void mousePressedRotate(MouseEvent e,
				Point5D.Double imagePoint, IcyCanvas canvas) {
			if (!isActiveFor(canvas)) {
				return;
			}
			ROI3DSnakeMSBrain.this.beginUpdate();
			try {
				Component component = canvas.getViewComponent();
				switch (snakeInteraction_) {
				case NOTHING:
					snakeMassCenter_ = null;
					firstControlPointInitPosition_ = null;
					break;
				case CONTROL_POINT:
					snakeMassCenter_ = null;
					firstControlPointInitPosition_ = null;
					break;
				case SCALE:
					if (component != null) {
						component.setCursor(rotateSnakeCursor_);
					}
					pressedImagePoint_ = imagePoint;
					snakeMassCenter_ = snake_.getCentroid();
					double[] pos = ROI3DSnakeMSBrain.this.getControlPoint(0);
					firstControlPointInitPosition_ = new Point2D.Double(pos[0],
							pos[1]);
					e.consume();
					break;
				}
			} finally {
				ROI3DSnakeMSBrain.this.endUpdate();
			}
		}

		// ----------------------------------------------------------------------------

		private void mouseDragDefault(MouseEvent e, Point5D.Double imagePoint,
				IcyCanvas canvas) {
			if (!isActiveFor(canvas)) {
				return;
			}
			ROI3DSnakeMSBrain.this.beginUpdate();
			try {
				Component component = canvas.getViewComponent();
				for (Anchor3DMSBrain pt : nodes_) {
					if (pt.isOver(canvas, imagePoint.getX(), imagePoint.getY())) {
						if (component != null) {
							component.setCursor(moveControlPointCursor_);
						}
						break;
					}
				}
				switch (snakeInteraction_) {
				case NOTHING:
					break;
				case CONTROL_POINT:
					e.consume();
					for (Anchor3DMSBrain pt : nodes_) {
						pt.mouseDrag(e, imagePoint, canvas);
					}
					break;
				case SCALE:
					e.consume();
					if (component != null) {
						component.setCursor(dragSnakeCursor_);
					}
					double dx = imagePoint.getX() - pressedImagePoint_.getX();
					double dy = imagePoint.getY() - pressedImagePoint_.getY();
					pressedImagePoint_ = imagePoint;
					ROI3DSnakeMSBrain.this.translate2D(dx, dy, 0);
					break;
				}
			} finally {
				ROI3DSnakeMSBrain.this.endUpdate();
			}
		}

		// ----------------------------------------------------------------------------

		private void dilateX(double dilationFactor) {

			for (Anchor3DMSBrain p : nodes_) {
				p.setPosition(snakeMassCenter_.x + dilationFactor
						* (p.getX() - snakeMassCenter_.x), p.getY());
			}
		}

		// ----------------------------------------------------------------------------

		private void dilateY(double dilationFactor) {
			for (Anchor3DMSBrain p : nodes_) {
				p.setPosition(p.getX(), snakeMassCenter_.y + dilationFactor
						* (p.getY() - snakeMassCenter_.y));
			}
		}

		// ----------------------------------------------------------------------------

		private void dilateZ(double dilationFactor) {
			for (Anchor3DMSBrain p : nodes_) {
				p.setPosition(p.getX(), p.getY(), snakeMassCenter_.z
						+ dilationFactor * (p.getZ() - snakeMassCenter_.z));
			}
		}

		// ----------------------------------------------------------------------------

		private void rotateX(double gamma) {
			for (Anchor3DMSBrain p : nodes_) {
				p.setPosition(
						p.getX(),
						(p.getY() - snakeMassCenter_.y) * Math.cos(gamma)
								- (p.getZ() - snakeMassCenter_.z)
								* Math.sin(gamma) + snakeMassCenter_.y,
						(p.getY() - snakeMassCenter_.y) * Math.sin(gamma)
								+ (p.getZ() - snakeMassCenter_.z)
								* Math.cos(gamma) + snakeMassCenter_.z);
			}
		}

		// ----------------------------------------------------------------------------

		private void rotateY(double beta) {
			for (Anchor3DMSBrain p : nodes_) {
				p.setPosition(
						(p.getX() - snakeMassCenter_.x) * Math.cos(beta)
								+ (p.getZ() - snakeMassCenter_.z)
								* Math.sin(beta) + snakeMassCenter_.x,
						p.getY(),
						-(p.getX() - snakeMassCenter_.x) * Math.sin(beta)
								+ (p.getZ() - snakeMassCenter_.z)
								* Math.cos(beta) + snakeMassCenter_.z);
			}
		}

		// ----------------------------------------------------------------------------

		private void rotateZ(double alpha) {
			for (Anchor3DMSBrain p : nodes_) {
				p.setPosition(snakeMassCenter_.x + Math.cos(alpha)
						* (p.getX() - snakeMassCenter_.x) - Math.sin(alpha)
						* (p.getY() - snakeMassCenter_.y), snakeMassCenter_.y
						+ Math.sin(alpha) * (p.getX() - snakeMassCenter_.x)
						+ Math.cos(alpha) * (p.getY() - snakeMassCenter_.y));
			}
		}

		// ----------------------------------------------------------------------------

		private void mouseDragDilate(MouseEvent e, Point5D.Double imagePoint,
				IcyCanvas canvas) {
			if (!isActiveFor(canvas)) {
				return;
			}
			ROI3DSnakeMSBrain.this.beginUpdate();
			try {
				if (pressedImagePoint_ != null && snakeMassCenter_ != null
						&& !nodes_.isEmpty()) {
					switch (snakeInteraction_) {
					case NOTHING:
						// e.consume();
						break;
					case CONTROL_POINT:
						// e.consume();
						break;
					case SCALE:

						// BEFORE
						/*
						 * double initDist = (new Point2D.Double(
						 * imagePoint.getX(), imagePoint.getY()))
						 * .distance(snakeMassCenter_.x, snakeMassCenter_.y);
						 */

						// TODO: Debug ESpline sphere snake
						// DEBUG
						double initDist = (new Point2D.Double(
								pressedImagePoint_.getX(),
								pressedImagePoint_.getY())).distance(
								snakeMassCenter_.x, snakeMassCenter_.y);
						// END DEBUG

						if (initDist == 0) {
							return;
						}

						double mouseDilationFactor = (new Point2D.Double(
								imagePoint.getX(), imagePoint.getY()))
								.distance(snakeMassCenter_.x,
										snakeMassCenter_.y)
								/ initDist;

						double snakeDilationFactor = snakeMassCenter_
								.distance(new Point3d(nodes_.get(0).getX(),
										nodes_.get(0).getY(),
										snakeMassCenter_.z))
								/ firstControlPointInitPosition_.distance(
										snakeMassCenter_.x, snakeMassCenter_.y);
						double dilationFactor = mouseDilationFactor
								/ snakeDilationFactor;

						try {
							nodesLock_.lock();
							switch (actionPlane_) {
							case XY:
								dilateX(dilationFactor);
								dilateY(dilationFactor);
								break;
							case YZ:
								dilateY(dilationFactor);
								dilateZ(dilationFactor);
								break;
							case XZ:
								dilateX(dilationFactor);
								dilateZ(dilationFactor);
								break;
							}
						} finally {
							nodesLock_.unlock();
						}
						e.consume();
						break;
					}
				}
			} finally {
				ROI3DSnakeMSBrain.this.endUpdate();
			}
		}

		// ----------------------------------------------------------------------------

		private void mouseDragRotate(MouseEvent e, Point5D.Double imagePoint,
				IcyCanvas canvas) {
			if (!isActiveFor(canvas)) {
				return;
			}
			ROI3DSnakeMSBrain.this.beginUpdate();
			try {
				if (pressedImagePoint_ != null && snakeMassCenter_ != null
						&& !nodes_.isEmpty()) {
					switch (snakeInteraction_) {
					case NOTHING:
						// e.consume();
						break;
					case CONTROL_POINT:
						// e.consume();
						break;
					case SCALE:
						double distMouse = (new Point2D.Double(
								imagePoint.getX(), imagePoint.getY()))
								.distance(snakeMassCenter_.x,
										snakeMassCenter_.y);
						if (distMouse == 0) {
							return;
						}
						double cosAngMouse = (imagePoint.getX() - snakeMassCenter_.x)
								/ distMouse;
						double sinAngMouse = (imagePoint.getY() - snakeMassCenter_.y)
								/ distMouse;

						double distMouseInit = (new Point2D.Double(
								pressedImagePoint_.getX(),
								pressedImagePoint_.getY())).distance(
								snakeMassCenter_.x, snakeMassCenter_.y);

						if (distMouseInit == 0) {
							return;
						}
						double cosAngMouseInit = (pressedImagePoint_.getX() - snakeMassCenter_.x)
								/ distMouseInit;
						double sinAngMouseInit = (pressedImagePoint_.getY() - snakeMassCenter_.y)
								/ distMouseInit;

						double cosDeltaAngMouse = cosAngMouse * cosAngMouseInit
								+ sinAngMouseInit * sinAngMouse;
						double sinDeltaAngMouse = sinAngMouse * cosAngMouseInit
								- sinAngMouseInit * cosAngMouse;

						// compute the rotation of the first control point
						double distFirstPointInit = firstControlPointInitPosition_
								.distance(snakeMassCenter_.x,
										snakeMassCenter_.y);

						if (distFirstPointInit == 0) {
							return;
						}
						double cosAngPointInit = (firstControlPointInitPosition_
								.getX() - snakeMassCenter_.x)
								/ distFirstPointInit;
						double sinAngPointInit = (firstControlPointInitPosition_
								.getY() - snakeMassCenter_.y)
								/ distFirstPointInit;

						double distFirstPoint = snakeMassCenter_
								.distance(new Point3d(nodes_.get(0).getX(),
										nodes_.get(0).getY(),
										snakeMassCenter_.z));

						if (distFirstPoint == 0) {
							return;
						}
						double cosAngPoint = (nodes_.get(0).getX() - snakeMassCenter_.x)
								/ distFirstPoint;
						double sinAngPoint = (nodes_.get(0).getY() - snakeMassCenter_.y)
								/ distFirstPoint;

						double cosDeltaAngPoint = cosAngPoint * cosAngPointInit
								+ sinAngPointInit * sinAngPoint;
						double sinDeltaAngPoint = sinAngPoint * cosAngPointInit
								- sinAngPointInit * cosAngPoint;

						double cosRotAngle = cosDeltaAngMouse
								* cosDeltaAngPoint + sinDeltaAngMouse
								* sinDeltaAngPoint;
						double sinRotAngle = sinDeltaAngMouse
								* cosDeltaAngPoint - sinDeltaAngPoint
								* cosDeltaAngMouse;

						try {
							nodesLock_.lock();
							switch (actionPlane_) {
							case XY:
								rotateZ(Math.atan2(sinRotAngle, cosRotAngle));
								break;
							case YZ:
								rotateX(Math.atan2(sinRotAngle, cosRotAngle));
								break;
							case XZ:
								rotateY(Math.atan2(sinRotAngle, cosRotAngle));
								break;
							}
						} finally {
							nodesLock_.unlock();
						}
						e.consume();
						break;
					}
				}
			} finally {
				ROI3DSnakeMSBrain.this.endUpdate();
			}
		}

		// ----------------------------------------------------------------------------

		private SnakeInteractionMSBrain getSnakeMouseInteraction(
				Point5D.Double imagePoint, IcyCanvas canvas) {
			for (Anchor3DMSBrain pt : nodes_) {
				if (pt.isOver(canvas, imagePoint.getX(), imagePoint.getY())) {
					return SnakeInteractionMSBrain.CONTROL_POINT;
				}
			}
			if (ROI3DSnakeMSBrain.this.sectionContains(new Point2D.Double(imagePoint
					.getX(), imagePoint.getY()))) {
				return SnakeInteractionMSBrain.SCALE;
			}
			return SnakeInteractionMSBrain.NOTHING;
		}

		// ----------------------------------------------------------------------------

		private void selectAllControlPoints() {
			nodesLock_.lock();
			try {
				for (Anchor3DMSBrain pt : nodes_) {
					pt.setSelected(true);
				}
			} finally {
				nodesLock_.unlock();
			}
		}

		// ----------------------------------------------------------------------------

		private void unselectAllControlPoints() {
			nodesLock_.lock();
			try {
				for (Anchor3DMSBrain pt : nodes_) {
					pt.setSelected(false);
				}
			} finally {
				nodesLock_.unlock();
			}
		}

		// ----------------------------------------------------------------------------

		private void init3DRenderer(vtkRenderer renderer) {
			renderer.SetGlobalWarningDisplay(0);
			scalesLock_.lock();
			try {
				for (int i = 0; i < scales_.size(); i += scaleSubsamplingFactor_) {
					Snake3DScale scale = scales_.get(i);
					double[][] scalePoints = scale.getCoordinates();

					final vtkPoints points = getScaledPoints(scalePoints,
							this.pixelSizeX, this.pixelSizeY, this.pixelSizeZ);
					final vtkCellArray cells;
					int numSegments = scalePoints.length - 1;
					if (scale.isClosed()) {
						numSegments++;
					}
					int[][] lineIdx = new int[numSegments][2];
					for (int j = 0; j < scalePoints.length - 1; j++) {
						lineIdx[j] = new int[] { j, j + 1 };
					}
					if (scale.isClosed()) {
						lineIdx[numSegments - 1] = new int[] { 0,
								numSegments - 1 };
					}
					// fast java data conversion for cells (polygons)
					cells = VtkUtil.getCells(scalePoints.length - 1,
							VtkUtil.prepareCells(lineIdx));

					vtkPolyData scaleData = new vtkPolyData();
					// set vertex
					scaleData.SetPoints(points);
					// set lines
					scaleData.SetLines(cells);
					scaleDataLists.add(scaleData);

					// add actor to the renderer
					final vtkPolyDataMapper polyMapper = new vtkPolyDataMapper();
					polyMapper.SetInputData(scaleData);
					vtkActor lineActor = new vtkActor();
					lineActor.SetMapper(polyMapper);
					// lineActor.GetProperty().SetRepresentationToWireframe();
					renderer.AddActor(lineActor);
					Color c = scale.getColor();
					double red = c.getRed();
					double green = c.getGreen();
					double blue = c.getBlue();
					// paint the snake in red if selected, in blue else
					if (isEditable_) {
						lineActor.GetProperty().SetColor(red / 255d,
								green / 255d, blue / 255d);
					} else {
						lineActor.GetProperty().SetColor(0, 0, 255d);

					}
					scaleActors.add(lineActor);
				}

				// display control points (nodes) on 3D vtk renderer
				createNodesActors(renderer);
				createPicker();

				painter3Dinitialized = true;
			} finally {
				scalesLock_.unlock();
			}
		}

		// ----------------------------------------------------------------------------

		private void updateScales3DRendererThreaded() {
			if (screenUpdateLock.tryLock()) {
				try {
					updateCondition.signalAll();
				} finally {
					screenUpdateLock.unlock();
				}
			}
		}

		// ----------------------------------------------------------------------------

		private vtkPoints getScaledPoints(double[][] coordinates,
				double scaleX, double scaleY, double scaleZ) {
			final vtkPoints result = new vtkPoints();
			if (coordinates.length < 1) {
				return result;
			}
			double[] coordinatesVector = new double[coordinates.length * 3];
			for (int i = 0; i < coordinates.length; i++) {
				coordinatesVector[3 * i] = coordinates[i][0] * scaleX;
				coordinatesVector[3 * i + 1] = coordinates[i][1] * scaleY;
				coordinatesVector[3 * i + 2] = coordinates[i][2] * scaleZ;
			}
			final vtkDoubleArray array = new vtkDoubleArray();
			array.SetJavaArray(coordinatesVector);
			array.SetNumberOfComponents(3);
			result.SetData(array);
			return result;
		}

		// ----------------------------------------------------------------------------

		private double[] pointToNodeScale(double[] coordinates, double scaleX,
				double scaleY, double scaleZ) {
			double[] scaledPoint = { coordinates[0] / scaleX,
					coordinates[1] / scaleY, coordinates[2] / scaleZ };
			return scaledPoint;
		}

		// ----------------------------------------------------------------------------

		private double[] nodeToWorldScale(double[] coordinates, double scaleX,
				double scaleY, double scaleZ) {
			double[] scaledPoint = { coordinates[0] * scaleX,
					coordinates[1] * scaleY, coordinates[2] * scaleZ };
			return scaledPoint;
		}

		// ----------------------------------------------------------------------------

		// method that create nodesActors from nodes_ list
		private void createNodesActors(vtkRenderer renderer) {
			nodesActors.clear();
			
			
			//for (int i = 0; i < nodes_.size(); i++) { // displaying tangent plane control points
			for (int i = 0; i < nodes_.size()-4; i++) { // not displaying tangent plane control points
				
				/*
				 * total number of points: M_ * (M_ - 1) + 6
				 * 
				 tangent plane control points:
				 				 				 
				 coef_[M_ * (M_ - 1) + 2]				
				 coef_[M_ * (M_ - 1) + 3] 
				 coef_[M_ * (M_ - 1) + 4] 
				 coef_[M_ * (M_ - 1) + 5] (last point of array)
				*/
				
				
				
				

				vtkSphereSource sphereNode = new vtkSphereSource();
				double[] nodePos = { nodes_.get(i).getX(),
						nodes_.get(i).getY(), nodes_.get(i).getZ() };
				sphereNode.SetCenter(nodeToWorldScale(nodePos, this.pixelSizeX,
						this.pixelSizeY, this.pixelSizeZ));

				sphereNode.SetRadius(2 * this.pixelSizeX);
				sphereNode.SetThetaResolution(25);
				sphereNode.SetPhiResolution(25);

				sphereNode.Update();

				vtkPolyData scaleData = sphereNode.GetOutput();
				scaleDataLists.add(scaleData);
				vtkPolyDataMapper sphereMapper = new vtkPolyDataMapper();
				sphereMapper.SetInputData(scaleData);
				vtkActor sphereActor = new vtkActor();
				sphereActor.GetProperty().SetColor(0, 0, 1); // color of 3D
																// control
																// points
				sphereActor.SetMapper(sphereMapper);

				renderer.AddActor(sphereActor);

				scaleActors.add(sphereActor);
				nodesActors.add(sphereActor);

			}

		}

		// ----------------------------------------------------------------------------

		// create the Picker of the nodes
		private void createPicker() {
			nodePicker = new vtkPointPicker();
			nodePicker.SetTolerance(0.001);
			for (int i = 0; i < nodesActors.size(); i++) {
				nodePicker.AddPickList(nodesActors.get(i));
			}
			nodePicker.PickFromListOn();
		}

		// ----------------------------------------------------------------------------

		// return the actor selected by the mouseEvent and null if no actor is
		// selected
		private vtkActor pickControlPoint(Canvas3D canvas3D, MouseEvent e) {
			nodePicker.Pick(e.getX(),
					canvas3D.getRenderer().GetSize()[1] - e.getY(), 0,
					canvas3D.getRenderer());
			if (nodePicker.GetActor() != null) {
				return nodePicker.GetActor();
			}
			return null;
		}

		// ----------------------------------------------------------------------------

		// return the index of the actor in nodes_ and -1 if it isn't in nodes_
		private int indexOfControlPoint(vtkActor actor) {
			for (int i = 0; i < nodesActors.size(); i++) {
				if (nodesActors.get(i) == actor)
					return i;
			}
			return -1;
		}

		// ----------------------------------------------------------------------------

		// reset the control points color
		private void resetControlPointColor() {
			/*
			 * for (int i = 0; i < nodesActors.size(); i++) {
			 * nodesActors.get(i).GetProperty().SetColor(0, 0, 1); }
			 */
			if (nodesActors.size() != 0) {
				nodesActors.get(coloredPointIndex).GetProperty()
						.SetColor(0, 0, 1);
			}
			painterChanged();

		}

		// ----------------------------------------------------------------------------

		// color the given control point in green
		private void colorControlPoint(vtkActor node) {
			if (node != null && isEditable_) {
				node.GetProperty().SetColor(0, 1, 0);
			}
			painterChanged();
		}

		// ----------------------------------------------------------------------------

		// transform a point from display to world
		private double[] transformDisplayPointToWorld(Canvas3D canvas3d,
				double[] displayPoint) {
			canvas3d.getRenderer().SetDisplayPoint(displayPoint);
			canvas3d.getRenderer().DisplayToWorld();
			double[] worldPoint = canvas3d.getRenderer().GetWorldPoint();
			return worldPoint;
		}

		// ----------------------------------------------------------------------------

		// return the z display coordinate from any point in world coordinates
		private double getInDisplayCoordinates(Canvas3D canvas3d,
				double[] controlPoint, int index) {
			double result = 0;
			canvas3d.getRenderer().SetWorldPoint(controlPoint);
			canvas3d.getRenderer().WorldToDisplay();
			double[] controlPointDisplay = canvas3d.getRenderer()
					.GetDisplayPoint();
			if (controlPointDisplay.length >= 3) {
				result = controlPointDisplay[index];
			}
			return result;
		}

		// --_--------------------------------------------------------------------------

		private void setPositionFromMouseEvent(Canvas3D canvas3D, MouseEvent e) {
			vtkActor controlPoint = nodesActors.get(selectedPointIndex);
			// scale the point
			double[] position = controlPoint.GetMapper().GetCenter();

			// get the z depth of the control point
			double depth = getInDisplayCoordinates(canvas3D, position, 2);

			// invert the y and create the mouse position point
			double[] mousePos = { e.getX(),
					canvas3D.getRenderer().GetSize()[1] - e.getY(), depth };

			// translate this point in world coordinate
			double[] worldPos = transformDisplayPointToWorld(canvas3D, mousePos);
			worldPos = pointToNodeScale(worldPos, this.pixelSizeX,
					this.pixelSizeY, this.pixelSizeZ);

			// modify the control point position in nodes_
			nodes_.get(indexOfControlPoint(controlPoint)).setPosition(
					worldPos[0], worldPos[1], worldPos[2]);
		}

		// ----------------------------------------------------------------------------

		// 3D VTK INTERACTION:
		private void setPositionFromVector(Canvas3D canvas3D, double[] vector,
				int i) {
			vtkActor controlPoint = nodesActors.get(i);
			// scale the point
			double[] position = controlPoint.GetMapper().GetCenter();

			// get the z depth of the control point
			double[] point = { getInDisplayCoordinates(canvas3D, position, 0),
					getInDisplayCoordinates(canvas3D, position, 1),
					getInDisplayCoordinates(canvas3D, position, 2) };

			// invert the y and create the mouse position point
			double[] pos = { point[0] + vector[0], point[1] - vector[1],
					point[2] };

			// translate this point in world coordinate
			double[] worldPos = transformDisplayPointToWorld(canvas3D, pos);
			worldPos = pointToNodeScale(worldPos, this.pixelSizeX,
					this.pixelSizeY, this.pixelSizeZ);

			// modify the control point position in nodes_
			nodes_.get(indexOfControlPoint(controlPoint)).setPosition(
					worldPos[0], worldPos[1], worldPos[2]);
		}

	}
}
