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

import icy.canvas.IcyCanvas;
import icy.canvas.IcyCanvas2D;
import icy.gui.viewer.Viewer;
import icy.painter.Anchor2D;
import icy.sequence.Sequence;
import icy.type.point.Point5D;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.event.MouseEvent;
import java.awt.geom.Line2D;
import java.awt.geom.Point2D;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

import org.w3c.dom.Node;

import plugins.kernel.canvas.VtkCanvas;
import vtk.vtkActor;
import vtk.vtkRenderer;

/**
 * 
 * @version May 3, 2014
 * 
 * @author Nicolas Chenouard (nicolas.chenouard@gmail.com)
 * @author Ricard Delgado-Gonzalo (ricard.delgado@gmail.com)
 */
public class Anchor3DMSBrain extends Anchor2D {

	private double z = 0d;
	private float stroke_ = 1f;
	private boolean isEditable_ = true;
	private boolean isBeingRemoved = false;
	private final boolean isVtkActorInitialized = false;
	private final vtkActor sphereActor = null;

	private final Lock screenUpdateLock = new ReentrantLock();
	private final Condition updateCondition = screenUpdateLock.newCondition();

	private double pixelSizeX = 1d;
	private double pixelSizeY = 1d;
	private double pixelSizeZ = 1d;

	private double unsyncX, unsyncY, unsyncZ;

	private final Thread updateThread_ = new Thread() {
		@Override
		public void run() {
			while (isAlive() && !this.isInterrupted()) {
				screenUpdateLock.lock();
				try {
					try {
						updateCondition.await();
						sphereActor.SetPosition(unsyncX * pixelSizeX, unsyncY
								* pixelSizeY, unsyncZ * pixelSizeZ);
						unsyncX = Anchor3DMSBrain.this.getX();
						unsyncY = Anchor3DMSBrain.this.getY();
						unsyncZ = Anchor3DMSBrain.this.getZ();
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

	public Anchor3DMSBrain(double x, double y, double z, Color color,
			Color selectedColor) {
		super(x, y, color, selectedColor);
		this.z = z;
		updateThread_.start();
	}

	// ----------------------------------------------------------------------------

	public Anchor3DMSBrain() {
		super();
		this.z = 0;
		updateThread_.start();
	}

	// ----------------------------------------------------------------------------

	// @Override
	// public void remove(Sequence sequence) {
	// sequence.addPainter(this);
	// sequence.removePainter(this);
	// }

	// ----------------------------------------------------------------------------

	public void setStroke(float stroke) {
		stroke_ = stroke;
	}

	// ----------------------------------------------------------------------------

	public double getZ() {
		return this.z;
	}

	// ----------------------------------------------------------------------------

	@Override
	public void setPosition(double x, double y) {
		this.setPosition(x, y, this.getZ());
	}

	// ----------------------------------------------------------------------------

	@Override
	public void setPosition(Point2D p) {
		this.setPosition(p.getX(), p.getY(), this.getZ());
	}

	// ----------------------------------------------------------------------------

	/**
	 * returns true is the position has changed
	 * */
	public boolean setPositionNoUpdate(final double x, final double y,
			final double z) {
		if ((this.getX() != x) || (this.getY() != y) || (this.getZ() != z)) {
			try {
				if (isVtkActorInitialized) {
					threadedUpdate();
					this.z = z;
					position.x = x;
					position.y = y;
				} else {
					this.z = z;
					super.setPosition(x, y);
				}
			} finally {
			}
			return true;
		} else {
			return false;
		}
	}

	// ----------------------------------------------------------------------------

	/**
	 * returns true is the position has changed
	 * */
	public boolean setPosition(final double x, final double y, final double z) {
		if ((this.getX() != x) || (this.getY() != y) || (this.getZ() != z)) {
			try {
				if (isVtkActorInitialized) {
					threadedUpdate();
					this.z = z;
					super.setPosition(x, y);// this will call some updates
				} else {
					this.z = z;
					super.setPosition(x, y);
				}
			} finally {

			}
			return true;
		} else {
			return false;
		}
	}

	// ----------------------------------------------------------------------------

	@Override
	public void paint(Graphics2D g, Sequence sequence, IcyCanvas canvas) {
		if (isBeingRemoved) {
			return;
		}
		if (canvas instanceof IcyCanvas2D) {
			if (isEditable_) {
				final double adjRayX = canvas.canvasToImageLogDeltaX(ray);
				final double adjRayY = canvas.canvasToImageLogDeltaY(ray);
				g.setStroke(new BasicStroke(stroke_));
				if (this.selected) {
					g.setColor(this.selectedColor);
				} else {
					g.setColor(this.color);
				}
				g.draw(new Line2D.Double(position.x - adjRayX, position.y,
						position.x + adjRayX, position.y));
				g.draw(new Line2D.Double(position.x, position.y - adjRayY,
						position.x, position.y + adjRayY));
			}

		}
	}

	// ----------------------------------------------------------------------------

	public void setEditable(boolean editable) {
		isEditable_ = editable;
	}

	// ----------------------------------------------------------------------------

	public void remove3DPainters(Sequence sequence) {
		isBeingRemoved = true;
		if (isVtkActorInitialized) {
			for (Viewer v : sequence.getViewers()) {
				IcyCanvas canvas = v.getCanvas();
				if (canvas instanceof VtkCanvas) {
					final VtkCanvas canvas3d = (VtkCanvas) canvas;
					vtkRenderer renderer = canvas3d.getRenderer();
					renderer.RemoveActor(sphereActor);
				}
			}
		}
	}

	// ----------------------------------------------------------------------------

	@Override
	public boolean saveToXML(Node node) {
		// TODO
		return false;
	}

	// ----------------------------------------------------------------------------

	@Override
	public boolean loadFromXML(Node node) {
		// TODO
		return false;
	}

	// ----------------------------------------------------------------------------

	public void setPixelSize(double pixelSizeX, double pixelSizeY,
			double pixelSizeZ) {
		this.pixelSizeX = pixelSizeX;
		this.pixelSizeY = pixelSizeY;
		this.pixelSizeZ = pixelSizeZ;
		if (isVtkActorInitialized) {
			threadedUpdate();
		}
	}

	// ----------------------------------------------------------------------------
	// MOUSE METHODS

	@Override
	public void mousePressed(MouseEvent e, Point5D.Double imagePoint,
			IcyCanvas canvas) {
		if (canvas instanceof IcyCanvas2D) {
			if (isEditable_) {
				super.mousePressed(e, imagePoint, canvas);
			}
		}
	}

	// ----------------------------------------------------------------------------

	@Override
	public void mouseDrag(MouseEvent e, Point5D.Double imagePoint,
			IcyCanvas canvas) {
		if (canvas instanceof IcyCanvas2D) {
			if (isEditable_) {
				if (isSelected()) {
					final double dx = imagePoint.getX() - position.x;
					final double dy = imagePoint.getY() - position.y;
					translate(dx, dy);
					e.consume();
				} else {
					super.mouseDrag(e, imagePoint, canvas);
				}
			}
		}
	}

	// ----------------------------------------------------------------------------

	@Override
	public void mouseMove(MouseEvent e, Point5D.Double imagePoint,
			IcyCanvas canvas) {
		if (canvas instanceof IcyCanvas2D) {
			if (isEditable_) {
				super.mouseMove(e, imagePoint, canvas);
			}
		}
	}

	// ----------------------------------------------------------------------------

	@Override
	public void mouseReleased(MouseEvent e, Point5D.Double imagePoint,
			IcyCanvas canvas) {
		if (canvas instanceof IcyCanvas2D) {
			if (isEditable_) {
				super.mouseReleased(e, imagePoint, canvas);
			}
		}
	}

	// ============================================================================
	// PRIVATE METHODS

	private void threadedUpdate() {
		if (screenUpdateLock.tryLock()) {
			try {
				updateCondition.signalAll();
			} finally {
				screenUpdateLock.unlock();
			}
		}
	}
}
