package plugins.big.vascular.utils;

import icy.canvas.IcyCanvas;
import icy.canvas.IcyCanvas2D;
import icy.painter.Overlay;
import icy.sequence.Sequence;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.event.KeyEvent;
import java.awt.event.MouseEvent;
import java.awt.geom.Line2D;
import java.awt.geom.Point2D;
import java.util.ArrayList;

import plugins.big.bigsnakeutils.icy.snake3D.Snake3DScale;
import plugins.big.vascular.roi.Anchor3D;

public class OrthoSlice extends Overlay {
	public Sequence sequence;
	
	/** Controls the number of scales shown on the screen. */
	private int scaleSubsamplingFactor_ = 1;	
	
	//private float stroke;
	private String view = "";	
	
	/** Group of all polylines forming the skin of the snake. */
	private ArrayList<Snake3DScale> scales_ = new ArrayList<Snake3DScale>();
	
	private ArrayList<Anchor3D> nodes_ = new ArrayList<Anchor3D>();
	
	public OrthoSlice(Sequence sequence, String view, String imgFunction){
		super(view+" - "+imgFunction);
		this.sequence = sequence;
		this.view = view;
		sequence.addOverlay(this);
		
		// hysteresis images
		if (view.equals("YZ") && imgFunction.equals("HYST"))
			Const.hystXInit = true;
		if (view.equals("XZ") && imgFunction.equals("HYST"))
			Const.hystYInit = true;
		
		// locNorm images
		if (view.equals("YZ") && imgFunction.equals("LOC_NORM"))
			Const.locNormXInit = true;
		if (view.equals("XZ") && imgFunction.equals("LOC_NORM"))
			Const.locNormYInit = true;
		if (view.equals("XY") && imgFunction.equals("LOC_NORM"))
			Const.locNormZInit = true;
		
		// orig images
		if (view.equals("YZ") && imgFunction.equals("ORIG"))
			Const.origXInit = true;
		if (view.equals("XZ") && imgFunction.equals("ORIG"))
			Const.origYInit = true;
		if (view.equals("XY") && imgFunction.equals("ORIG"))
			Const.origZInit = true;
		
		
	}
	
	public void scalesChanged(ArrayList<Snake3DScale> scales, ArrayList<Anchor3D> nodes){
		
		scales_ = scales;
		nodes_ = nodes;
		//System.out.println("OrthoSlice.scalesChanged");
		
		painterChanged();		
	}

	@Override
	public void paint(Graphics2D g, Sequence sequence, IcyCanvas canvas) {

		//System.out.println("OrthoSlice.paint()");
		
		if (canvas instanceof IcyCanvas2D) {
			final Graphics2D g2 = (Graphics2D) g.create();
			//float stroke = (float) getAdjustedStroke(canvas);
			// draw scales
			if (scales_ != null) {					
				
				double minZnodes = sequence.getSizeZ();
				/*
				double minZnodes = 0;
				if (view.equals("XY"))
					minZnodes = sequence.getSizeZ();
				if (view.equals("YZ"))
					minZnodes = sequence.getSizeX();
				if (view.equals("XZ"))
					minZnodes = sequence.getSizeY();
				*/
				
				double maxZnodes = 0;				
				
				for (Anchor3D node : nodes_) {											
					
					
					if (view.equals("XY")){
						if (node.getZ() > maxZnodes) 
							maxZnodes = node.getZ();
						if (node.getZ() < minZnodes)
							minZnodes = node.getZ();					
					}
					if (view.equals("YZ")){
						if (node.getX() > maxZnodes) 
							maxZnodes = node.getX();
						if (node.getX() < minZnodes)
							minZnodes = node.getX();					
					}
					if (view.equals("XZ")){
						if (node.getY() > maxZnodes) 
							maxZnodes = node.getY();
						if (node.getY() < minZnodes)
							minZnodes = node.getY();					
					}	
					
				}				
				
				int zSlide = canvas.getPositionZ();					
				float maxDist = Math.max(zSlide, sequence.getSizeZ() - zSlide) / 5;
				
				//g2.setStroke(new BasicStroke((float) stroke / 2));

				for (int i = 0; i < scales_.size(); i += scaleSubsamplingFactor_) {
					Snake3DScale scale = scales_.get(i);
					if (scale != null) {
						Color c = scale.getColor();
						if (c != null)
							g2.setColor(c);
						double[] prevPos = new double[3];
						boolean firstOne = true;
						for (double[] pos : scale.getCoordinates()) {
							if (firstOne) {
								firstOne = false;
							} else {
								//float meanZ = (float) (pos[2] + prevPos[2]) / 2.f;
								float meanZ = 0;
								if (view.equals("XY"))
									meanZ = (float) (pos[2] + prevPos[2]) / 2.f;
								if (view.equals("YZ"))
									meanZ = (float) (pos[0] + prevPos[0]) / 2.f;
								if (view.equals("XZ"))
									meanZ = (float) (pos[1] + prevPos[1]) / 2.f;
								
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
								
								// shift by 0.5 because coordinate of a
								// pixel corresponds to its top-left corner
								/*
								g2.draw(new Line2D.Double(prevPos[0] + 0.5,
										prevPos[1] + 0.5, pos[0] + 0.5,
										pos[1] + 0.5));
										*/
								if (view.equals("XY")){
									g2.draw(new Line2D.Double(prevPos[0] + 0.5,
											prevPos[1] + 0.5, pos[0] + 0.5,
											pos[1] + 0.5));
								}
								if (view.equals("YZ")){
									g2.draw(new Line2D.Double(prevPos[1] + 0.5,
											prevPos[2] + 0.5, pos[1] + 0.5,
											pos[2] + 0.5));
								}
								if (view.equals("XZ")){
									g2.draw(new Line2D.Double(prevPos[0] + 0.5,
											prevPos[2] + 0.5, pos[0] + 0.5,
											pos[2] + 0.5));
								}
								
							}
							prevPos[0] = pos[0];
							prevPos[1] = pos[1];
							prevPos[2] = pos[2];
						}
						if (scale.isClosed()) {
							double[] pos = scale.getCoordinates()[0];
							//float meanZ = (float) (pos[2] + prevPos[2]) / 2.f;
							float meanZ = 0;
							if (view.equals("XY"))
								meanZ = (float) (pos[2] + prevPos[2]) / 2.f;
							if (view.equals("YZ"))
								meanZ = (float) (pos[0] + prevPos[0]) / 2.f;
							if (view.equals("XZ"))
								meanZ = (float) (pos[1] + prevPos[1]) / 2.f;
							
							
							float transparency = Math.max(0,
									1 - Math.abs(meanZ - zSlide) / maxDist);
							float green = (float) Math
									.abs((meanZ - minZnodes)
											/ (maxZnodes - minZnodes));
							green = Math.max(green, 0);
							green = Math.min(green, 1);
							g2.setColor(new Color(1.f, green, 1.f - green,
									transparency));
							// shift by 0.5 because coordinate of a pixel
							// corresponds to its top-left corner
							/*
							g2.draw(new Line2D.Double(prevPos[0] + 0.5,
									prevPos[1] + 0.5, pos[0] + 0.5,
									pos[1] + 0.5));			
							*/
							if (view.equals("XY")){
								g2.draw(new Line2D.Double(prevPos[0] + 0.5,
										prevPos[1] + 0.5, pos[0] + 0.5,
										pos[1] + 0.5));
							}
							if (view.equals("YZ")){
								g2.draw(new Line2D.Double(prevPos[1] + 0.5,
										prevPos[2] + 0.5, pos[1] + 0.5,
										pos[2] + 0.5));
							}
							if (view.equals("XZ")){
								g2.draw(new Line2D.Double(prevPos[0] + 0.5,
										prevPos[2] + 0.5, pos[0] + 0.5,
										pos[2] + 0.5));
							}
						}							
					}
				}
			}
			
			g2.dispose();
		}
	
		
	}

	@Override
	public void mousePressed(MouseEvent e, Point2D imagePoint, IcyCanvas canvas) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void mouseReleased(MouseEvent e, Point2D imagePoint, IcyCanvas canvas) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void mouseClick(MouseEvent e, Point2D imagePoint, IcyCanvas canvas) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void mouseMove(MouseEvent e, Point2D imagePoint, IcyCanvas canvas) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void mouseDrag(MouseEvent e, Point2D imagePoint, IcyCanvas canvas) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void keyPressed(KeyEvent e, Point2D imagePoint, IcyCanvas canvas) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void keyReleased(KeyEvent e, Point2D imagePoint, IcyCanvas canvas) {
		// TODO Auto-generated method stub
		
	}

}
