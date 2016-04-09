package com.sli.deeplearning_experiment;
import java.util.Random;

import org.jzy3d.analysis.AbstractAnalysis;
import org.jzy3d.analysis.AnalysisLauncher;
import org.jzy3d.chart.factories.AWTChartComponentFactory;
import org.jzy3d.colors.Color;
import org.jzy3d.colors.ColorMapper;
import org.jzy3d.colors.colormaps.ColorMapRainbow;
import org.jzy3d.contour.DefaultContourColoringPolicy;
import org.jzy3d.contour.MapperContourPictureGenerator;
import org.jzy3d.maths.Coord3d;
import org.jzy3d.maths.Range;
import org.jzy3d.plot3d.builder.Builder;
import org.jzy3d.plot3d.builder.Mapper;
import org.jzy3d.plot3d.builder.concrete.OrthonormalGrid;
import org.jzy3d.plot3d.primitives.Scatter;
import org.jzy3d.plot3d.primitives.Shape;
import org.jzy3d.plot3d.primitives.axes.ContourAxeBox;
import org.jzy3d.plot3d.primitives.selectable.SelectableScatter;
import org.jzy3d.plot3d.rendering.canvas.Quality;
import org.jzy3d.plot3d.rendering.legends.colorbars.AWTColorbarLegend;


public class ContourPlot extends AbstractAnalysis{
	 private Range theRangeX;
    private int theStepX;
    private Range theRangeY;
    private int theStepY;
    private Mapper theMapper;
    private double theTheta0;
    private double theTheta1;
    public ContourPlot(Range rangeX, int stepX, Range rangeY, int stepY, Mapper mapper, double theta0, double theta1){
    	this.theRangeX = rangeX;
    	this.theStepX = stepX;
    	this.theRangeY = rangeY;
    	this.theStepY = stepY;
    	this.theMapper = mapper;
    	this.theTheta0 = theta0;
    	this.theTheta1 = theta1;
    	
    }

    public void plot() throws Exception{
    	AnalysisLauncher.open(this);
    }
    
	
	public void init() throws Exception {
	

		// Define range and precision for the function to plot
		
		final Shape surface = (Shape)Builder.buildOrthonormal(new OrthonormalGrid(theRangeX, theStepX, theRangeY, theStepY), theMapper);
		ColorMapper myColorMapper=new ColorMapper(new ColorMapRainbow(), surface.getBounds().getZmin(), surface.getBounds().getZmax(), new Color(1,1,1,.4f)); 
		surface.setColorMapper(myColorMapper);
		surface.setFaceDisplayed(true);
		surface.setWireframeDisplayed(false);
		surface.setWireframeColor(Color.BLACK);

	
		chart = AWTChartComponentFactory.chart(Quality.Advanced, getCanvasType());
		chart.getView().setAxe(new ContourAxeBox(chart.getView().getAxe().getBoxBounds()));
		ContourAxeBox cab = (ContourAxeBox) chart.getView().getAxe();
		 
		MapperContourPictureGenerator contour = new MapperContourPictureGenerator(theMapper, theRangeX, theRangeY);
		
		cab.setContourImg( contour.getContourImage(new DefaultContourColoringPolicy(myColorMapper), theStepX, theStepY, 40), theRangeX, theRangeY);
		
		// Add the surface and its colorbar
		chart.addDrawable(surface);
		surface.setLegend(new AWTColorbarLegend(surface, 
				chart.getView().getAxe().getLayout().getZTickProvider(), 
				chart.getView().getAxe().getLayout().getZTickRenderer()));
		surface.setLegendDisplayed(true); // opens a colorbar on the right part of the display
		
		Scatter ss = new Scatter(new Coord3d[]{new Coord3d(theTheta0, theTheta1, 1.0)}, new Color[]{Color.BLUE},10f);
		chart.addDrawable(ss);
		
	    
	}
}

