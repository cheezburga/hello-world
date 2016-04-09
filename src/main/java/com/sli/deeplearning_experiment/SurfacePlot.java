package com.sli.deeplearning_experiment;


import java.util.Random;

import org.jzy3d.analysis.AbstractAnalysis;
import org.jzy3d.analysis.AnalysisLauncher;
import org.jzy3d.chart.factories.AWTChartComponentFactory;
import org.jzy3d.colors.Color;
import org.jzy3d.colors.ColorMapper;
import org.jzy3d.colors.colormaps.ColorMapRainbow;
import org.jzy3d.maths.Coord3d;
import org.jzy3d.maths.Range;
import org.jzy3d.plot3d.builder.Builder;
import org.jzy3d.plot3d.builder.Mapper;
import org.jzy3d.plot3d.builder.concrete.OrthonormalGrid;
import org.jzy3d.plot3d.primitives.Scatter;
import org.jzy3d.plot3d.primitives.Shape;
import org.jzy3d.plot3d.rendering.canvas.Quality;

public class SurfacePlot extends AbstractAnalysis {

    private Range theRangeX;
    private int theStepX;
    private Range theRangeY;
    private int theStepY;
    private Mapper theMapper;
    
    public SurfacePlot(Range rangeX, int stepX, Range rangeY, int stepY, Mapper mapper){
    	
    	this.theRangeX = rangeX;
    	this.theStepX = stepX;
    	this.theRangeY = rangeY;
    	this.theStepY = stepY;
    	this.theMapper = mapper;
    }
    
    public void plot() throws Exception{
    	AnalysisLauncher.open(this);
    }
    
    
    public void init() {
  

        // Create the object to represent the function over the given range.
        final Shape surface = Builder.buildOrthonormal(new OrthonormalGrid(theRangeX, theStepX, theRangeY, theStepY), theMapper);
        surface.setColorMapper(new ColorMapper(new ColorMapRainbow(), surface.getBounds().getZmin(), surface.getBounds().getZmax(), new Color(1, 1, 1, .5f)));
        surface.setFaceDisplayed(true);
        surface.setWireframeDisplayed(true);
        
        // Create a chart
        chart = AWTChartComponentFactory.chart(Quality.Advanced, getCanvasType());
        chart.getScene().getGraph().add(surface);
        chart.getAxeLayout().setXAxeLabel("theta0");
        chart.getAxeLayout().setYAxeLabel("theta1");
        chart.getAxeLayout().setZAxeLabel("cost");
        
   
    }
}
