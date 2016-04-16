package com.sli.deeplearning_experiment;
import java.util.ArrayList;
import java.util.List;

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
import org.jzy3d.plot3d.rendering.canvas.Quality;
import org.jzy3d.plot3d.rendering.legends.colorbars.AWTColorbarLegend;
import org.nd4j.linalg.api.ndarray.INDArray;


public class ContourPlot2 extends AbstractAnalysis{
	 private Range theRangeX;
    private int theStepX;
    private Range theRangeY;
    private int theStepY;
    private Mapper theMapper;
    private INDArray x1;
    private INDArray x2;
    private INDArray y;
    public ContourPlot2(Range rangeX, int stepX, Range rangeY, int stepY, Mapper mapper, INDArray x1, INDArray x2, INDArray y){
    	this.theRangeX = rangeX;
    	this.theStepX = stepX;
    	this.theRangeY = rangeY;
    	this.theStepY = stepY;
    	this.theMapper = mapper;
    	this.x1 = x1;
    	this.x2 = x2;
    	this.y = y;
    	
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
		
		cab.setContourImg( contour.getContourImage(new DefaultContourColoringPolicy(myColorMapper), theStepX, theStepY, 100), theRangeX, theRangeY);
		
		// Add the surface and its colorbar
		chart.addDrawable(surface);
		surface.setLegend(new AWTColorbarLegend(surface, 
				chart.getView().getAxe().getLayout().getZTickProvider(), 
				chart.getView().getAxe().getLayout().getZTickRenderer()));
		surface.setLegendDisplayed(true); // opens a colorbar on the right part of the display
		
		List<Coord3d> cd3ds = new ArrayList<Coord3d>();
		List<Color> clrs = new ArrayList<Color>();
		for(int i=0; i<x1.size(0); i++){
			cd3ds.add(new Coord3d(x1.getRow(i).getDouble(0), x2.getRow(i).getDouble(0), 1.0));
			if(y.getRow(i).getDouble(0)>0.0){
				clrs.add(Color.RED);
			}else{
				clrs.add(Color.YELLOW);
			}
		}
		Scatter ss = new Scatter(cd3ds.toArray(new Coord3d[cd3ds.size()]), clrs.toArray(new Color[clrs.size()]),10f);
		chart.addDrawable(ss);
		
	    
	}
}

