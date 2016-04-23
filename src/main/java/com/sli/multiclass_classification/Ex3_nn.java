package com.sli.multiclass_classification;
import java.awt.Color;
import java.awt.Shape;
import java.io.IOException;
import java.io.Serializable;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import javax.swing.JFrame;
import javax.swing.WindowConstants;

import org.apache.commons.math3.analysis.DifferentiableMultivariateVectorFunction;
import org.apache.commons.math3.analysis.MultivariateMatrixFunction;
import org.apache.commons.math3.optimization.PointVectorValuePair;
import org.apache.commons.math3.optimization.general.LevenbergMarquardtOptimizer;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.util.ShapeUtilities;
import org.jzy3d.maths.Range;
import org.jzy3d.plot3d.builder.Mapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.springframework.core.io.ClassPathResource;

import com.sli.deeplearning_experiment.ColorPan;
import com.sli.deeplearning_experiment.ContourPlot2;
import com.sli.deeplearning_experiment.CostMapper;
import com.sli.deeplearning_experiment.SurfacePlot;

import de.jungblut.classification.regression.LogisticRegressionCostFunction;
import de.jungblut.math.DoubleMatrix;
import de.jungblut.math.DoubleVector;
import de.jungblut.math.dense.DenseDoubleMatrix;
import de.jungblut.math.dense.DenseDoubleVector;
import de.jungblut.math.minimize.CostFunction;
import de.jungblut.math.minimize.Fmincg;

public class Ex3_nn {
	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		
		System.out.println("Loading and Visualizing Data ...\n");
		int input_layer_size = 400;//20x20 Input Images of Digits
		int num_labels = 10;//10 labels, from 1 to 10 (note that we have mapped "0" to label 10)
		
		//=========== Part 1: Loading and Visualizing Data =============
		//We start the exercise by first loading and visualizing the dataset.
		//You will be working with a dataset that contains handwritten digits.
		
		
		
		//The first two columns contains the exam scores and the third column
		//contains the label.
		int totalSamples = 5000;
		//for some reason when we port octave file over, it ended up with an extra column on the right.
		int[] colIndices = new int[400];
		for(int i=0; i<400; i++){
			colIndices[i]=i;
		}
		INDArray x_5000x400 = load("machine-learning/multiclass_classification/ex3data1.txt", totalSamples, " ").getColumns(colIndices);
		System.out.println(x_5000x400.rows()+" col:"+x_5000x400.columns());
//		double[] cols = x_5000x400.getRow(0).dup().data().asDouble();
//		for(int col=0; col<400; col++){
//			System.out.println("col:"+col+" value:"+cols[col]);
//		}
		
		INDArray y_5000x1 = load("machine-learning/multiclass_classification/ex3data2.txt", totalSamples, " ");
	
		System.out.println("Y row:"+y_5000x1.rows()+" col:"+y_5000x1.columns()+"\n"+y_5000x1);
		//Randomly select 100 data points to display
		
		
		List<Integer> arrList = new ArrayList<Integer>();
	    for (int i = 0; i < totalSamples; i++) {
	        arrList.add(i);
	    }
	   
	    Collections.shuffle(arrList);
	    int[] arr = new int[100];
	    for(int i=0; i< 100; i++){
	    	arr[i] = arrList.get(i);
	    }
	   
//	    System.out.println(Arrays.toString(arr));
	    INDArray x_100x400 = x_5000x400.getRows(arr);
	    displayData(x_100x400.dup());
	    
	    INDArray y_100x1 = y_5000x1.getRows(arr);
	    
		//================ Part 2: Loading Pameters ================
		//In this part of the exercise, we load some pre-initialized
		// neural network parameters
		
		
		
		
		System.out.println("\nLoading Saved Neural Network Parameters ...\n");
		colIndices = new int[401];
		for(int i=0; i<401; i++){
			colIndices[i]=i;
		}
		INDArray theta1_25x401 = load("machine-learning/multiclass_classification/ex3weights1.txt", 25, " ").getColumns(colIndices);//do not know why it has 402 columnes
		//System.out.println(theta1_25x402.columns());
		
		colIndices = new int[26];
		for(int i=0; i<26; i++){
			colIndices[i]=i;
		}
		INDArray theta2_10x26 = load("machine-learning/multiclass_classification/ex3weights2.txt", 10, " ").getColumns(colIndices);//do not know why it has 27 columnes;
		//System.out.println(theta2_10x27.columns());
		{
			int correctCount = 0;
			int incorrectCount = 0;
			INDArray activationAtOutputLayer_5000x10 = predict(theta1_25x401, theta2_10x26, x_5000x400);
			INDArray maxValues_5000x1 = activationAtOutputLayer_5000x10.max(1);
			for(int row=0; row<5000; row++){
				int label = y_5000x1.getRow(row).getInt(0);
				int col = label-1;//converting label into column index
				if(label == 10){
					col = 9;
				}
				boolean isCorrect = (activationAtOutputLayer_5000x10.getDouble(row, col) == maxValues_5000x1.getRow(row).dup().getDouble(0));
				if(isCorrect){
					correctCount++;
				}else{
					incorrectCount++;
				}
				System.out.println("pred@"+row+"/"+4999+": "+isCorrect);
			}
			System.out.println("\nTraining Set Accuracy: "+(double)correctCount/(correctCount+incorrectCount));
			
		}
		
		
		INDArray activationAtOutputLayer_100x10 = predict(theta1_25x401, theta2_10x26, x_100x400);
		INDArray maxValues_100x1 = activationAtOutputLayer_100x10.max(1);
		int correctCount = 0;
		int incorrectCount = 0;
		for(int row=0; row<100; row++){
			int label = y_100x1.getRow(row).getInt(0);
			int col = label-1;//converting label into column index
			if(label == 10){
				col = 9;
			}
			boolean isCorrect = (activationAtOutputLayer_100x10.getDouble(row, col) == maxValues_100x1.getRow(row).dup().getDouble(0));
			if(isCorrect){
				correctCount++;
			}else{
				incorrectCount++;
			}
			System.out.println("pred@"+row+"/"+99+": "+isCorrect);
		}
		System.out.println("\nTraining Set Accuracy: "+(double)correctCount/(correctCount+incorrectCount));
		
		
	}
	
	private static INDArray predict(INDArray theta1_25x401, INDArray theta2_10x26, INDArray x){
		int m = x.rows();
		INDArray X = Nd4j.hstack(Nd4j.ones(m, 1), x);//5000x401
		INDArray activationAtHiddenLayer_5000x25 = Transforms.sigmoid(X.mmul(theta1_25x401.transpose()),true);
		INDArray activationAtOutputLayer_5000x10 = Transforms.sigmoid(Nd4j.hstack(Nd4j.ones(m, 1),activationAtHiddenLayer_5000x25).mmul(theta2_10x26.transpose()));
		System.out.println(activationAtOutputLayer_5000x10);
		return activationAtOutputLayer_5000x10;
		
	}
	
	private static INDArray predictOneVsAll(INDArray all_theta, INDArray x){
		int num_lables = all_theta.rows();
		int m = x.rows();
		INDArray X = Nd4j.hstack(Nd4j.ones(m, 1), x);
		INDArray k = X.mmul(all_theta.transpose());//5000x401 * 401x10 = 5000x10
		INDArray o = k.subColumnVector(k.max(1));//5000x10 - 5000x1
		System.out.println("x"+o);
		return o;
	}
	
	private static INDArray oneVsAll(INDArray X, INDArray y, int num_labels, double lambda){
		int m = X.rows();
		int n = X.columns();
		INDArray all_theta_10x401 = Nd4j.zeros(num_labels, n+1);
		INDArray x_5000x401 = Nd4j.hstack(Nd4j.ones(m, 1),X);
		for(int c=0; c<num_labels; c++){
			//INDArray initial_theta_401x1 = Nd4j.zeros(n+1, 1);
			//10 labels, from 1 to 10 (note that we have mapped "0" to label 10)
			double[] theta = null;
			if(c == 0){
				theta = fmincg(x_5000x401, y.eq(10), lambda);
			}else{
				theta = fmincg(x_5000x401, y.eq(c), lambda);
			}
			//System.out.println(Arrays.toString(theta));
			all_theta_10x401.getRow(c).assign(Nd4j.create(theta));
			System.out.println("fmincg @"+c +"/"+9);
		}
		return all_theta_10x401;
	}
	
	
	private static double[] fmincg(INDArray X, INDArray y, double lambda){
		DoubleVector startingTheta = new DenseDoubleVector(X.size(1), 0);
		
		double[][] dm_x_arr = new double[X.rows()][X.columns()];
		for(int row=0; row<X.rows(); row++){
			for(int col=0; col<X.columns(); col++){
				dm_x_arr[row][col] = X.getDouble(row, col);
			}
		}
		DoubleMatrix dm_x = new DenseDoubleMatrix(dm_x_arr);

		double[][] dm_y_arr = new double[y.rows()][y.columns()];
		for(int row=0; row<y.rows(); row++){
			for(int col=0; col<y.columns(); col++){
				dm_y_arr[row][col] = y.getDouble(row, col);
			}
		}
		DoubleMatrix dm_y = new DenseDoubleMatrix(dm_y_arr).transpose();
		
		CostFunction cf = new LogisticRegressionCostFunction(dm_x,dm_y,lambda);
		DoubleVector dv = Fmincg.minimizeFunction(cf, startingTheta, 50, true);
		//System.out.println("XX"+Arrays.toString(dv.toArray()));
		//System.out.println("cost:"+cf.evaluateCost(dv).getCost());
		return dv.toArray();
	}
	private static void displayData(INDArray x_100x400){
		ColorPan cp = new ColorPan(x_100x400);
		cp.plot();
	}
	private static INDArray predict(INDArray theta, INDArray X){
		int m = X.size(0);
		INDArray p = Nd4j.create(m, 1);
		for(int i=0; i<m; i++){
			double hypo = sigmoid(theta.transpose().mmul(X.getRow(i).transpose())).getDouble(0);
			if(hypo >= 0.5){
				p.getRow(i).assign(1);
			}else{
				p.getRow(i).assign(0);
			}
		}
		return p;
	}
	
	private static class SigmoidProblem implements DifferentiableMultivariateVectorFunction, Serializable {
		INDArray X;
		INDArray y;
		double lambda;
		public SigmoidProblem(final INDArray theX, final INDArray theY, final double lambda) {
			this.X = theX;
			this.y = theY;
			this.lambda = lambda;
		}

		public double[] value(double[] thetaArr) throws IllegalArgumentException {
			INDArray theta = Nd4j.create(thetaArr).transpose();// 3x1
			int m = X.size(0);
			double[] values = new double[m];
			for (int i = 0; i < m; i++) {
				INDArray row = X.getRow(i);
				INDArray hypo = sigmoid(theta.transpose().mmul(row.transpose()));// hypo
																					// =
																					// sigmoid(theta'
																					// *
																					// X(i,:)')

				values[i] = hypo.getDouble(0);// sum = sum - y(i)*log(hypo) -
												// (1-y(i))*log(1-hypo)
				// values[i] = (theta[0] * x.get(i) + theta[1]) * x.get(i) +
				// theta[2];
			}
			return values;
		}

		public double[] calculateTarget() {
			return y.data().asDouble();
		}

		private double[][] jacobian(double[] thetaArr){
			 INDArray theta = Nd4j.create(thetaArr).transpose();//3x1
			 int m = X.size(0);
			 int n = X.size(1);
			 double[][] jacobian = new double[m][n];
			 for(int i=0; i < m; i++){
				 INDArray row = X.getRow(i).dup();
				 INDArray hypo = sigmoid(theta.transpose().mmul(row.transpose()));//hypo = sigmoid(theta' * X(i,:)')
				 for(int j=0; j < n; j++){
					 jacobian[i][j] = hypo.getDouble(0)*row.getColumn(j).dup().getDouble(0);
					 
				 }
				 
			 }
			 return jacobian;
		 }

		public MultivariateMatrixFunction jacobian() {
			return new MultivariateMatrixFunction() {

				public double[][] value(double[] point) {
					return jacobian(point);
				}
			};
		}



	}

	private static double[] psudoFminunc(INDArray theX, INDArray theY, INDArray initialTheta, double lambda) {
		SigmoidProblem problem = new SigmoidProblem(theX, theY, lambda);
		LevenbergMarquardtOptimizer optimizer = new LevenbergMarquardtOptimizer();

		 
		 int m = theY.size(0);
		 final double[] weights = Nd4j.ones(m).data().asDouble();
		 
		 final double[] initialSolution = initialTheta.data().asDouble();

		 PointVectorValuePair optimum = optimizer.optimize(10,
		                                                   problem,
		                                                   problem.calculateTarget(),
		                                                   weights,
		                                                   initialSolution);

		 final double[] optimalValues = optimum.getPoint();
		 
		 return optimalValues;
	}

	private static void plotDecisionBoundary(double[] thetaArr, INDArray x1, INDArray x2, INDArray y) throws Exception{
	
		INDArray theta = Nd4j.create(thetaArr).transpose();
	
		//System.out.println(z);
		contour(x1, x2, y, theta);
	}
	private static double[] linspace(double from, double to, int steps){
		double[] lin = new double[steps+1];
		double stepSize = (to - from)/steps;
		
		for(int i=0; i<lin.length; i++){
			lin[i] = stepSize*i+from;  
		}
		return lin;
	}
	/**
	% MAPFEATURE Feature mapping function to polynomial features
	%
	%   MAPFEATURE(X1, X2) maps the two input features
	%   to quadratic features used in the regularization exercise.
	%
	%   Returns a new feature array with more features, comprising of
	%   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
	%
	%   Inputs X1, X2 must be the same size
	%
	 */
	private static INDArray mapFeature(INDArray x1, INDArray x2){
		int degree = 6;
		int m = x1.size(0);
		
		INDArray out = Nd4j.ones(m, 1);
		for(int i=1; i<=degree; i++){
			for(int j=0; j<=i; j++){
				out = Nd4j.hstack(out,Transforms.pow(x1, i-j, true).mul(Transforms.pow(x2, j, true)));
			}
		}
		return out;
	}
	
	private static Pair<Double,Double[]> costFunctionReg(INDArray theta, INDArray X, INDArray y, int lambda){
		int m = y.size(0);
		double sum = 0.0;
		for(int i=0; i<m; i++){
			INDArray row = X.getRow(i);
			INDArray hypo = sigmoid(theta.transpose().mmul(row.transpose()));//hypo = sigmoid(theta' * X(i,:)'
			INDArray firstTerm = y.getRow(i).neg().mmul(Transforms.log(hypo));
			INDArray secondTermA = y.getRow(i).sub(1);
			INDArray secondTermB = Transforms.log(hypo.neg().add(1));
			INDArray secondTerm = secondTermA.mmul(secondTermB);
			sum += firstTerm.add(secondTerm).getDouble(0);//sum = sum - y(i)*log(hypo) - (1-y(i))*log(1-hypo)
		}
		double J = (double)sum/m + lambda/(2*m)*(theta.transpose().mmul(theta).getDouble(0) - theta.getDouble(0)*theta.getDouble(0));
		
		Double[] grads = new Double[theta.size(0)];
		for(int j=0; j<theta.size(0); j++){
			sum = 0.0;
			
			for(int i=0; i<m; i++){
				INDArray row = X.getRow(i);			
				INDArray hypo = sigmoid(theta.transpose().mmul(row.transpose()));//hypo = sigmoid(theta' * X(i,:)'
				INDArray firstTerm = hypo.sub(y.getRow(i));
				INDArray secondTerm = row.getColumn(j);
				sum += firstTerm.mmul(secondTerm).getDouble(0);//sum = sum + (hypo-y(i))*X(i,j)				
			}
			if(j==0){
				grads[j] = (double)sum/m ;
			}else{
				grads[j] = (double)sum/m + theta.getDouble(j)*lambda/m;
				
			}
		}
		return new Pair<Double,Double[]>(J,grads);
	}
	
	private static INDArray sigmoid(INDArray z){
		return Transforms.pow(Transforms.exp(z.neg()).add(1),-1, true);
		
	}
	private static void surf(INDArray X, INDArray y) throws Exception{
	      // Define a function to plot
       CostMapper mapper = new CostMapper(X, y);

        // Define range and precision for the function to plot
        Range rangeX = new Range(-10, 10);
        int stepX = 100;
        Range rangeY = new Range(-1,4);
        int stepY = 100;
        SurfacePlot sp = new SurfacePlot(rangeX, stepX, rangeY, stepY, mapper);
        sp.plot();
	}
	
	private static void contour(INDArray x1, INDArray x2, INDArray y, INDArray theta) throws Exception{
	      // Define a function to plot
     CostMapper2 mapper = new CostMapper2(theta);

      // Define range and precision for the function to plot
      Range rangeX = new Range(-1.0f, 1.5f);
      int stepX = 100;
      Range rangeY = new Range(-1.0f, 1.5f);
      int stepY = 100;
      ContourPlot2 sp = new ContourPlot2(rangeX, stepX, rangeY, stepY, mapper, x1, x2, y);
      sp.plot();
	}
	
	public static class CostMapper2 extends Mapper {
		private INDArray theta;
		private Map<String, Double> cached = new HashMap<String, Double>();
		public CostMapper2(INDArray theta){
			
			this.theta = theta;
		}
		@Override
		public double f(double x, double y) {
			String key = x+","+y;
			Double cache = cached.get(key);
			if(cache == null){
				
				
				cache = mapFeature(Nd4j.create(new double[]{x}), Nd4j.create(new double[]{y})).mmul(theta).getDouble(0);
				//System.out.println(cache);
				if(Math.round(cache*10.0) != 0L){
					cached.put(key, cache);
					
				}else{
					cached.put(key, 1000.0);
					cache = 1000.0;
				}
			}
			return cache; 
		}

	}
	private static INDArray[] gradientDescent(INDArray X, INDArray y, INDArray theta, double alpha, int iterations){
		int m = y.size(0);
		INDArray jHistory = Nd4j.zeros(m, 1);
		for(int i=0; i<iterations; i++){
			INDArray secondTerm = X.mmul(theta).sub(y).mul((double) alpha/m);
			theta = theta.sub(X.transpose().mmul(secondTerm));			
			jHistory.getRow(i).assign(computeCost(X, y, theta));
		}
		
		return new INDArray[]{theta,jHistory};
	}
	
	public static double computeCost(INDArray X, INDArray y, INDArray theta){
		int m = y.size(0);
		return Transforms.pow(X.mmul(theta).sub(y),2,true).sumNumber().doubleValue()/(2*m);
	}
	private static INDArray load(String filePath, int totalSamples, String delimiter) throws IOException, InterruptedException{
		
        RecordReader recordReader = new CSVRecordReader(0, delimiter);
        recordReader.initialize(new FileSplit(new ClassPathResource(filePath).getFile()));
        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader);     
        DataSet set = iterator.next(totalSamples);      
        return set.getFeatureMatrix();
	}
	
	private static void warmUpExercise(){
		INDArray tmp = Nd4j.eye(5);
		System.out.println(tmp);
	}
	
	private static void pause(){
	   System.out.println("Program paused. Press enter to continue.\n");
	   Scanner scanner = new Scanner(System.in);
	   scanner.nextLine();	
	}

	private static JFrame plotData2(INDArray x, INDArray y, INDArray lrx, INDArray lry){
		final XYSeriesCollection dataSet = new XYSeriesCollection();
		final XYSeriesCollection dataSet2 = new XYSeriesCollection();
        addSeries(dataSet,x,y,"Training Data");
        addSeries2(dataSet2,lrx,lry,"Logistic regression");

        final JFreeChart chart = ChartFactory.createScatterPlot(
                " ",      // chart title
                "Population of City in 10,000s",                        // x axis label
                "Profit in $10,000s", // y axis label
                dataSet,                    // data
                PlotOrientation.VERTICAL,
                true,                       // include legend
                true,                       // tooltips
                false                       // urls
        );

        final ChartPanel panel = new ChartPanel(chart);
        XYPlot xyPlot = chart.getXYPlot();
        XYItemRenderer renderer1 = new XYLineAndShapeRenderer(false, true);//shapes
        Shape cross = ShapeUtilities.createDiagonalCross(3, 1);
        renderer1.setSeriesShape(0, cross);
        renderer1.setSeriesPaint(1, Color.yellow);
        xyPlot.setRenderer(0, renderer1);
        xyPlot.setDataset(0, dataSet);
        
        
        
        XYItemRenderer renderer2 = new XYLineAndShapeRenderer(true, false);//lines
        renderer2.setSeriesPaint(0, Color.blue);
        xyPlot.setRenderer(1, renderer2);
        xyPlot.setDataset(1, dataSet2);
        
        final JFrame f = new JFrame();
        f.add(panel);
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        f.pack();

        f.setVisible(true);
        return f;
	}
	private static JFrame plotData(INDArray x, INDArray y){
		final XYSeriesCollection dataSet = new XYSeriesCollection();
        addSeries(dataSet,x,y,"Training Data");
      

        final JFreeChart chart = ChartFactory.createScatterPlot(
                " ",      // chart title
                "Microchip Test 1",                        // x axis label
                "Microchip Test 2", // y axis label
                dataSet,                    // data
                PlotOrientation.VERTICAL,
                true,                       // include legend
                true,                       // tooltips
                false                       // urls
        );

        final ChartPanel panel = new ChartPanel(chart);
        
        XYItemRenderer renderer = chart.getXYPlot().getRenderer();
        Shape cross = ShapeUtilities.createRegularCross(3, 1);
        renderer.setSeriesShape(0, cross);       
        renderer.setSeriesPaint(1, Color.yellow);
        final JFrame f = new JFrame();
        f.add(panel);
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        f.pack();

        f.setVisible(true);
        return f;
	}
	private static void addSeries(final XYSeriesCollection dataSet, final INDArray x, final INDArray y, final String label){
       
        final double[] yd = y.data().asDouble();
        final XYSeries pos = new XYSeries("y=1");
        final XYSeries neg = new XYSeries("y=0");
        for( int j=0; j<yd.length; j++ ) {
        	INDArray row = x.getRow(j);
        	if(yd[j] == 1.0){
        		pos.add(row.getDouble(0),row.getDouble(1));
        	}else{
        		neg.add(row.getDouble(0),row.getDouble(1));
        	}
        	
        }
        dataSet.addSeries(pos);
        dataSet.addSeries(neg);
    }
	
	private static void addSeries2(final XYSeriesCollection dataSet, final INDArray x, final INDArray y, final String label){
	       
        final double[] yd = y.data().asDouble();
        final XYSeries lr = new XYSeries("Decision Boundary");
       
        for( int j=0; j<yd.length; j++ ) {
        	lr.add(x.getRow(j).getDouble(0),y.getRow(j).getDouble(0));
        }
        dataSet.addSeries(lr);
        
    }
}
