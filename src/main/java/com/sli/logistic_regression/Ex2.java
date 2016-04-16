package com.sli.logistic_regression;

import java.awt.Color;
import java.awt.Shape;
import java.io.IOException;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Scanner;

import javax.swing.JFrame;
import javax.swing.WindowConstants;

import org.apache.commons.math3.analysis.DifferentiableMultivariateVectorFunction;
import org.apache.commons.math3.analysis.MultivariateMatrixFunction;
import org.apache.commons.math3.analysis.differentiation.DerivativeStructure;
import org.apache.commons.math3.analysis.differentiation.MultivariateDifferentiableVectorFunction;
import org.apache.commons.math3.exception.MathIllegalArgumentException;
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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.springframework.core.io.ClassPathResource;

import com.sli.deeplearning_experiment.ContourPlot;
import com.sli.deeplearning_experiment.CostMapper;
import com.sli.deeplearning_experiment.SurfacePlot;

public class Ex2 {
	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {

		System.out.println("Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n");

		// The first two columns contains the exam scores and the third column
		// contains the label.
		int totalSamples = 100;
		INDArray data = load("machine-learning/logistic_regression/ex2data1.txt", totalSamples);
		INDArray x = data.getColumns(0, 1).dup();
		INDArray y = data.getColumn(2).dup();

		// ==================== Part 1: Plotting ====================
		plotData(x, y);

		// ============ Part 2: Compute Cost and Gradient ============
		System.out.println("Running Gradient Descent ...\n");
		// Add intercept term to x and X_test
		INDArray X = Nd4j.hstack(Nd4j.ones(totalSamples, 1), x);

		// Compute and display initial cost and gradient
		int col = X.size(1);
		INDArray initialTheta = Nd4j.zeros(col, 1);
		Pair<Double, Double[]> result = costFunction(initialTheta, X, y);
		System.out.println("Cost at initial theta (zeros): " + result.getFirst() + "\n");
		System.out.println("Gradient at initial theta (zeros): " + Arrays.toString(result.getSecond()) + "\n");

		// ============= Part 3: Optimizing using fminunc =============
		double[] thetaFromFminunc = new double[] { -25.161272, 0.206233, 0.201470 };// TODO
																					// dsze
																					// need
																					// to
																					// find
																					// fminunc
																					// java
																					// implementation
																					// instead
																					// of
																					// cheating
																					// off
																					// of
																					// octave
		
		//double[] thetaFromFminunc = psudoFminunc(X, y, initialTheta); //computed values are { -20.787734661837316, 0.16952576514500547, 0.1694823440653863 };
		
		INDArray plotX = Nd4j.vstack(x.getColumn(0).min(0).sub(2), x.getColumn(0).max(0).add(2));// use
																									// 2
																									// to
																									// offset
																									// in
																									// case
																									// they
																									// have
																									// the
																									// same
																									// point
		INDArray plotY = plotX.mul(thetaFromFminunc[1]).add(thetaFromFminunc[0]).mul(-1.0 / thetaFromFminunc[2]);// (-1./theta(3)).*(theta(2).*plot_x
																													// +
																													// theta(1));
		plotData2(x, y, plotX, plotY);

	}

	private static class SigmoidProblem implements DifferentiableMultivariateVectorFunction, Serializable {
		INDArray X;
		INDArray y;

		public SigmoidProblem(final INDArray theX, final INDArray theY) {
			this.X = theX;
			this.y = theY;
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

	private static double[] psudoFminunc(INDArray theX, INDArray theY, INDArray initialTheta) {
		SigmoidProblem problem = new SigmoidProblem(theX, theY);
		LevenbergMarquardtOptimizer optimizer = new LevenbergMarquardtOptimizer();

		 
		 int m = theY.size(0);
		 final double[] weights = Nd4j.ones(m).data().asDouble();
		 
		 final double[] initialSolution = initialTheta.data().asDouble();

		 PointVectorValuePair optimum = optimizer.optimize(1000,
		                                                   problem,
		                                                   problem.calculateTarget(),
		                                                   weights,
		                                                   initialSolution);

		 final double[] optimalValues = optimum.getPoint();
		 System.out.println("A: " + optimalValues[0]);
		 System.out.println("B: " + optimalValues[1]);
		 System.out.println("C: " + optimalValues[2]);
		 return optimalValues;
	}

	private static Pair<Double, Double[]> costFunction(INDArray theta, INDArray X, INDArray y) {
		int m = y.size(0);
		double sum = 0.0;
		for (int i = 0; i < m; i++) {
			INDArray row = X.getRow(i);
			INDArray hypo = sigmoid(theta.transpose().mmul(row.transpose()));// hypo
																				// =
																				// sigmoid(theta'
																				// *
																				// X(i,:)')
			INDArray firstTerm = y.getRow(i).neg().mmul(Transforms.log(hypo));
			INDArray secondTermA = y.getRow(i).sub(1);
			INDArray secondTermB = Transforms.log(hypo.neg().add(1));
			INDArray secondTerm = secondTermA.mmul(secondTermB);
			sum += firstTerm.add(secondTerm).getDouble(0);// sum = sum -
															// y(i)*log(hypo) -
															// (1-y(i))*log(1-hypo)
		}
		double J = (double) sum / m;// cost

		Double[] grads = new Double[theta.size(0)];
		for (int j = 0; j < theta.size(0); j++) {
			sum = 0.0;

			for (int i = 0; i < m; i++) {
				INDArray row = X.getRow(i);
				INDArray hypo = sigmoid(theta.transpose().mmul(row.transpose()));// hypo
																					// =
																					// sigmoid(theta'
																					// *
																					// X(i,:)')
				INDArray firstTerm = hypo.sub(y.getRow(i));
				INDArray secondTerm = row.getColumn(j);
				sum += firstTerm.mmul(secondTerm).getDouble(0);// sum = sum +
																// (hypo-y(i))*X(i,j)
			}
			grads[j] = (double) sum / m;// derivatives
		}
		return new Pair<Double, Double[]>(J, grads);
	}

	private static INDArray sigmoid(INDArray z) {
		return Transforms.pow(Transforms.exp(z.neg()).add(1), -1, true);

	}

	private static void surf(INDArray X, INDArray y) throws Exception {
		// Define a function to plot
		CostMapper mapper = new CostMapper(X, y);

		// Define range and precision for the function to plot
		Range rangeX = new Range(-10, 10);
		int stepX = 100;
		Range rangeY = new Range(-1, 4);
		int stepY = 100;
		SurfacePlot sp = new SurfacePlot(rangeX, stepX, rangeY, stepY, mapper);
		sp.plot();
	}

	private static void contour(INDArray X, INDArray y, double theta0, double theta1) throws Exception {
		// Define a function to plot
		CostMapper mapper = new CostMapper(X, y);

		// Define range and precision for the function to plot
		Range rangeX = new Range(-10, 10);
		int stepX = 150;
		Range rangeY = new Range(-1, 4);
		int stepY = 150;
		ContourPlot sp = new ContourPlot(rangeX, stepX, rangeY, stepY, mapper, theta0, theta1);
		sp.plot();
	}

	private static INDArray[] gradientDescent(INDArray X, INDArray y, INDArray theta, double alpha, int iterations) {
		int m = y.size(0);
		INDArray jHistory = Nd4j.zeros(m, 1);
		for (int i = 0; i < iterations; i++) {
			INDArray secondTerm = X.mmul(theta).sub(y).mul((double) alpha / m);
			theta = theta.sub(X.transpose().mmul(secondTerm));
			jHistory.getRow(i).assign(computeCost(X, y, theta));
		}

		return new INDArray[] { theta, jHistory };
	}

	public static double computeCost(INDArray X, INDArray y, INDArray theta) {
		int m = y.size(0);
		return Transforms.pow(X.mmul(theta).sub(y), 2, true).sumNumber().doubleValue() / (2 * m);
	}

	private static INDArray load(String filePath, int totalSamples) throws IOException, InterruptedException {

		RecordReader recordReader = new CSVRecordReader();
		recordReader.initialize(new FileSplit(new ClassPathResource(filePath).getFile()));
		DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader);
		DataSet set = iterator.next(totalSamples);
		return set.getFeatureMatrix();
	}

	private static void warmUpExercise() {
		INDArray tmp = Nd4j.eye(5);
		System.out.println(tmp);
	}

	private static void pause() {
		System.out.println("Program paused. Press enter to continue.\n");
		Scanner scanner = new Scanner(System.in);
		scanner.nextLine();
	}

	private static JFrame plotData2(INDArray x, INDArray y, INDArray lrx, INDArray lry) {
		final XYSeriesCollection dataSet = new XYSeriesCollection();
		final XYSeriesCollection dataSet2 = new XYSeriesCollection();
		addSeries(dataSet, x, y, "Training Data");
		addSeries2(dataSet2, lrx, lry, "Logistic regression");

		final JFreeChart chart = ChartFactory.createScatterPlot(" ", // chart
																		// title
				"Population of City in 10,000s", // x axis label
				"Profit in $10,000s", // y axis label
				dataSet, // data
				PlotOrientation.VERTICAL, true, // include legend
				true, // tooltips
				false // urls
		);

		final ChartPanel panel = new ChartPanel(chart);
		XYPlot xyPlot = chart.getXYPlot();
		XYItemRenderer renderer1 = new XYLineAndShapeRenderer(false, true);// shapes
		Shape cross = ShapeUtilities.createDiagonalCross(3, 1);
		renderer1.setSeriesShape(0, cross);
		renderer1.setSeriesPaint(1, Color.yellow);
		xyPlot.setRenderer(0, renderer1);
		xyPlot.setDataset(0, dataSet);

		XYItemRenderer renderer2 = new XYLineAndShapeRenderer(true, false);// lines
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

	private static JFrame plotData(INDArray x, INDArray y) {
		final XYSeriesCollection dataSet = new XYSeriesCollection();
		addSeries(dataSet, x, y, "Training Data");

		final JFreeChart chart = ChartFactory.createScatterPlot(" ", // chart
																		// title
				"Exam 1 score", // x axis label
				"Exam 2 score", // y axis label
				dataSet, // data
				PlotOrientation.VERTICAL, true, // include legend
				true, // tooltips
				false // urls
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

	private static void addSeries(final XYSeriesCollection dataSet, final INDArray x, final INDArray y,
			final String label) {

		final double[] yd = y.data().asDouble();
		final XYSeries pos = new XYSeries("Admitted");
		final XYSeries neg = new XYSeries("Not admitted");
		for (int j = 0; j < yd.length; j++) {
			INDArray row = x.getRow(j);
			if (yd[j] == 1.0) {
				pos.add(row.getDouble(0), row.getDouble(1));
			} else {
				neg.add(row.getDouble(0), row.getDouble(1));
			}

		}
		dataSet.addSeries(pos);
		dataSet.addSeries(neg);
	}

	private static void addSeries2(final XYSeriesCollection dataSet, final INDArray x, final INDArray y,
			final String label) {

		final double[] yd = y.data().asDouble();
		final XYSeries lr = new XYSeries("Decision Boundary");

		for (int j = 0; j < yd.length; j++) {
			lr.add(x.getRow(j).getDouble(0), y.getRow(j).getDouble(0));
		}
		dataSet.addSeries(lr);

	}
}
