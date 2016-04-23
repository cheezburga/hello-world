package com.sli.deeplearning_experiment;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args )
    {
//        INDArray tmp = Nd4j.create(new double[]{1,2,3,4}, new int[]{4,1});
//        System.out.println(tmp.reshape(2,2));
        
        INDArray nd2 = Nd4j.create(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}/*, new int[]{2, 6}*/);
        
        System.out.println(nd2.ordering());
        System.out.println(nd2);
        INDArray tmp2 = nd2.dup('f').reshape(3,4);
        System.out.println(tmp2);
        INDArray flat = tmp2.ravel();
        System.out.println(flat);
        System.out.println(flat.get(NDArrayIndex.interval(0, 4)).reshape(2, 2));
//        System.out.println(tmp2.max(0));
//        System.out.println("row:"+tmp2.max(0).rows()+" col:"+tmp2.max(0).columns());
//        System.out.println(tmp2.max(1));
//        System.out.println("row:"+tmp2.max(1).rows()+" col:"+tmp2.max(1).columns());
        
    }
}
