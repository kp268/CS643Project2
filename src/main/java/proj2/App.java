package proj2;


import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.sql.DataFrameReader;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.SparkSession;

public class App {

    public static void main(String [] args){

        App app = new App();
        app.run();

    }

    public void run(){
        AppSession appSession = new AppSession();
        appSession.setSparkSession("spark://0.0.0.0:7077");


        Dataset d = appSession.sparkSession.read()
                .option("header", "true")
                .option("delimiter", ";")
                .option("inferSchema", "true")
                .csv("TrainingDataset.csv");


        d.show();


    }



}
