package proj2;

import org.apache.spark.SparkConf;
import org.apache.spark.sql.SparkSession;

public class AppSession {

    public SparkSession sparkSession;

    public void setSparkSession(String master){
        this.sparkSession = SparkSession.builder()
                .appName("wine-app")
                .master(master)
                .config(new SparkConf())
                .getOrCreate();
    }

}
