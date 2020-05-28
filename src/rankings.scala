import org.apache.spark.mllib.evaluation.{RankingMetrics, RegressionMetrics}
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel

val ratings = spark.read.textFile("interactions_0.01_train_clean.csv").rdd.map { line =>
  val fields = line.split(",")
  Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble)
}.cache()

val binarizedRatings = ratings.map(r => Rating(r.user, r.product,
  if (r.rating > 0) 1.0 else 0.0)
).cache()


Array(10).foreach { numIterations =>
  
  Array(10, 30, 50, 70, 90).foreach { rank =>

    Array(0.01, 0.05, 0.1).foreach { lambda => 
        val model = ALS.train(ratings, rank, numIterations, lambda)

        def scaledRating(r: Rating): Rating = {
          val scaledRating = math.max(math.min(r.rating, 1.0), 0.0)
          Rating(r.user, r.product, scaledRating)
        }

        val userRecommended = model.recommendProductsForUsers(500).map { case (user, recs) =>
          (user, recs.map(scaledRating))
        }

        val userBooks = binarizedRatings.groupBy(_.user)
        val relevantDocuments = userBooks.join(userRecommended).map { case (user, (actual,
        predictions)) =>
          (predictions.map(_.product), actual.filter(_.rating > 0.0).map(_.product).toArray)
        }

        println(s"$rank,$numIterations,$lambda")

        val metrics = new RankingMetrics(relevantDocuments)

        Array(100, 300, 500).foreach { k =>
          println(s"Precision at $k = ${metrics.precisionAt(k)}")
        }

        println(s"Mean average precision = ${metrics.meanAveragePrecision}")

        Array(100, 300, 500).foreach { k =>
          println(s"NDCG at $k = ${metrics.ndcgAt(k)}")
        }
    }
  }
}


Array(100, 300, 500).foreach { k =>
  println(s"NDCG at $k = ${metrics.ndcgAt(k)}")
}










import org.apache.spark.mllib.evaluation.{RankingMetrics, RegressionMetrics}
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel




val ratings = spark.read.textFile("hdfs:/user/ns4486/recommendations/data/interactions_0.1_train.csv").rdd.map { line =>
  val fields = line.split(",")
  Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble)
}.cache()

val binarizedRatings = ratings.map(r => Rating(r.user, r.product,
  if (r.rating > 0) 1.0 else 0.0)
).cache()





val model = ALS.train(ratings, rank, numIterations, lambda)

def scaledRating(r: Rating): Rating = {
  val scaledRating = math.max(math.min(r.rating, 1.0), 0.0)
  Rating(r.user, r.product, scaledRating)
}

val userRecommended = model.recommendProductsForUsers(500).map { case (user, recs) =>
  (user, recs.map(scaledRating))
}

val userBooks = binarizedRatings.groupBy(_.user)
val relevantDocuments = userBooks.join(userRecommended).map { case (user, (actual,
predictions)) =>
  (predictions.map(_.product), actual.filter(_.rating > 0.0).map(_.product).toArray)
}

println(s"$rank,$numIterations,$lambda")

val metrics = new RankingMetrics(relevantDocuments)

Array(100, 300, 500).foreach { k =>
  println(s"Precision at $k = ${metrics.precisionAt(k)}")
}

println(s"Mean average precision = ${metrics.meanAveragePrecision}")

Array(100, 300, 500).foreach { k =>
  println(s"NDCG at $k = ${metrics.ndcgAt(k)}")
}