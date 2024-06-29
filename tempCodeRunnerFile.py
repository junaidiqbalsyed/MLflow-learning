
    # # now for every time a model is created have a mlflow run to track it
    # with mlflow.start_run(experiment_id=exp.experiment_id):
    #     lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    #     lr.fit(train_x, train_y)

    #     predicted_qualities = lr.predict(test_x)

    #     (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    #     print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
    #     print("  RMSE: %s" % rmse)
    #     print("  MAE: %s" % mae)
    #     print("  R2: %s" % r2)

    #     # log parameters
    #     mlflow.log_param("alpha", alpha)
    #     mlflow.log_param("l1_ratio", l1_ratio)

    #     # log metric
    #     mlflow.log_metric("RMSE", rmse)
    #     mlflow.log_metric("MAE", mae)
    #     mlflow.log_metric("R2", r2)

    #     # log the model
    #     mlflow.sklearn.log_model(lr, "elasticnet model")

    #     mlflow.end_run()